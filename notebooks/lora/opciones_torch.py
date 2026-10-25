"""Opciones de PyTorch: hardware disponible y decisión de precisión, TF32, torch.compile, AdamW fusionado,
pin_memory y QLoRA según lo que soporte la máquina (celda 3).
"""

import contextlib
import os
import platform
import shutil

import torch

from configuracion import Config

# Dependencias opcionales: si faltan, el programa sigue y desactiva lo que dependa de ellas.
try:
    import triton  # torch.compile en GPU
except ImportError:
    triton = None
try:
    import bitsandbytes  # QLoRA
    ERROR_BITSANDBYTES = None
except Exception as e:  # noqa: BLE001  (ImportError, o RuntimeError si no encuentra sus binarios)
    bitsandbytes, ERROR_BITSANDBYTES = None, f"{type(e).__name__}: {e}"


def detectar_hardware(colab: bool, cuda: bool) -> dict:
    """Describe la máquina: sistema, GPU y su capacidad de cómputo (cc), memoria y librerías opcionales disponibles."""
    hw = {
        "sistema": platform.system(),
        "colab": colab,
        "nucleos_cpu": os.cpu_count(),
        "dispositivo": "cpu",
        "gpu": None,
        "cc": None,
        "vram_gb": None,
        "triton": triton is not None,
        "compilador_c": any(shutil.which(c) for c in ("gcc", "cc", "clang")),
        "bitsandbytes": bitsandbytes is not None,
    }
    if cuda:
        p = torch.cuda.get_device_properties(0)
        hw.update(dispositivo="cuda", gpu=p.name, cc=list(torch.cuda.get_device_capability(0)),
                  vram_gb=round(p.total_memory / 1024**3, 1), n_gpus=torch.cuda.device_count())
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        hw["dispositivo"] = "mps"
    return hw


def decidir_optimizaciones(hw: dict, cfg: Config) -> dict:
    """Decide cada optimización según el hardware y la Config. Devuelve la decisión y el motivo de cada una."""
    cuda = hw["dispositivo"] == "cuda"
    cc = tuple(hw["cc"] or (0, 0))
    ampere = cuda and cc >= (8, 0)  # A100 (8.0), L4 (8.9), H100 (9.0), G4 / RTX PRO 6000 (12.0)
    motivos = {}

    # Precisión mixta
    if cfg.precision == "auto":
        precision = "bf16" if ampere else ("fp16" if cuda else "fp32")
        motivos["precision"] = ("bf16 nativo (cc ≥ 8.0)" if ampere
                                else "GPU sin bf16 nativo: fp16 con GradScaler" if cuda
                                else f"{hw['dispositivo'].upper()}: fp32")
    else:
        precision = cfg.precision
        motivos["precision"] = "fijada por configuración"
        if precision == "bf16" and not ampere:
            precision = "fp16" if cuda else "fp32"
            motivos["precision"] = f"bf16 pedido pero la GPU no lo soporta: {precision}"
        elif precision == "fp16" and not cuda:
            precision = "fp32"
            motivos["precision"] = "fp16 pedido sin CUDA: fp32"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

    # TF32 para las multiplicaciones que quedan en fp32
    tf32 = ampere
    motivos["tf32"] = "multiplicaciones fp32 en tensor cores" if tf32 else "requiere GPU con cc ≥ 8.0"

    # torch.compile (inductor + triton)
    requisitos = (("CUDA", cuda), ("triton", hw["triton"]), ("compilador C", hw["compilador_c"]),
                  ("Linux", hw["sistema"] == "Linux"), ("cc ≥ 7.0", not cuda or cc >= (7, 0)))
    faltan = [nombre for nombre, ok in requisitos if not ok]
    if cfg.compilar == "no":
        compilar, motivos["compilar"] = False, "desactivado por configuración"
    elif faltan:
        compilar, motivos["compilar"] = False, "no soportado aquí: falta " + ", ".join(faltan)
    else:
        compilar, motivos["compilar"] = True, "inductor + triton; si falla, sigue sin compilar"

    motivos["adamw_fusionado"] = "kernel único de CUDA para todos los parámetros" if cuda else "solo en CUDA"
    motivos["pin_memory"] = "copias asíncronas CPU → GPU" if cuda else "solo en CUDA"
    motivos["multiplo_padding"] = ("64: menos formas distintas que recompilar" if compilar
                                   else "8: alinea con tensor cores")
    motivos["atencion"] = ("SDPA: PyTorch elige flash, memory-efficient o cuDNN según la GPU" if cfg.atencion == "sdpa"
                           else "fijada por configuración")
    cuant = cuda and hw["bitsandbytes"]
    motivos["cuant_4bit"] = ("disponible para LORA_TIPO=qlora" if cuant
                             else "requiere GPU NVIDIA" if hw["bitsandbytes"]
                             else f"requiere bitsandbytes ({(ERROR_BITSANDBYTES or 'no está instalado')[:120]})")
    return {
        "dispositivo": hw["dispositivo"], "precision": precision, "dtype": dtype,
        "autocast": precision != "fp32", "escalador": precision == "fp16", "tf32": tf32,
        "compilar": compilar, "adamw_fusionado": cuda, "pin_memory": cuda, "non_blocking": cuda,
        "multiplo_padding": 64 if compilar else 8, "atencion": cfg.atencion, "cuant_4bit": cuant,
        "motivos": motivos,
    }


def aplicar_optimizaciones_globales(opt: dict) -> None:
    """Aplica los ajustes globales de torch: TF32 en GPUs con cc ≥ 8.0 y más recompilaciones permitidas al compilar.

    TF32 se activa con set_float32_matmul_precision, que mantiene coherentes la API clásica y la nueva de torch ≥ 2.9.
    Fijar solo la nueva (fp32_precision) deja a torch.get_float32_matmul_precision() lanzando un error, y
    torch.compile lo consulta al autoajustar multiplicaciones.
    """
    if opt["tf32"]:
        torch.set_float32_matmul_precision("high")
    if opt["compilar"]:
        # Las longitudes de lote varían: se permiten más recompilaciones antes de volver a eager.
        dynamo = torch._dynamo.config
        limite = "recompile_limit" if hasattr(dynamo, "recompile_limit") else "cache_size_limit"  # nombre previo a torch 2.7
        setattr(dynamo, limite, max(getattr(dynamo, limite), 32))


def contexto_autocast(opt: dict):
    """Devuelve el contexto de precisión mixta (bf16 o fp16) del dispositivo, o uno vacío en fp32."""
    if opt["autocast"]:
        return torch.autocast(device_type=opt["dispositivo"], dtype=opt["dtype"])
    return contextlib.nullcontext()


def imprimir_decisiones(hw: dict, opt: dict) -> None:
    """Imprime el hardware detectado y, para cada optimización, si quedó activa y por qué."""
    if hw["dispositivo"] == "cuda":
        print(f"  {hw['gpu']} · cc {hw['cc'][0]}.{hw['cc'][1]} · {hw['vram_gb']} GB · {hw['sistema']}"
              + (" · Colab" if hw["colab"] else ""))
    else:
        print(f"  {hw['dispositivo'].upper()} · {hw['nucleos_cpu']} núcleos · {hw['sistema']}"
              + (" · Colab" if hw["colab"] else ""))
    filas = (("precisión", opt["precision"], "precision"), ("TF32", opt["tf32"], "tf32"),
             ("torch.compile", opt["compilar"], "compilar"), ("AdamW fusionado", opt["adamw_fusionado"], "adamw_fusionado"),
             ("pin_memory", opt["pin_memory"], "pin_memory"), ("padding múltiplo", opt["multiplo_padding"], "multiplo_padding"),
             ("atención", opt["atencion"], "atencion"), ("QLoRA 4-bit", opt["cuant_4bit"], "cuant_4bit"))
    for nombre, valor, clave in filas:
        valor = {True: "sí", False: "no"}.get(valor, valor)
        print(f"  {nombre:<18}{str(valor):<7}{opt['motivos'][clave]}")
