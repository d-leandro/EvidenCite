"""Variantes del adaptador LoRA y su aplicación al modelo (celda 8)."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import (EvaConfig, LoraConfig, TaskType, VeraConfig, get_peft_model, initialize_lora_eva_weights,
                  prepare_model_for_kbit_training)

from configuracion import Config
from modelo import CABEZAS
from opciones_torch import contexto_autocast


@dataclass
class VarianteLoRA:
    """Describe una variante del adaptador: cómo construir su configuración de PEFT y qué trato especial necesita."""

    descripcion: str
    construir: object                 # función (modelo, cfg) -> configuración de PEFT
    optimizador: str = "adamw"        # adamw | lora_plus | lora_fa
    necesita_datos: bool = False      # EVA: se inicializa con activaciones de lotes de train
    muta_base: bool = False           # PiSSA / OLoRA modifican W; al guardar se convierte a LoRA estándar
    cuantizar_4bit: bool = False      # QLoRA: base en NF4 (solo CUDA + bitsandbytes)


def modulos_lineales(modelo, excluir=CABEZAS) -> list[str]:
    """Nombres (último tramo) de todas las capas lineales salvo la cabeza: el equivalente de 'all-linear'."""
    nombres = set()
    for nombre, modulo in modelo.named_modules():
        hoja = nombre.rsplit(".", 1)[-1]
        if hoja in excluir:
            continue
        if isinstance(modulo, torch.nn.Linear) or type(modulo).__name__ in ("Linear4bit", "Linear8bitLt"):
            nombres.add(hoja)
    return sorted(nombres)


def _modulos(modelo, cfg: Config) -> list[str]:
    """Capas donde va el adaptador: todas las lineales (LORA_MODULOS=auto) o la lista separada por comas."""
    if cfg.lora_modulos == "auto":
        return modulos_lineales(modelo)
    return [m.strip() for m in cfg.lora_modulos.split(",") if m.strip()]


def _lora(modelo, cfg: Config, **extra) -> LoraConfig:
    """LoraConfig de clasificación con r, alpha y dropout de la Config; `extra` activa la variante (DoRA, PiSSA...)."""
    return LoraConfig(task_type=TaskType.SEQ_CLS, r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                      lora_dropout=cfg.lora_dropout, target_modules=_modulos(modelo, cfg), bias="none", **extra)


def _eva(modelo, cfg: Config) -> LoraConfig:
    """LoraConfig de EVA: inicialización a partir de las activaciones (ver aplicar_lora)."""
    return _lora(modelo, cfg, init_lora_weights="eva", eva_config=EvaConfig(rho=2.0))


def _vera(modelo, cfg: Config):
    """VeraConfig: proyecciones aleatorias compartidas entre capas; solo se entrenan vectores de escala."""
    # El rango suele ser alto porque solo se entrenan vectores.
    return VeraConfig(task_type=TaskType.SEQ_CLS, r=256, target_modules=_modulos(modelo, cfg),
                      vera_dropout=cfg.lora_dropout, d_initial=0.1, save_projection=True)


VARIANTES_LORA = {
    "lora": VarianteLoRA("LoRA estándar: A aleatoria, B en cero", lambda m, c: _lora(m, c)),
    "rslora": VarianteLoRA("rsLoRA: escala α/√r en vez de α/r", lambda m, c: _lora(m, c, use_rslora=True)),
    "dora": VarianteLoRA("DoRA: separa magnitud y dirección de W", lambda m, c: _lora(m, c, use_dora=True)),
    "pissa": VarianteLoRA("PiSSA: inicia con las componentes principales de W (SVD rápida)",
                          lambda m, c: _lora(m, c, init_lora_weights="pissa_niter_4"), muta_base=True),
    "olora": VarianteLoRA("OLoRA: inicia con una base ortonormal (QR de W)",
                          lambda m, c: _lora(m, c, init_lora_weights="olora"), muta_base=True),
    "eva": VarianteLoRA("EVA: inicia con las direcciones de mayor varianza de las activaciones y reparte el rango",
                        _eva, necesita_datos=True),
    "lora_plus": VarianteLoRA("LoRA+: la tasa de B es 16 veces la de A", lambda m, c: _lora(m, c), optimizador="lora_plus"),
    "lora_fa": VarianteLoRA("LoRA-FA: congela A y entrena solo B", lambda m, c: _lora(m, c), optimizador="lora_fa"),
    "vera": VarianteLoRA("VeRA: proyecciones aleatorias compartidas; solo entrena vectores de escala", _vera),
    "qlora": VarianteLoRA("QLoRA: base cuantizada a 4 bits (NF4) + LoRA", lambda m, c: _lora(m, c), cuantizar_4bit=True),
}


def listar_variantes() -> None:
    """Imprime las variantes disponibles con sus particularidades."""
    for nombre, v in VARIANTES_LORA.items():
        notas = [n for n, ok in (("usa lotes de train al iniciar", v.necesita_datos), ("modifica W", v.muta_base),
                                 ("optimizador propio", v.optimizador != "adamw"), ("solo CUDA", v.cuantizar_4bit)) if ok]
        print(f"  {nombre:<10}{v.descripcion}" + (f"  [{', '.join(notas)}]" if notas else ""))


def resolver_variante(cfg: Config, opt: dict):
    """Devuelve la variante pedida en LORA_TIPO y una lista de avisos.

    QLoRA necesita GPU NVIDIA y bitsandbytes; si falta algo, se degrada a LoRA estándar sin cuantizar y lo avisa.
    """
    if cfg.lora_tipo not in VARIANTES_LORA:
        raise KeyError(f"LORA_TIPO='{cfg.lora_tipo}' no existe. Opciones: {', '.join(VARIANTES_LORA)}")
    var, avisos = VARIANTES_LORA[cfg.lora_tipo], []
    if var.cuantizar_4bit and not opt["cuant_4bit"]:
        avisos.append(f"qlora desactivada ({opt['motivos']['cuant_4bit']}); se entrena LoRA estándar sin cuantizar")
        var = dataclasses.replace(var, cuantizar_4bit=False, descripcion=var.descripcion + " · degradada a LoRA")
    return var, avisos


def asegurar_entrenables_fp32(modelo) -> None:
    """Pasa a fp32 los pesos que se optimizan (la cabeza nueva hereda el dtype de la base)."""
    for p in modelo.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()


def contar_parametros(modelo) -> tuple[int, int]:
    """Devuelve (parámetros entrenables, parámetros totales)."""
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return entrenables, sum(p.numel() for p in modelo.parameters())


def aplicar_lora(modelo, cfg: Config, var: VarianteLoRA, opt: dict, lotes_init=None, dir_init: Path | None = None):
    """Envuelve el modelo con el adaptador de la variante y devuelve (modelo PEFT, información del adaptador).

    - QLoRA: prepara la base cuantizada para entrenar.
    - EVA: inicializa el adaptador con las activaciones de `lotes_init`.
    - PiSSA / OLoRA: guarda el adaptador inicial en `dir_init` para convertirlo a LoRA estándar al final.
    """
    if var.cuantizar_4bit:
        modelo = prepare_model_for_kbit_training(modelo, use_gradient_checkpointing=cfg.grad_checkpointing)
    config_peft = var.construir(modelo, cfg)
    modelo = get_peft_model(modelo, config_peft, low_cpu_mem_usage=var.necesita_datos)  # EVA lo recomienda
    if var.necesita_datos:
        if not lotes_init:
            raise ValueError(f"{cfg.lora_tipo} necesita lotes de entrenamiento para inicializarse")
        with contexto_autocast(opt):
            initialize_lora_eva_weights(modelo, dataloader=lotes_init, show_progress_bar=False)
    if var.muta_base and dir_init is not None:
        modelo.save_pretrained(dir_init)  # adaptador inicial: permite convertir a LoRA estándar al guardar
    asegurar_entrenables_fp32(modelo)
    entrenables, totales = contar_parametros(modelo)
    info = {"tipo": cfg.lora_tipo, "descripcion": var.descripcion, "modulos": sorted(config_peft.target_modules),
            "entrenables": entrenables, "totales": totales, "porcentaje": 100.0 * entrenables / totales}
    print(f"  {var.descripcion}")
    print(f"  módulos: {', '.join(info['modulos'])}")
    print(f"  parámetros entrenables: {entrenables:,} de {totales:,} ({info['porcentaje']:.2f} %)")
    return modelo, info
