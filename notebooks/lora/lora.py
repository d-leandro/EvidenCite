"""Baseline LoRA · función de la cita (rama LLM): programa principal.

Entrena un clasificador de las 9 funciones de cita sobre un LLM decoder con un adaptador LoRA intercambiable, elige el
mejor punto del entrenamiento por macro-F1 de dev (early stopping) y lo evalúa. Es baseline_lora.ipynb repartido en
módulos, con las mismas funciones:

- entorno.py: versiones de las librerías, token de Hugging Face y detección de Colab.
- configuracion.py: Config, que se arma con variables de entorno (MODO, MODELO, LORA_TIPO, LR...) y CONFIG_JSON.
- opciones_torch.py: hardware y optimizaciones de PyTorch (precisión, TF32, torch.compile, AdamW fusionado, QLoRA).
- preprocesamiento.py: datos, texto de entrada, tokenización y lotes.
- modelo.py, variantes_lora.py, evaluacion.py, entrenamiento.py, guardado.py y registro_mlflow.py.

Aquí están la orquestación (ejecutar_experimento, probar_variantes) y el bloque principal. Ese bloque calcula una
sola vez lo que depende de dónde corre el programa (Colab, CUDA, hardware, versiones y token de Hugging Face) y lo
pasa como parámetro a las funciones que lo usan.

Uso: MODO=mini python lora.py. Con PROBAR_VARIANTES=1 prueba además 2 pasos con cada variante de LoRA.
"""
import os
import sys
import warnings

# Variables que deben fijarse antes de importar torch, transformers y los módulos del proyecto.
# expandable_segments reduce la fragmentación de la memoria de CUDA. Se usa el nombre clásico de la variable
# porque lo leen todas las versiones de torch (PYTORCH_ALLOC_CONF solo existe desde la 2.9).
if sys.platform == "linux" and not any(v in os.environ for v in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # las barras con \r ensucian la salida de la CLI
# Cliente de MLflow (solo con MLFLOW_IP): pocos reintentos y espera corta por pedido. Si el servidor cae, quien
# reintenta es RegistroMLflow (registro_mlflow.py), en segundo plano y sin frenar el entrenamiento.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")  # mlflow 3 manda estadísticas de uso a sus servidores
# Jupyter local sin ipywidgets: tqdm avisa al importarse aunque el notebook no muestra barras.
warnings.filterwarnings("ignore", message="IProgress not found")

# El resto de los imports va después de fijar las variables de entorno (por eso no están arriba).
import contextlib
import gc
import itertools
import time
from dataclasses import asdict
from pathlib import Path

import torch
import transformers

from configuracion import Config, construir_config, imprimir_config
from entorno import configurar_hf_token, en_colab, titulo, verificar_versiones
from entrenamiento import (compile_model, compute_class_weights, compute_num_training_steps, compute_steps_per_epoch,
                           create_optimizer, create_scheduler, set_seed, train)
from evaluacion import evaluate, print_metrics
from guardado import guardar_corrida
from modelo import crear_modelo, verificar_pooling
from opciones_torch import (aplicar_optimizaciones_globales, decidir_optimizaciones, detectar_hardware,
                            imprimir_decisiones)
from preprocesamiento import cargar_datos, crear_dataloader, mover, partes_texto, resumen_tokens, tokenizar_split
from registro_mlflow import RegistroMLflow, abrir_mlflow, cerrar_mlflow
from variantes_lora import VARIANTES_LORA, aplicar_lora, listar_variantes, resolver_variante

transformers.utils.logging.disable_progress_bar()


def liberar_memoria(cuda: bool) -> None:
    """Libera la memoria de corridas anteriores: recolector de basura, caché de CUDA y gráficos compilados."""
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    with contextlib.suppress(Exception):
        torch._dynamo.reset()


def ejecutar_experimento(cfg: Config | None = None, devolver_modelo: bool = True, *, colab: bool, cuda: bool, hw: dict,
                         versiones: dict, token_hf: str | None, **cambios) -> dict:
    """Corre un experimento completo. Acepta una Config o cambios sueltos: ejecutar_experimento(lora_tipo="dora", ...).

    colab, cuda, hw, versiones y token_hf son el contexto de ejecución: el bloque principal de lora.py los calcula
    una sola vez y los pasa a cada corrida.

    Etapas: configuración, hardware, datos, modelo, tokenización, adaptador, entrenamiento con early stopping,
    evaluación con los mejores pesos y guardado. Devuelve la Config, el resumen, las métricas y predicciones por
    split, la ruta del zip y, si devolver_modelo=True, el modelo y el tokenizador.

    Para encadenar corridas con modelos grandes, borra antes el resultado anterior (`del resultado`) o usa
    devolver_modelo=False: si no, el modelo previo sigue ocupando memoria de la GPU.

    Con MLFLOW_IP también registra la corrida en MLflow mientras entrena (registro_mlflow.py). Si el servidor falla, el
    entrenamiento sigue y los resultados locales se guardan igual. Si la corrida falla, queda en FAILED en MLflow.
    """
    cfg = cfg or construir_config(colab, cuda, **cambios)
    registro = abrir_mlflow(cfg, colab, versiones)
    try:
        salida = _ejecutar_experimento(cfg, devolver_modelo, registro, cuda, hw, versiones, token_hf)
    except BaseException as e:
        if registro is not None:
            registro.cerrar("KILLED" if isinstance(e, KeyboardInterrupt) else "FAILED", esperar=False)
        raise
    if registro is not None:
        cerrar_mlflow(registro, cfg, salida)
    return salida


def _ejecutar_experimento(cfg: Config, devolver_modelo: bool, registro: RegistroMLflow | None, cuda: bool, hw: dict,
                          versiones: dict, token_hf: str | None) -> dict:
    """Las etapas de ejecutar_experimento con la Config ya armada. `registro` (o None) recibe las curvas del entrenamiento."""
    liberar_memoria(cuda)
    t_inicio = time.perf_counter()
    titulo("Configuración")
    imprimir_config(cfg)
    set_seed(cfg.semilla)

    titulo("Hardware y optimizaciones")
    opt = decidir_optimizaciones(hw, cfg)
    aplicar_optimizaciones_globales(opt)
    imprimir_decisiones(hw, opt)
    var, avisos = resolver_variante(cfg, opt)
    for a in avisos:
        print(f"AVISO: {a}")

    titulo("Datos")
    datos = cargar_datos(cfg)

    titulo("Modelo")
    textos_humo = [" ".join(partes_texto(f, cfg)) for f in datos["train"]] if cfg.modelo == "humo" else None
    modelo, tok, ruta_modelo = crear_modelo(cfg, opt, var, textos_humo, token_hf)
    print(f"  base: {ruta_modelo}")

    titulo("Tokenización")
    splits = ["train", "dev"] + (["test"] if cfg.evaluar_test else [])
    items = {s: tokenizar_split(datos[s], tok, cfg) for s in splits}
    for s in splits:
        r = resumen_tokens(items[s])
        print(f"  {s:<6}tokens media {r['media']:.0f} · p95 {r['p95']:.0f} · máx {r['max']} · "
              f"truncados {r['truncados_pct']:.1f} % (máx {cfg.max_longitud})")
    dl_train = crear_dataloader(items["train"], cfg.batch, True, cfg, opt, tok.pad_token_id)
    dl_eval = {s: crear_dataloader(items[s], cfg.batch_eval, False, cfg, opt, tok.pad_token_id) for s in splits[1:]}

    titulo("Adaptador")
    lotes_init = None
    if var.necesita_datos:
        lotes_init = [{k: v for k, v in mover(l, opt).items() if k in ("input_ids", "attention_mask")}
                      for l in itertools.islice(dl_train, 8)]
    dir_init = Path(cfg.salidas_dir) / "_tmp" / cfg.nombre_corrida / "adaptador_inicial" if var.muta_base else None
    modelo, info_lora = aplicar_lora(modelo, cfg, var, opt, lotes_init, dir_init)

    titulo("Entrenamiento")
    class_weights = compute_class_weights(items["train"], cfg.pesos_clase, opt)
    num_training_steps = compute_num_training_steps(dl_train, cfg)
    optimizer = create_optimizer(modelo, var, cfg, opt)
    scheduler = create_scheduler(optimizer, num_training_steps, cfg)
    lote0 = mover(next(iter(dl_train)), opt)
    verificar_pooling(lote0, tok.pad_token_id)
    modelo_fw, info_compilacion = compile_model(modelo, lote0, cfg, opt, class_weights)
    print(f"  {num_training_steps} pasos · lote {cfg.batch} × acumulación {cfg.acumulacion} · {len(items['train'])} ejemplos · "
          f"dev {cfg.evals_por_epoca} veces por época · paciencia {cfg.paciencia or 'sin límite'}")
    entrenamiento = train(modelo, modelo_fw, dl_train, dl_eval["dev"], optimizer, scheduler, num_training_steps,
                          cfg, opt, class_weights, info_compilacion,
                          log_callback=registro.registrar_fila if registro is not None else None)
    print(f"  {entrenamiento['segundos']:.1f} s ({entrenamiento['segundos_eval_dev']:.1f} s evaluando dev) · "
          f"{entrenamiento['ejemplos_por_s']:.1f} ej/s · {entrenamiento['tokens_por_s']:.0f} tokens/s"
          + (f" · memoria pico {entrenamiento['memoria_pico_gb']:.2f} GB" if entrenamiento["memoria_pico_gb"] else ""))

    titulo("Evaluación (pesos del mejor paso)")
    n_bootstrap = 1000 if cfg.modo == "completo" else 200
    resultados = {}
    for s, dl in dl_eval.items():
        metricas, predicciones = evaluate(modelo, dl, items[s], opt, n_bootstrap)
        print_metrics(metricas, s)
        resultados[s] = {"metricas": metricas, "predicciones": predicciones}

    opt_json = {k: v for k, v in opt.items() if k != "dtype"} | {"dtype": str(opt["dtype"])}
    seleccion = {"criterio": "macro-F1 de dev", "evals_por_epoca": cfg.evals_por_epoca, "paciencia": cfg.paciencia,
                 "delta_minimo": cfg.delta_minimo, "mejor_paso": entrenamiento["mejor_paso"],
                 "mejor_macro_f1_dev": entrenamiento["mejor_macro_f1_dev"],
                 "mejor_epoca": round(entrenamiento["mejor_paso"] / compute_steps_per_epoch(dl_train, cfg), 2),
                 "parada_temprana": entrenamiento["parada_temprana"], "pasos_entrenados": entrenamiento["pasos"],
                 "pasos_planeados": entrenamiento["pasos_planeados"]}
    resumen = {
        "corrida": cfg.nombre_corrida, "modo": cfg.modo, "modelo": ruta_modelo if cfg.modelo != "humo" else "humo",
        "lora": info_lora, "configuracion": cfg.configuracion, "avisos": avisos,
        "metricas": {s: {k: r["metricas"][k] for k in ("macro_f1", "macro_f1_ic95", "f1_macro_9", "f1_macro_5",
                                                        "f1_weighted", "accuracy", "ece", "n")}
                     for s, r in resultados.items()},
        "seleccion": seleccion,
        "entrenamiento": {k: v for k, v in entrenamiento.items() if k not in ("registro", "evaluaciones")},
        "compilacion": info_compilacion, "precision": opt["precision"], "hardware": hw,
        "segundos_totales": time.perf_counter() - t_inicio,
    }
    zip_corrida = None
    if cfg.guardar:
        archivos = {"config.json": asdict(cfg), "resumen.json": resumen, "optimizaciones.json": opt_json,
                    "versiones.json": versiones, "registro_entrenamiento.json": entrenamiento["registro"],
                    "evaluaciones_dev.json": entrenamiento["evaluaciones"]}
        for s, r in resultados.items():
            archivos[f"metricas_{s}.json"] = r["metricas"]
            archivos[f"predicciones_{s}.jsonl"] = r["predicciones"]
        zip_corrida = guardar_corrida(cfg, modelo, tok, var, dir_init, archivos)

    titulo("Resumen")
    dev = resumen["metricas"]["dev"]
    print(f"  {cfg.nombre_corrida}")
    print(f"  macro-F1 dev {dev['macro_f1']:.4f} · accuracy {dev['accuracy']:.4f} · "
          f"{info_lora['entrenables']:,} parámetros entrenables ({info_lora['porcentaje']:.2f} %)")
    for s, x in resumen["metricas"].items():
        print(f"  {s}: F1-macro-9 {x['f1_macro_9']:.4f} · F1-macro-5 {x['f1_macro_5']:.4f} · F1 ponderado {x['f1_weighted']:.4f}")
    print(f"  mejor paso {seleccion['mejor_paso']} (época {seleccion['mejor_epoca']}) de {seleccion['pasos_entrenados']} "
          "entrenados" + (" (early stopping)" if seleccion["parada_temprana"] else ""))
    print(f"  {opt['precision']} · compilado: {'sí' if info_compilacion['compilado'] else 'no'} · "
          f"{resumen['segundos_totales']:.0f} s en total")
    if zip_corrida:
        print(f"  resultados: {zip_corrida} (copia en {Path(cfg.salidas_dir) / 'ultima_corrida.zip'})")
    salida = {"cfg": cfg, "resumen": resumen, "resultados": resultados, "zip": zip_corrida}
    if devolver_modelo:
        salida |= {"modelo": modelo, "tokenizer": tok}
    return salida


def probar_variantes(variantes=None, *, colab: bool, cuda: bool, hw: dict, versiones: dict, token_hf: str | None,
                     **cambios) -> list[dict]:
    """Entrena 2 pasos en modo mini con cada variante, guarda y evalúa. Sirve para verificar el hardware local."""
    tabla = []
    base = Path(construir_config(colab, cuda, modo="mini").salidas_dir) / "pruebas_variantes"
    for nombre in variantes or list(VARIANTES_LORA):
        t0 = time.perf_counter()
        try:
            r = ejecutar_experimento(modo="mini", lora_tipo=nombre, max_pasos=2, mini_train=32, mini_dev=16,
                                     salidas_dir=str(base), nombre_corrida=f"prueba-{nombre}",
                                     devolver_modelo=False, mlflow_ip="", colab=colab, cuda=cuda, hw=hw,
                                     versiones=versiones, token_hf=token_hf, **cambios)  # sin MLflow
            estado = f"ok · macro-F1 dev {r['resumen']['metricas']['dev']['macro_f1']:.3f}"
            if r["resumen"]["avisos"]:
                estado += " · " + "; ".join(r["resumen"]["avisos"])
            del r
        except Exception as e:  # noqa: BLE001
            estado = f"ERROR {type(e).__name__}: {str(e)[:200]}"
        tabla.append({"variante": nombre, "estado": estado, "segundos": round(time.perf_counter() - t0, 1)})
        liberar_memoria(cuda)
    titulo("Prueba de variantes")
    for fila in tabla:
        print(f"  {fila['variante']:<10}{fila['segundos']:>7.1f} s  {fila['estado']}")
    return tabla


if __name__ == "__main__":
    # Contexto de ejecución: lo que depende de dónde corre el programa se calcula una sola vez, aquí, y se pasa
    # como parámetro a las funciones que lo usan.
    COLAB = en_colab()
    CUDA = torch.cuda.is_available()
    HW = detectar_hardware(COLAB, CUDA)

    titulo("Entorno")
    VERSIONES = verificar_versiones()
    ORIGEN_HF_TOKEN = configurar_hf_token(COLAB)
    TOKEN_HF = os.environ.get("HF_TOKEN") or None  # configurar_hf_token lo deja en esa variable; nunca se imprime
    listar_variantes()

    # Celda 13 del notebook: el experimento, con la Config de las variables de entorno
    resultado = ejecutar_experimento(colab=COLAB, cuda=CUDA, hw=HW, versiones=VERSIONES, token_hf=TOKEN_HF)

    # Celda 14: todas las variantes (opcional, PROBAR_VARIANTES=1)
    if os.environ.get("PROBAR_VARIANTES", "0") == "1":
        tabla_variantes = probar_variantes(colab=COLAB, cuda=CUDA, hw=HW, versiones=VERSIONES, token_hf=TOKEN_HF)
    else:
        print("Omitido. Con PROBAR_VARIANTES=1 se entrena 2 pasos en modo mini con cada variante de LoRA.")
