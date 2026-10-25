"""Entrenamiento: optimizador, planificador de la tasa, early stopping, torch.compile y el bucle de
entrenamiento (celda 10).
"""

import math
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from peft.optimizers import create_lorafa_optimizer, create_loraplus_optimizer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm  # barra de texto: se ve en Jupyter, en Colab y en la terminal de la CLI
from transformers import (get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup,
                          get_cosine_with_min_lr_schedule_with_warmup, get_linear_schedule_with_warmup,
                          get_wsd_schedule)

from configuracion import Config
from evaluacion import validate
from opciones_torch import contexto_autocast
from preprocesamiento import ETIQUETAS, mover
from variantes_lora import VarianteLoRA


def set_seed(seed: int) -> None:
    """Fija las semillas de random, numpy y torch (CPU y todas las GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(items: list[dict], scheme: str, opt: dict):
    """Pesos por clase para la pérdida: None (ninguno), inverso de la frecuencia o su raíz cuadrada."""
    if scheme == "ninguno":
        return None
    counts = np.bincount([x["etiqueta"] for x in items], minlength=len(ETIQUETAS)).astype(float)
    weights = np.where(counts > 0, counts.sum() / (len(ETIQUETAS) * np.maximum(counts, 1)), 0.0)
    if scheme == "raiz_inversa":
        weights = np.sqrt(weights)
    return torch.tensor(weights, dtype=torch.float32, device=opt["dispositivo"])


def compute_loss(logits, labels, class_weights=None):
    """Entropía cruzada en fp32, con pesos por clase opcionales."""
    return F.cross_entropy(logits.float(), labels, weight=class_weights)


def create_optimizer(model, variant: VarianteLoRA, cfg: Config, opt: dict):
    """Crea el optimizador: AdamW (fusionado en CUDA), o el propio de LoRA+ o LoRA-FA según la variante."""
    extra = {"fused": True} if opt["adamw_fusionado"] else {}
    if variant.optimizador == "lora_plus":
        return create_loraplus_optimizer(model=model, optimizer_cls=torch.optim.AdamW, lr=cfg.lr,
                                         loraplus_lr_ratio=16.0, weight_decay=cfg.weight_decay, **extra)
    if variant.optimizador == "lora_fa":
        return create_lorafa_optimizer(model=model, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lr=cfg.lr,
                                       weight_decay=cfg.weight_decay)
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, **extra)


def build_scheduler(name: str, optimizer: Optimizer, warmup_steps: int, total_steps: int) -> LRScheduler:
    """Crea el planificador de la tasa de aprendizaje `name`. Todos empiezan con `warmup_steps` de calentamiento lineal.

    - linear: decae en línea recta hasta 0 (BERT, RoBERTa con LoRA; Mosbach et al., 2021).
    - cosine: decae hasta 0 siguiendo medio coseno (Loshchilov y Hutter, 2017).
    - cosine_min: medio coseno, pero solo hasta el 10 % de la tasa (LLaMA; Touvron et al., 2023).
    - constant: se queda constante (QLoRA, Dettmers et al., 2023; Pareja et al., 2024). No depende de cuántos pasos se
      planearon, así que encaja con el early stopping.
    - wsd: constante y decae en el 20 % final con forma 1-sqrt (MiniCPM, Hu et al., 2024; Hägele et al., 2024).
    - plateau: constante; train() la baja a la mitad cada vez que el macro-F1 de dev no mejora.
    """
    if name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    if name == "cosine_min":
        return get_cosine_with_min_lr_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                           num_training_steps=total_steps, min_lr_rate=0.1)
    if name in ("constant", "plateau"):
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    if name == "wsd":
        return get_wsd_schedule(optimizer, num_warmup_steps=warmup_steps, num_decay_steps=max(1, int(0.2 * total_steps)),
                                num_training_steps=total_steps, decay_type="1-sqrt")
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)


def create_scheduler(optimizer, num_training_steps: int, cfg: Config):
    """Planificador elegido en SCHEDULER, con calentamiento durante la fracción `warmup` de los pasos."""
    return build_scheduler(cfg.scheduler, optimizer, max(1, int(cfg.warmup * num_training_steps)), num_training_steps)


def compute_steps_per_epoch(train_loader, cfg: Config) -> int:
    """Pasos de optimizador por época: lotes por época divididos por la acumulación, redondeando hacia arriba."""
    return math.ceil(len(train_loader) / cfg.acumulacion)


def compute_num_training_steps(train_loader, cfg: Config) -> int:
    """Pasos de optimizador totales: pasos por época × épocas, acotado por max_pasos si está definido."""
    steps = compute_steps_per_epoch(train_loader, cfg) * cfg.epocas
    return min(steps, cfg.max_pasos) if cfg.max_pasos else steps


def compute_eval_steps(steps_per_epoch: int, num_training_steps: int, evals_per_epoch: int) -> set[int]:
    """Pasos de optimizador en los que se evalúa dev: `evals_per_epoch` puntos repartidos en cada época, más el último paso."""
    steps = {num_training_steps}
    epoch = 0
    while epoch * steps_per_epoch < num_training_steps:
        for k in range(1, evals_per_epoch + 1):
            step = epoch * steps_per_epoch + max(1, round(k * steps_per_epoch / evals_per_epoch))
            if step <= num_training_steps:
                steps.add(step)
        epoch += 1
    return steps


@dataclass
class EarlyStopping:
    """Sigue el macro-F1 de dev y decide cuándo parar.

    Una evaluación cuenta como mejora si supera al mejor valor en más de `min_delta`. Tras `patience` evaluaciones
    seguidas sin mejora, `should_stop` pasa a True. Con `patience=0` nunca para, pero sigue registrando el mejor paso.
    """

    patience: int
    min_delta: float = 0.0
    best_score: float = -math.inf
    best_step: int = 0
    num_bad_evals: int = 0

    def step(self, score: float, step: int) -> bool:
        """Registra el macro-F1 de una evaluación hecha en el paso `step`. Devuelve True si es el mejor hasta ahora."""
        if score > self.best_score + self.min_delta:
            self.best_score, self.best_step, self.num_bad_evals = score, step, 0
            return True
        self.num_bad_evals += 1
        return False

    @property
    def should_stop(self) -> bool:
        """True cuando se agotó la paciencia."""
        return self.patience > 0 and self.num_bad_evals >= self.patience


def get_trainable_state(model) -> dict:
    """Copia en CPU de los parámetros entrenables (adaptador y cabeza): basta para volver a un punto del entrenamiento."""
    return {name: p.detach().to("cpu", copy=True) for name, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, state: dict) -> None:
    """Restaura en el modelo, en su sitio, los parámetros guardados con get_trainable_state."""
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value.to(params[name].device))


def training_step(forward_model, batch: dict, cfg: Config, opt: dict, class_weights, scaler):
    """Forward y backward de un lote. Divide la pérdida por la acumulación y la escala en fp16. Devuelve la pérdida."""
    with contexto_autocast(opt):
        logits = forward_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    loss = compute_loss(logits, batch["labels"], class_weights) / cfg.acumulacion
    (scaler.scale(loss) if scaler is not None else loss).backward()
    return loss.detach()


def optimizer_step(params, optimizer, scheduler, scaler, cfg: Config):
    """Recorta la norma del gradiente, actualiza los pesos y la tasa y borra los gradientes. Devuelve la norma."""
    if scaler is not None:
        scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm) if cfg.max_grad_norm > 0 else None
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm


def is_compile_error(error: BaseException) -> bool:
    """True si el error, o alguna de sus causas encadenadas, viene de torch.compile (dynamo, inductor o triton)."""
    chain, e = [], error
    while e is not None and len(chain) < 10:
        chain.append(f"{type(e).__module__}.{type(e).__name__}")
        e = e.__cause__ or e.__context__
    text = " ".join(chain)
    return any(k in text for k in ("torch._dynamo", "torch._inductor", "triton", "BackendCompilerFailed", "InductorError"))


def compile_model(model, batch: dict, cfg: Config, opt: dict, class_weights):
    """Compila el modelo y lo calienta con un lote real (forward + backward). Devuelve (modelo para el forward, info).

    Si la compilación no aplica o falla, devuelve el modelo sin compilar. Los gradientes del calentamiento se borran.
    """
    if not opt["compilar"]:
        return model, {"compilado": False, "motivo": opt["motivos"]["compilar"], "segundos": 0.0}
    t0 = time.perf_counter()
    try:
        # dynamic=True: un solo grafo para todas las longitudes y tamaños de lote. Sin él, dynamo recompila al ver
        # la segunda longitud y otra vez con el último lote (más corto), y la GPU queda parada minutos cada vez.
        compiled = torch.compile(model, dynamic=True,
                                 mode=None if cfg.modo_compilacion == "default" else cfg.modo_compilacion)
        training_step(compiled, batch, cfg, opt, class_weights, scaler=None)
        if opt["dispositivo"] == "cuda":
            torch.cuda.synchronize()
        info = {"compilado": True, "motivo": "ok", "segundos": time.perf_counter() - t0}
        print(f"  torch.compile listo en {info['segundos']:.1f} s")
        return compiled, info
    except Exception as e:  # noqa: BLE001
        torch._dynamo.reset()
        info = {"compilado": False, "motivo": f"falló: {type(e).__name__}: {str(e)[:300]}", "segundos": time.perf_counter() - t0}
        print(f"AVISO: torch.compile falló y sigo sin compilar · {info['motivo']}")
        return model, info
    finally:
        model.zero_grad(set_to_none=True)


def train(model, forward_model, train_loader, dev_loader, optimizer, scheduler, num_training_steps: int,
          cfg: Config, opt: dict, class_weights, compile_info: dict, log_callback=None) -> dict:
    """Bucle de entrenamiento con acumulación de gradiente, evaluación periódica de dev y early stopping.

    - `forward_model` es el modelo compilado (o el mismo `model` si no se compiló). Si la compilación falla a
      mitad, sigue con `model`.
    - Evalúa dev en los pasos de compute_eval_steps con el modelo sin compilar, para no forzar recompilaciones.
      Cada vez que el macro-F1 mejora, guarda en CPU una copia de los pesos entrenables.
    - Para cuando el macro-F1 lleva `cfg.paciencia` evaluaciones sin mejorar. Al terminar, por early stopping
      o por agotar los pasos, restaura en `model` los pesos del mejor paso.
    - Con SCHEDULER=plateau, cada evaluación sin mejora baja la tasa a la mitad antes de seguir.
    - Muestra una barra de progreso por paso de optimizador (BARRA_PROGRESO=0 la quita).
    - Si se pasa `log_callback(tipo, fila)`, lo llama con cada fila del registro ("train") y con cada evaluación de
      dev ("dev"). Así MLflow recibe las curvas mientras entrena.
    - Devuelve tiempos (sin contar las evaluaciones), memoria pico, el registro por paso y el de evaluaciones.
    """
    device = opt["dispositivo"]
    params = [p for p in model.parameters() if p.requires_grad]
    scaler = torch.amp.GradScaler(device) if opt["escalador"] else None
    eval_steps = compute_eval_steps(compute_steps_per_epoch(train_loader, cfg), num_training_steps, cfg.evals_por_epoca)
    early_stopping = EarlyStopping(patience=cfg.paciencia, min_delta=cfg.delta_minimo)
    best_state, step_log, eval_log = None, [], []
    step, examples, tokens, stopped_early = 0, 0, 0, False
    window_loss, window_batches, window_examples = torch.zeros((), device=device), 0, 0
    eval_seconds = window_eval_seconds = 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = t_window = time.perf_counter()
    model.train()
    # Avanza un paso por actualización de pesos. Va por stdout, como los mensajes del bucle (progress.write), para
    # que el kernel no los entregue desordenados; si fuera por stderr, la CLI mezclaría la barra con los mensajes.
    progress = tqdm(total=num_training_steps, desc="entrenando", unit="paso", ncols=100, mininterval=2,
                    file=sys.stdout, disable=not cfg.barra_progreso)
    for epoch in range(1, cfg.epocas + 1):
        for i, batch in enumerate(train_loader):
            batch = mover(batch, opt)
            try:
                loss = training_step(forward_model, batch, cfg, opt, class_weights, scaler)
            except Exception as e:  # noqa: BLE001
                if forward_model is model or not is_compile_error(e):
                    raise
                progress.write(f"AVISO: torch.compile falló en el paso {step + 1} ({type(e).__name__}); sigo sin compilar.")
                compile_info.update(compilado=False, motivo=f"falló en el paso {step + 1}: {type(e).__name__}")
                forward_model = model
                optimizer.zero_grad(set_to_none=True)
                loss = training_step(forward_model, batch, cfg, opt, class_weights, scaler)
            window_loss += loss
            window_batches += 1
            n = batch["labels"].shape[0]
            examples, window_examples = examples + n, window_examples + n
            tokens += int(batch["attention_mask"].sum())
            if (i + 1) % cfg.acumulacion and (i + 1) != len(train_loader):
                continue
            grad_norm = optimizer_step(params, optimizer, scheduler, scaler, cfg)
            step += 1
            progress.update(1)

            if step % cfg.log_cada == 0 or step == num_training_steps:
                now = time.perf_counter()
                row = {"paso": step, "epoca": epoch, "perdida": float(window_loss) * cfg.acumulacion / window_batches,
                       "lr": scheduler.get_last_lr()[0], "norma_grad": float(grad_norm) if grad_norm is not None else None,
                       "ejemplos_por_s": window_examples / max(now - t_window - window_eval_seconds, 1e-9)}
                step_log.append(row)
                if log_callback:
                    log_callback("train", row)
                progress.set_postfix(perdida=f"{row['perdida']:.4f}", refresh=False)
                progress.write(f"  paso {step:>5}/{num_training_steps} · época {epoch} · pérdida {row['perdida']:.4f} · "
                               f"lr {row['lr']:.2e} · {row['ejemplos_por_s']:.1f} ej/s")
                window_loss.zero_()
                window_batches, window_examples, window_eval_seconds, t_window = 0, 0, 0.0, now

            if step in eval_steps:
                t_eval = time.perf_counter()
                result = validate(model, dev_loader, opt)
                seconds = time.perf_counter() - t_eval
                eval_seconds += seconds
                window_eval_seconds += seconds
                improved = early_stopping.step(result["macro_f1"], step)
                if improved:
                    best_state = get_trainable_state(model)
                eval_log.append({"paso": step, "epoca": epoch, **result, "mejor": improved, "segundos": seconds,
                                 "lr": scheduler.get_last_lr()[0]})
                if log_callback:
                    log_callback("dev", eval_log[-1])
                state = ("mejor hasta ahora" if improved
                         else f"sin mejora {early_stopping.num_bad_evals}/{cfg.paciencia or '∞'}")
                if cfg.scheduler == "plateau" and not improved and not early_stopping.should_stop:
                    # LambdaLR multiplica base_lrs por el factor del paso: bajarlas reduce la tasa desde el paso siguiente.
                    scheduler.base_lrs = [lr * 0.5 for lr in scheduler.base_lrs]
                    state += f" · la tasa baja a {scheduler.base_lrs[0]:.2e}"
                progress.write(f"  dev en el paso {step}: macro-F1 {result['macro_f1']:.4f} · accuracy {result['accuracy']:.4f} · "
                               f"pérdida {result['perdida']:.4f} · {state}")
                if early_stopping.should_stop:
                    stopped_early = True
                    progress.write(f"  early stopping: el macro-F1 de dev no mejora desde el paso {early_stopping.best_step}")
                    break
            if step >= num_training_steps:
                break
        if stopped_early or step >= num_training_steps:
            break
    progress.close()

    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    restored = best_state is not None and early_stopping.best_step != step
    if restored:
        load_trainable_state(model, best_state)
        print(f"  restauro los pesos del paso {early_stopping.best_step} (macro-F1 dev {early_stopping.best_score:.4f}); "
              f"el último paso fue el {step}")
    train_seconds = max(seconds - eval_seconds, 1e-9)
    memory = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None
    return {"pasos": step, "pasos_planeados": num_training_steps, "ejemplos": examples, "segundos": seconds,
            "segundos_eval_dev": eval_seconds, "ejemplos_por_s": examples / train_seconds,
            "tokens_por_s": tokens / train_seconds, "memoria_pico_gb": memory,
            "mejor_paso": early_stopping.best_step, "mejor_macro_f1_dev": early_stopping.best_score,
            "parada_temprana": stopped_early, "pesos_restaurados": restored,
            "compilado_al_final": forward_model is not model, "registro": step_log, "evaluaciones": eval_log}
