"""Registro de la corrida en un servidor de MLflow desde un hilo aparte, sin frenar el entrenamiento si el servidor
se cae (registro_mlflow.py de lora.zip).

abrir_mlflow crea la corrida al empezar, así que se ve en RUNNING mientras entrena; RealTimeMLflowCallback le pasa lo
que registra el Trainer, y train.py envía al final las métricas A6/A7, los archivos y el mejor modelo (columna Models)
y la cierra en FINISHED (o FAILED si falla). Si no hay servidor, avisa y los resultados quedan en local.
descargar_modelo baja de MLflow el mejor modelo de una corrida, para quien lo consume (create_predictor).
"""

import contextlib
import json
import os
import platform
import queue
import sys
import tempfile
import threading
import time
import urllib.request
import warnings
from datetime import datetime, timedelta
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import TrainerCallback

from src.config.schema import ExperimentConfig

# Cliente de MLflow: pocos reintentos y espera corta por pedido. Si el servidor cae, quien reintenta es RegistroMLflow,
# en segundo plano y sin frenar el entrenamiento.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")  # mlflow 3 manda estadísticas de uso a sus servidores
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

try:
    from mlflow import MlflowClient
    from mlflow.entities import Dataset, DatasetInput, InputTag, LoggedModelOutput, Metric, Param, RunTag
except ImportError:
    MlflowClient = Dataset = DatasetInput = InputTag = LoggedModelOutput = Metric = Param = RunTag = None

AVISO_SIN_SERVIDOR = "servidor de MLflow no encontrado, los resultados se guardarán localmente"
# Si se define (python -m src.modeling lo hace), la salida de la consola se sube a la corrida como salida.log.
SALIDA_CONSOLA: Path | None = None


class MLflowNoEncontrado(UserWarning):
    """Advertencia de que no hay servidor de MLflow: los resultados se guardan en local."""


warnings.simplefilter("always", MLflowNoEncontrado)  # se muestra en cada corrida, no solo la primera vez


class RegistroMLflow:
    """Registra una corrida en un servidor de MLflow desde un hilo aparte, sin frenar ni tumbar el entrenamiento.

    - Lo que se registra (parámetros, métricas con su paso, artefactos y el cierre) entra a una cola, y un hilo lo
      envía en orden. Si el servidor no responde, el hilo guarda lo pendiente y reintenta cada `reintento_s`
      segundos. Cuando vuelve, envía todo con el paso y la hora originales y sigue en la misma corrida, que termina
      en FINISHED: MLflow solo marca FAILED si el cliente se lo pide.
    - En cada intento prueba la IP de la configuración y, si existe, la escrita en `archivo_ip`. Así se puede
      cambiar la IP a mitad de corrida (en AWS Academy cambia al reiniciar la sesión, salvo con una Elastic IP).
    - La corrida se crea la primera vez que el servidor responde. Si al volver ya no la tiene (perdió su base de
      datos), crea otra con el mismo nombre y reenvía todo.
    - Solo descarta un envío si MLflow dice que su contenido es inválido (INVALID_PARAMETER_VALUE), o un artefacto
      que no se pudo subir con el servidor funcionando. Cualquier otro error, como un 403 porque la IP no está en el
      --allowed-hosts del servidor, se trata como caída y se reintenta.
    - cerrar() espera a que se envíe lo pendiente. Con el servidor caído espera como mucho hasta `espera_min`
      minutos después de que dejó de responder. Si no vuelve, cierra sin MLflow y la corrida queda en RUNNING e
      incompleta en el servidor; cerrar() devuelve False y train.py conserva los resultados en local.
    """

    def __init__(self, ip: str, experimento: str, nombre: str, espera_min: float, archivo_ip: Path,
                 reintento_s: float = 30.0):
        """Guarda la configuración y arranca el hilo que envía a MLflow."""
        self.ip, self.experimento, self.nombre, self.archivo_ip = ip, experimento, nombre, Path(archivo_ip)
        self.espera_s, self.reintento_s = espera_min * 60, reintento_s
        self.inicio_ms = int(time.time() * 1000)
        self.uri = self.cliente = self.run_id = self.experimento_id = self.modelo_id = None
        self.cola = queue.Queue()
        self.pendientes, self.enviados = [], []    # lo que falta enviar y lo ya enviado (por si hay que reenviar)
        self.caido_desde = None                    # time.monotonic() del primer intento fallido de la caída actual
        self.proximo_intento, self.descartados, self.recreaciones, self.ultimo_error = 0.0, 0, 0, None
        self.abandonado = self.avise_espera = False
        self.hilo = threading.Thread(target=self._bucle, name="mlflow", daemon=True)
        self.hilo.start()

    # Lo que llama el entrenamiento: solo encola, no espera a la red.
    def parametros(self, valores: dict) -> None:
        """Encola parámetros como texto (hasta 6000 caracteres), en tandas de 100 como pide MLflow."""
        items = [Param(k, str(v)[:6000]) for k, v in valores.items()]
        for i in range(0, len(items), 100):
            self.cola.put(("datos", {"params": items[i:i + 100]}))

    def etiquetas(self, valores: dict) -> None:
        """Encola etiquetas (texto); salta las que valen None."""
        items = [RunTag(k, str(v)[:5000]) for k, v in valores.items() if v is not None]
        for i in range(0, len(items), 100):
            self.cola.put(("datos", {"tags": items[i:i + 100]}))

    def datasets(self, hashes: dict, rutas: dict) -> None:
        """Encola los splits como entradas de la corrida (columna Dataset)."""
        tag, carpeta = os.environ.get("DATOS_TAG"), os.environ.get("DATOS_DVC")
        version = tag if en_colab() and tag else "local"
        entradas = [
            DatasetInput(
                Dataset(name=f"{version}/{split}", digest=sha[:16], source_type="dvc" if version != "local" else "local",
                        source=f"{carpeta}@{tag}" if version != "local" else str(rutas[split])),
                tags=[InputTag("mlflow.data.context", split)],
            )
            for split, sha in hashes.items()
        ]
        self.cola.put(("entradas", entradas))

    def metricas(self, valores: dict, paso: int = 0) -> None:
        """Encola métricas con su paso y la hora de ahora. Salta lo que no es un número (None, texto)."""
        ahora = int(time.time() * 1000)
        items = [Metric(k, float(v), ahora, int(paso)) for k, v in valores.items()
                 if isinstance(v, (int, float, np.integer, np.floating))]
        for i in range(0, len(items), 1000):
            self.cola.put(("datos", {"metrics": items[i:i + 1000]}))

    def artefacto(self, archivo: Path) -> None:
        """Encola un archivo como artefacto de la corrida."""
        self.cola.put(("artefacto", (str(archivo), None)))

    def modelo(self, nombre: str, carpetas: dict, tipo: str, parametros: dict) -> None:
        """Encola el mejor modelo: un modelo registrado de la corrida (columna Models) con los archivos de `carpetas`
        ({subcarpeta en el modelo o None para la raíz: carpeta local}). Las carpetas deben existir hasta cerrar()."""
        self.cola.put(("modelo", {"nombre": nombre, "carpetas": {k: str(v) for k, v in carpetas.items()},
                                  "tipo": tipo, "parametros": {k: str(v)[:6000] for k, v in parametros.items()},
                                  "model_id": None}))

    def cerrar(self, estado: str = "FINISHED", esperar: bool = True) -> bool:
        """Encola el cierre con `estado` y espera al hilo. Devuelve True si todo llegó a MLflow, sin descartes.

        Con esperar=False (corrida que falló o se interrumpió) no espera a un servidor caído.
        """
        self.cola.put(("fin", (estado, esperar)))
        self.hilo.join()
        return not self.abandonado and self.run_id is not None and self.descartados == 0

    def url_corrida(self) -> str | None:
        """Dirección de la corrida en la interfaz de MLflow, o None si todavía no se creó."""
        if self.run_id is None:
            return None
        return f"{self.uri}/#/experiments/{self.experimento_id}/runs/{self.run_id}"

    def _subir_modelo(self, estado: dict) -> None:
        """Crea el modelo registrado ligado a la corrida, sube sus archivos y lo deja en READY. Si falla a mitad, el
        reintento sigue con el mismo modelo (model_id queda en `estado`)."""
        if estado["model_id"] is None:
            modelo = self.cliente.create_logged_model(self.experimento_id, name=estado["nombre"],
                                                      source_run_id=self.run_id, params=estado["parametros"],
                                                      model_type=estado["tipo"])
            self.cliente.log_outputs(self.run_id, [LoggedModelOutput(modelo.model_id, step=0)])
            estado["model_id"] = modelo.model_id
        for destino, carpeta in estado["carpetas"].items():
            self.cliente.log_model_artifacts(estado["model_id"], carpeta, destino)
        self.cliente.finalize_logged_model(estado["model_id"], "READY")
        self.modelo_id = estado["model_id"]

    # Lo que hace el hilo.
    @staticmethod
    def _avisar(texto: str) -> None:
        """Imprime un aviso sin romper la barra de progreso del entrenamiento."""
        tqdm.write(f"  [mlflow] {texto}", file=sys.stdout)

    @staticmethod
    def _responde(uri: str) -> bool:
        """True si el endpoint /health del servidor contesta en menos de 5 s."""
        try:
            with urllib.request.urlopen(f"{uri}/health", timeout=5) as respuesta:
                return respuesta.status == 200
        except Exception:  # noqa: BLE001  (conexión rechazada, tiempo agotado, DNS, HTTP 5xx...)
            return False

    def _direcciones(self) -> list[str]:
        """URIs a probar: la IP de la configuración y, si existe y es distinta, la de `archivo_ip`."""
        candidatas = [self.ip]
        try:
            candidatas.append(self.archivo_ip.read_text(encoding="utf-8").strip())
        except OSError:
            pass
        uris = [c if "://" in c else f"http://{c}" for c in candidatas if c]
        return list(dict.fromkeys(uris))

    def _conectar(self) -> bool:
        """Busca una dirección que responda y acepte pedidos, y crea la corrida si hace falta. True si quedó lista.

        Con la conexión normal solo consulta /health. Al conectar por primera vez o al volver de una caída, además
        comprueba con un pedido real que la dirección acepta pedidos (un servidor puede responder /health y rechazar
        el resto) y que todavía tiene la corrida.
        """
        verificar = self.caido_desde is not None or self.run_id is None
        alguna_responde = False
        for uri in self._direcciones():
            if not self._responde(uri):
                continue
            alguna_responde = True
            if uri == self.uri and not verificar:
                return True
            try:
                cliente = self.cliente if uri == self.uri else MlflowClient(tracking_uri=uri)
                if self.run_id is not None:
                    try:
                        cliente.get_run(self.run_id)
                    except Exception as e:  # noqa: BLE001
                        if getattr(e, "error_code", "") != "RESOURCE_DOES_NOT_EXIST" or self.recreaciones >= 3:
                            raise
                        self._olvidar_corrida()
                if self.run_id is None:
                    experimento = cliente.get_experiment_by_name(self.experimento)
                    if experimento is None:
                        experimento_id = cliente.create_experiment(self.experimento)
                    else:
                        experimento_id = experimento.experiment_id
                        if experimento.lifecycle_stage == "deleted":
                            cliente.restore_experiment(experimento_id)
                    corrida = cliente.create_run(experimento_id, start_time=self.inicio_ms, run_name=self.nombre)
                    self.experimento_id, self.run_id = experimento_id, corrida.info.run_id
                    self._avisar(f"corrida «{self.nombre}»: {uri}/#/experiments/{experimento_id}/runs/{self.run_id}")
            except Exception as e:  # noqa: BLE001  (esta dirección no sirve: se prueba la siguiente)
                self.ultimo_error = f"{getattr(e, 'error_code', '') or type(e).__name__}: {str(e)[:200]}"
                continue
            if uri != self.uri:
                if self.uri is not None:
                    self._avisar(f"sigo en la nueva dirección {uri}")
                self.uri, self.cliente = uri, cliente
            return True
        if not alguna_responde:
            self.ultimo_error = None
        return False

    def _acepta_pedidos(self) -> bool:
        """True si el servidor actual responde /health y además contesta un pedido sobre la corrida."""
        if not self._responde(self.uri):
            return False
        try:
            self.cliente.get_run(self.run_id)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _olvidar_corrida(self) -> None:
        """El servidor ya no tiene la corrida (perdió su base de datos): se creará otra y se reenviará todo."""
        self.recreaciones += 1
        self._avisar("el servidor ya no tiene la corrida (¿perdió su base de datos?): creo otra con el mismo nombre "
                     "y reenvío todo")
        self.run_id = None
        self.pendientes[:0], self.enviados = self.enviados, []

    def _tomar_lote(self, maximo: int) -> tuple[dict, int]:
        """Junta datos consecutivos del frente de la cola sin pasar los límites de log_batch (1000 en total)."""
        lote, n = {"metrics": [], "params": [], "tags": []}, 0
        for tipo, contenido in self.pendientes[:maximo]:
            if tipo != "datos":
                break
            (clave, items), = contenido.items()
            limite = 1000 if clave == "metrics" else 100
            if n and (len(lote[clave]) + len(items) > limite or sum(map(len, lote.values())) + len(items) > 1000):
                break
            lote[clave] += items
            n += 1
        return lote, n

    def _enviar(self) -> bool:
        """Envía lo pendiente en orden. Devuelve True cuando envió el cierre."""
        if not self._conectar():
            self._marcar_caida()
            return False
        self._marcar_recuperado()
        maximo = len(self.pendientes)
        while self.pendientes:
            tipo, contenido = self.pendientes[0]
            n = 1
            try:
                if tipo == "fin":
                    self.cliente.set_terminated(self.run_id, status=contenido[0], end_time=int(time.time() * 1000))
                    self.pendientes.pop(0)
                    return True
                if tipo == "artefacto":
                    self.cliente.log_artifact(self.run_id, contenido[0], contenido[1])
                elif tipo == "entradas":
                    self.cliente.log_inputs(self.run_id, datasets=contenido)
                elif tipo == "modelo":
                    self._subir_modelo(contenido)
                else:
                    lote, n = self._tomar_lote(maximo)
                    self.cliente.log_batch(self.run_id, metrics=lote["metrics"], params=lote["params"], tags=lote["tags"])
                    self.enviados += self.pendientes[:n]
                del self.pendientes[:n]
                maximo = len(self.pendientes)
            except Exception as e:  # noqa: BLE001
                codigo = getattr(e, "error_code", "")
                if codigo == "RESOURCE_DOES_NOT_EXIST" and self.recreaciones < 3:
                    self._olvidar_corrida()
                    return False
                if tipo in ("artefacto", "modelo") and codigo != "INVALID_PARAMETER_VALUE" and self._acepta_pedidos():
                    # El servidor funciona pero el archivo no se pudo subir (por ejemplo, guarda los artefactos en
                    # S3 sin pasarlos por él y aquí no hay credenciales): se salta ese archivo y se sigue. La corrida
                    # queda incompleta y train.py conserva los resultados en local.
                    self.descartados += 1
                    nombre = Path(contenido[0]).name if tipo == "artefacto" else f"el modelo {contenido['nombre']}"
                    if self.descartados <= 3:
                        self._avisar(f"no pude subir {nombre} ({type(e).__name__}: {str(e)[:150]}); "
                                     "el servidor responde, así que lo salto")
                    self.pendientes.pop(0)
                    continue
                if codigo != "INVALID_PARAMETER_VALUE":
                    # Conexión, servidor caído o que rechaza el pedido (PERMISSION_DENIED si la IP no está en su
                    # --allowed-hosts): se trata como caída y se reintenta, sin descartar nada. El motivo solo se
                    # anota si /health contesta; si no, el aviso dice que el servidor no responde.
                    rechaza = self._responde(self.uri)
                    self.ultimo_error = f"{codigo or type(e).__name__}: {str(e)[:200]}" if rechaza else None
                    self._marcar_caida()
                    return False
                if n > 1:
                    maximo = 1     # el lote tenía algo inválido: se reintenta de a uno para descartar solo eso
                    continue
                self.descartados += 1
                if self.descartados <= 3:
                    self._avisar(f"MLflow rechazó un envío ({tipo}) por inválido y lo descarto: {str(e)[:200]}")
                self.pendientes.pop(0)
        return False

    def _marcar_caida(self) -> None:
        """Anota el primer intento fallido de una caída y cuándo toca el próximo intento."""
        ahora = time.monotonic()
        self.proximo_intento = ahora + self.reintento_s
        if self.caido_desde is None:
            self.caido_desde = ahora
            hasta = datetime.now() + timedelta(seconds=self.espera_s)
            motivo = f"rechaza los pedidos ({self.ultimo_error})" if self.ultimo_error else "no responde"
            if self.ultimo_error and "Invalid Host header" in self.ultimo_error:
                motivo += "; la dirección tiene que estar en el --allowed-hosts del servidor"
            self._avisar(f"el servidor {motivo} ({', '.join(self._direcciones())}). Sigo entrenando y reintento "
                         f"cada {self.reintento_s:.0f} s; al terminar lo espero hasta las {hasta:%H:%M}. "
                         f"Si cambió la IP, escríbela en {self.archivo_ip}.")

    def _marcar_recuperado(self) -> None:
        """Si venía de una caída, avisa que el servidor volvió."""
        if self.caido_desde is not None:
            self._avisar(f"el servidor volvió tras {(time.monotonic() - self.caido_desde) / 60:.1f} min; "
                         "envío lo pendiente")
            self.caido_desde, self.ultimo_error = None, None

    def _bucle(self) -> None:
        """Hilo: toma lo encolado, lo envía y reintenta mientras el servidor no responda, hasta procesar el cierre."""
        try:
            while True:
                if self.caido_desde is not None:
                    espera = max(0.05, self.proximo_intento - time.monotonic())
                else:
                    espera = 0.0 if self.pendientes else 5.0
                try:
                    self.pendientes.append(self.cola.get(timeout=espera) if espera else self.cola.get_nowait())
                    while True:
                        self.pendientes.append(self.cola.get_nowait())
                except queue.Empty:
                    pass
                fin = next((contenido for tipo, contenido in self.pendientes if tipo == "fin"), None)
                if self.abandonado:
                    self.pendientes.clear()
                    if fin:
                        return
                    continue
                if not self.pendientes or (self.caido_desde is not None and time.monotonic() < self.proximo_intento):
                    continue
                if self._enviar():
                    if self.descartados > 3:
                        self._avisar(f"en total se descartaron {self.descartados} envíos rechazados")
                    return
                if fin and self.caido_desde is not None:
                    limite = self.caido_desde + self.espera_s
                    if not fin[1] or time.monotonic() >= limite:
                        self.abandonado = True
                        self._avisar("el servidor no volvió a tiempo: cierro sin MLflow. Los resultados quedan en "
                                     "local; en MLflow la corrida quedó incompleta.")
                        return
                    if not self.avise_espera:
                        self.avise_espera = True
                        hasta = datetime.now() + timedelta(seconds=limite - time.monotonic())
                        self._avisar(f"terminó la corrida y el servidor no responde: lo espero hasta las {hasta:%H:%M}")
        except Exception as e:  # noqa: BLE001  (un error propio no debe tumbar el entrenamiento)
            self.abandonado = True
            self._avisar(f"error inesperado, dejo de enviar a MLflow: {type(e).__name__}: {e}")


def archivo_ip_mlflow(colab: bool) -> Path:
    """Archivo donde se puede escribir una IP nueva para MLflow: /content/mlflow_ip.txt en Colab, ./mlflow_ip.txt en local."""
    return Path("/content/mlflow_ip.txt") if colab else Path("mlflow_ip.txt")


def en_colab() -> bool:
    """True en un runtime de Google Colab, desde la interfaz web o desde la CLI."""
    return "COLAB_RELEASE_TAG" in os.environ


def versiones() -> dict:
    """Versión de Python y de las librerías del entrenamiento (None si no está instalada), para las etiquetas."""
    v = {"python": platform.python_version()}
    for paquete in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "mlflow", "mlflow-skinny"):
        try:
            v[paquete] = metadata.version(paquete)
        except metadata.PackageNotFoundError:
            v[paquete] = None
    return v


def trazabilidad() -> dict:
    """Etiquetas con el código y los datos de la corrida, desde las variables que pasan los scripts de scripts/.

    CODIGO_COMMIT es el commit del código (MLflow lo muestra como Git Commit) y CODIGO_SIN_COMMIT dice si src/ o
    configs/ tenían cambios sin commit. En Colab, DATOS_TAG y DATOS_DVC dicen de qué tag y qué carpeta bajó DVC los
    datos; en local no se registran, porque los datos salen de la carpeta local y no de un tag.
    """
    etiquetas = {
        "mlflow.source.git.commit": os.environ.get("CODIGO_COMMIT"),
        "codigo.cambios_sin_commit": os.environ.get("CODIGO_SIN_COMMIT"),
    }
    if en_colab():
        etiquetas |= {"datos.tag": os.environ.get("DATOS_TAG"), "datos.dvc": os.environ.get("DATOS_DVC")}
    return {k: v for k, v in etiquetas.items() if v}


def parametros_experimento(cfg: ExperimentConfig) -> dict:
    """Los parámetros que registraba train.py más todos los de la familia (tasa, lotes, LoRA, optimizaciones)."""
    rep = cfg.input_representation
    familia = getattr(cfg, cfg.model_family, None)
    return {
        "experiment_name": cfg.experiment_name,
        "model_family": cfg.model_family,
        "configuration": rep.configuration,
        "seed": cfg.seed,
        "max_seq_length": rep.max_seq_length,
        "enforce_per_chunk_budget": rep.enforce_per_chunk_budget,
        **(familia.model_dump() if familia is not None else {}),
    }


def direcciones_mlflow(uri_config: str = "", colab: bool = False) -> list[str]:
    """Servidores de MLflow candidatos, en orden: el de la configuración (paths.mlflow_tracking_uri), las variables
    MLFLOW_TRACKING_URI y MLFLOW_IP, y la IP escrita en mlflow_ip.txt. Solo servidores http(s)."""
    candidatas = [uri_config, os.environ.get("MLFLOW_TRACKING_URI", ""), os.environ.get("MLFLOW_IP", "")]
    try:
        candidatas.append(archivo_ip_mlflow(colab).read_text(encoding="utf-8").strip())
    except OSError:
        pass
    uris = [c.strip().rstrip("/") if "://" in c else f"http://{c.strip()}" for c in candidatas if c and c.strip()]
    return [u for u in dict.fromkeys(uris) if u.startswith(("http://", "https://"))]


def _servidor(uri_config: str, colab: bool) -> tuple[str | None, str]:
    """El primer servidor candidato que responde, o None con el motivo."""
    if MlflowClient is None:
        return None, "el paquete mlflow no está instalado"
    candidatas = direcciones_mlflow(uri_config, colab)
    if not candidatas:
        return None, "no hay dirección: define MLFLOW_IP o escríbela en mlflow_ip.txt"
    for _ in range(2):  # un segundo intento por si el primero coincide con un corte breve
        for uri in candidatas:
            if RegistroMLflow._responde(uri):
                return uri, ""
        time.sleep(2)
    return None, f"no responde {', '.join(candidatas)}"


def abrir_mlflow(cfg: ExperimentConfig, colab: bool, versiones: dict) -> RegistroMLflow | None:
    """Abre la corrida en el servidor de MLflow y encola sus parámetros, versiones y GPU. La corrida queda en
    RUNNING hasta que train.py la cierra.

    Si ningún servidor candidato (direcciones_mlflow) responde, lanza la advertencia AVISO_SIN_SERVIDOR y devuelve
    None: los resultados se guardan en local. Si el servidor se cae después, RegistroMLflow reintenta.
    """
    uri, motivo = _servidor(cfg.paths.mlflow_tracking_uri, colab)
    if uri is None:
        warnings.warn(f"{AVISO_SIN_SERVIDOR} en {cfg.paths.output_dir / cfg.experiment_name} ({motivo})",
                      MLflowNoEncontrado, stacklevel=2)
        return None
    experimento = cfg.paths.mlflow_experiment_name
    print(f"MLflow: {uri} · experimento {experimento} · corrida {cfg.experiment_name}")
    registro = RegistroMLflow(uri, experimento, cfg.experiment_name,
                              float(os.environ.get("MLFLOW_ESPERA_MIN", "20")), archivo_ip_mlflow(colab))
    registro.parametros(parametros_experimento(cfg))
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    registro.etiquetas({"gpu": gpu, **{f"version.{k}": v for k, v in versiones.items()}, **trazabilidad()})
    return registro


def _corrida_terminada(cliente, experimento: str, nombre: str):
    """La última corrida FINISHED de `experimento` llamada `nombre` (o con ese id), o None."""
    exp = cliente.get_experiment_by_name(experimento)
    if exp is None:
        return None
    corridas = cliente.search_runs([exp.experiment_id], f"attributes.run_name = '{nombre}'",
                                   order_by=["attributes.start_time DESC"])
    corridas = [c for c in corridas if c.info.status == "FINISHED"]
    return cliente.get_run(corridas[0].info.run_id) if corridas else None


def existe_en_mlflow(cfg: ExperimentConfig, colab: bool = False) -> bool:
    """True si el servidor ya tiene una corrida FINISHED con el nombre de la configuración."""
    uri, _ = _servidor(cfg.paths.mlflow_tracking_uri, colab)
    if uri is None:
        return False
    return _corrida_terminada(MlflowClient(tracking_uri=uri), cfg.paths.mlflow_experiment_name,
                              cfg.experiment_name) is not None


def descargar_modelo(cfg: ExperimentConfig, destino: Path | None = None) -> Path:
    """Descarga de MLflow el mejor modelo de la corrida cfg.experiment_name y devuelve su carpeta.

    Trae lo que train.py registró como modelo: los .safetensors con su config y tokenizador (encoder o adaptador
    LoRA) o tfidf_pipeline.joblib. Sin `destino`, va a una carpeta temporal: nada queda en la carpeta del proyecto.
    """
    from mlflow.artifacts import download_artifacts

    uri, motivo = _servidor(cfg.paths.mlflow_tracking_uri, en_colab())
    if uri is None:
        raise FileNotFoundError(f"No hay modelo local de «{cfg.experiment_name}» y MLflow no está disponible ({motivo})")
    cliente = MlflowClient(tracking_uri=uri)
    corrida = _corrida_terminada(cliente, cfg.paths.mlflow_experiment_name, cfg.experiment_name)
    salidas = corrida.outputs.model_outputs if corrida is not None and corrida.outputs else []
    if not salidas:
        raise FileNotFoundError(f"MLflow no tiene un modelo de la corrida «{cfg.experiment_name}» en el experimento "
                                f"«{cfg.paths.mlflow_experiment_name}»")
    # La ubicación de los archivos del modelo, no models:/<id>: esa forma busca el modelo con la URI global de
    # MLflow y no con la de este servidor.
    ubicacion = cliente.get_logged_model(salidas[-1].model_id).artifact_location
    destino = Path(destino or tempfile.mkdtemp(prefix=f"{cfg.experiment_name}_"))
    ruta = Path(download_artifacts(artifact_uri=ubicacion, dst_path=str(destino), tracking_uri=uri))
    # Modelo de humo: la base minúscula viaja con el adaptador en modelo_base/; el adaptador pasa a apuntar a ella
    adaptador = ruta / "adapter_config.json"
    if (ruta / "modelo_base").is_dir() and adaptador.is_file():
        datos = json.loads(adaptador.read_text(encoding="utf-8"))
        datos["base_model_name_or_path"] = str(ruta / "modelo_base")
        adaptador.write_text(json.dumps(datos, indent=2), encoding="utf-8")
    return ruta


@contextlib.contextmanager
def modelo_local_o_descargado(cfg: ExperimentConfig, local: Path, archivo: str = ""):
    """La ruta local del modelo si existe; si no (los resultados quedaron en MLflow), el modelo descargado a una
    carpeta temporal que se borra al salir del bloque, cuando ya está cargado en memoria."""
    if local.exists():
        yield local
        return
    with tempfile.TemporaryDirectory(prefix=f"{cfg.experiment_name}_", ignore_cleanup_errors=True) as tmp:
        ruta = descargar_modelo(cfg, Path(tmp))
        yield ruta / archivo if archivo else ruta


class RealTimeMLflowCallback(TrainerCallback):
    """Envía a MLflow, mientras entrena, lo que registra el Trainer: pérdida, tasa y norma del gradiente cada
    logging_steps, y las métricas de dev en cada evaluación."""

    def __init__(self, registro: RegistroMLflow) -> None:
        self.registro = registro

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.registro.metricas(logs, paso=state.global_step)
