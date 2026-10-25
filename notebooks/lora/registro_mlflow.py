"""Registro de la corrida en MLflow desde un hilo aparte, sin frenar el entrenamiento si el servidor se
cae (celda 11).
"""

import queue
import sys
import threading
import time
import urllib.request
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm

from configuracion import Config
from entorno import titulo

try:
    from mlflow import MlflowClient  # registro en MLflow, solo con MLFLOW_IP (pip install mlflow-skinny)
    from mlflow.entities import Metric, Param, RunTag
except ImportError:
    MlflowClient = Metric = Param = RunTag = None


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
      incompleta en el servidor. Los resultados locales no dependen de nada de esto.
    """

    CAMPOS_FILA = {"train": ("perdida", "lr", "norma_grad", "ejemplos_por_s", "epoca"),
                   "dev": ("macro_f1", "f1_macro_9", "f1_macro_5", "f1_weighted", "accuracy", "perdida")}

    def __init__(self, ip: str, experimento: str, nombre: str, espera_min: float, archivo_ip: Path,
                 reintento_s: float = 30.0):
        """Guarda la configuración y arranca el hilo que envía a MLflow."""
        self.ip, self.experimento, self.nombre, self.archivo_ip = ip, experimento, nombre, Path(archivo_ip)
        self.espera_s, self.reintento_s = espera_min * 60, reintento_s
        self.inicio_ms = int(time.time() * 1000)
        self.uri = self.cliente = self.run_id = self.experimento_id = None
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

    def metricas(self, valores: dict, paso: int = 0) -> None:
        """Encola métricas con su paso y la hora de ahora. Salta lo que no es un número (None, texto)."""
        ahora = int(time.time() * 1000)
        items = [Metric(k, float(v), ahora, int(paso)) for k, v in valores.items()
                 if isinstance(v, (int, float, np.integer, np.floating))]
        for i in range(0, len(items), 1000):
            self.cola.put(("datos", {"metrics": items[i:i + 1000]}))

    def registrar_fila(self, tipo: str, fila: dict) -> None:
        """Pasa a MLflow una fila del registro de entrenamiento ("train") o de una evaluación de dev ("dev")."""
        self.metricas({f"{tipo}/{k}": fila.get(k) for k in self.CAMPOS_FILA[tipo]}, fila["paso"])

    def artefactos(self, carpeta: Path, excluir: tuple = ()) -> None:
        """Encola cada archivo de `carpeta` como artefacto, con su subcarpeta. Salta las subcarpetas de `excluir`."""
        carpeta = Path(carpeta)
        for archivo in sorted(p for p in carpeta.rglob("*") if p.is_file()):
            relativo = archivo.relative_to(carpeta)
            if relativo.parts[0] not in excluir:
                destino = None if relativo.parent == Path(".") else relativo.parent.as_posix()
                self.cola.put(("artefacto", (str(archivo), destino)))

    def cerrar(self, estado: str = "FINISHED", esperar: bool = True) -> bool:
        """Encola el cierre con `estado` y espera al hilo. Devuelve True si todo llegó a MLflow.

        Con esperar=False (corrida que falló o se interrumpió) no espera a un servidor caído.
        """
        self.cola.put(("fin", (estado, esperar)))
        self.hilo.join()
        return not self.abandonado and self.run_id is not None

    def url_corrida(self) -> str | None:
        """Dirección de la corrida en la interfaz de MLflow, o None si todavía no se creó."""
        if self.run_id is None:
            return None
        return f"{self.uri}/#/experiments/{self.experimento_id}/runs/{self.run_id}"

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
        uris = [c if c.startswith(("http://", "https://")) else f"http://{c}" for c in candidatas if c]
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
            cliente = self.cliente if uri == self.uri else MlflowClient(tracking_uri=uri)
            try:
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
                if tipo == "artefacto" and codigo != "INVALID_PARAMETER_VALUE" and self._acepta_pedidos():
                    # El servidor funciona pero el archivo no se pudo subir (por ejemplo, guarda los artefactos en
                    # S3 sin pasarlos por él y aquí no hay credenciales): se salta ese archivo y se sigue.
                    self.descartados += 1
                    if self.descartados <= 3:
                        self._avisar(f"no pude subir {Path(contenido[0]).name} ({type(e).__name__}: {str(e)[:150]}); "
                                     "el servidor responde, así que salto ese archivo")
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
                        self._avisar("el servidor no volvió a tiempo: cierro sin MLflow. Los resultados locales están "
                                     "completos; en MLflow la corrida quedó incompleta.")
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


def abrir_mlflow(cfg: Config, colab: bool, versiones: dict) -> RegistroMLflow | None:
    """Abre el registro en MLflow si MLFLOW_IP está definido y encola la Config como parámetros; si no, devuelve None.

    No espera al servidor: el entrenamiento empieza igual y el hilo avisa enseguida si no responde.
    """
    if not cfg.mlflow_ip:
        return None
    if MlflowClient is None:
        raise ImportError("MLFLOW_IP está definido pero mlflow no está instalado: pip install mlflow-skinny")
    titulo("MLflow")
    print(f"  servidor {cfg.mlflow_ip} · experimento {cfg.mlflow_experimento} · corrida {cfg.nombre_corrida}")
    registro = RegistroMLflow(cfg.mlflow_ip, cfg.mlflow_experimento, cfg.nombre_corrida, cfg.mlflow_espera_min,
                              archivo_ip_mlflow(colab))
    # strategy, model, seed y selection_metric con los nombres de la rama SciBERT, para agrupar y comparar corridas
    registro.parametros(asdict(cfg) | {"strategy": cfg.configuracion, "model": cfg.modelo, "seed": cfg.semilla,
                                       "selection_metric": "dev_f1_macro_present"})
    registro.etiquetas({f"version.{k}": v for k, v in versiones.items()})
    return registro


def metricas_finales(salida: dict) -> dict:
    """Aplana para MLflow las métricas finales de cada split (global, por clase y por fuente) y el resumen.

    Dev va como dev_final/..., para no mezclarse con la curva dev/... del entrenamiento; test va como test/...
    Además van test_accuracy, test_f1_macro_9, test_f1_macro_present, test_f1_weighted, train_seconds, latency_ms y
    best_epoch, con los nombres y las definiciones de la rama SciBERT (02_scibert_ab), para compararlas en MLflow, y
    test_f1_macro_5, que en test vale lo mismo que test_f1_macro_present.
    """
    m = {}
    for split, r in salida["resultados"].items():
        x, p = r["metricas"], "dev_final" if split == "dev" else split
        m |= {f"{p}/{k}": x[k] for k in ("macro_f1", "f1_macro_9", "f1_macro_5", "f1_weighted", "accuracy", "ece", "n",
                                         "segundos", "ejemplos_por_s")}
        if x["macro_f1_ic95"]:
            m[f"{p}/macro_f1_ic95_inf"], m[f"{p}/macro_f1_ic95_sup"] = x["macro_f1_ic95"]
        for clase, v in x["por_clase"].items():
            m |= {f"{p}/{k}/{clase}": v[k] for k in ("f1", "precision", "recall", "soporte")}
        for fuente, v in x["por_fuente"].items():
            m |= {f"{p}/fuente/{fuente}/{k}": v[k] for k in ("macro_f1", "accuracy", "n")}
        m |= {f"{p}/predicciones/{clase}": n for clase, n in x["predicciones_por_clase"].items()}
    resumen = salida["resumen"]
    m |= {f"entrenamiento/{k}": v for k, v in resumen["entrenamiento"].items()}
    m |= {f"adaptador/{k}": resumen["lora"][k] for k in ("entrenables", "totales", "porcentaje")}
    m["compilacion/segundos"] = resumen["compilacion"]["segundos"]
    m["segundos_totales"] = resumen["segundos_totales"]
    if "test" in salida["resultados"]:
        t = salida["resultados"]["test"]["metricas"]
        m |= {"test_accuracy": t["accuracy"], "test_f1_macro_9": t["f1_macro_9"], "test_f1_macro_5": t["f1_macro_5"],
              "test_f1_macro_present": t["macro_f1"], "test_f1_weighted": t["f1_weighted"],
              "latency_ms": 1000 * t["segundos"] / t["n"]}
    m |= {"train_seconds": resumen["entrenamiento"]["segundos"], "best_epoch": resumen["seleccion"]["mejor_epoca"]}
    return m


def cerrar_mlflow(registro: RegistroMLflow, cfg: Config, salida: dict) -> None:
    """Envía las métricas finales, la GPU y los archivos de la corrida, y cierra en FINISHED.

    Los artefactos son los archivos de la carpeta de la corrida, los mismos del zip; el adaptador solo con
    MLFLOW_ADAPTADOR=1. Espera a que todo llegue (ver RegistroMLflow.cerrar) y dice dónde quedó.
    """
    hw = salida["resumen"]["hardware"]
    registro.etiquetas({"gpu": hw["gpu"] or hw["dispositivo"], "modelo": salida["resumen"]["modelo"]})
    registro.metricas(metricas_finales(salida))
    if salida["zip"]:
        registro.artefactos(Path(cfg.salidas_dir) / cfg.nombre_corrida, () if cfg.mlflow_adaptador else ("adaptador",))
    titulo("MLflow")
    print("  envío las métricas finales y los archivos de la corrida")
    completo = registro.cerrar("FINISHED")
    print(f"  {'listo' if completo else 'incompleto'}: {registro.url_corrida() or 'la corrida no llegó a crearse'}")
