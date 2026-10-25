"""Preprocesamiento: carga y validación del dataset, texto de entrada, tokenización y lotes (celdas 4, 5 y 6)."""

import functools
import json
import math
import random
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from configuracion import Config


# Datos (celda 4) -----------------------------------------------------------------------------------------------------

# Mismo orden que CANONICAL_9_CLASSES en src/data/builder.py del repo EvidenCite.
ETIQUETAS = ["Background", "Uses", "Comparison", "Extends", "Motivation",
             "Future Work", "Evidence", "Originator", "Further Reading"]
ETIQ2ID = {e: i for i, e in enumerate(ETIQUETAS)}
SPLITS = ("train", "dev", "test")
CAMPOS_OBLIGATORIOS = ("instance_id", "label", "annotation_metadata", "citing_paper", "citation_context", "cited_paper")


def _tiene_splits(carpeta: Path) -> bool:
    """True si la carpeta contiene train.jsonl, dev.jsonl y test.jsonl."""
    return all((carpeta / f"{s}.jsonl").is_file() for s in SPLITS)


def _extraer_zip(ruta_zip: Path, destino: Path) -> None:
    """Extrae el zip en `destino`, omitiendo carpetas ocultas y de macOS y rechazando rutas que salgan del destino."""
    destino = destino.resolve()
    with zipfile.ZipFile(ruta_zip) as z:
        for miembro in z.infolist():
            partes = Path(miembro.filename).parts
            if not partes or partes[0] == "__MACOSX" or any(p.startswith(".") for p in partes):
                continue
            objetivo = (destino / miembro.filename).resolve()
            if not objetivo.is_relative_to(destino):
                raise ValueError(f"Ruta insegura dentro del zip: {miembro.filename}")
            z.extract(miembro, destino)


def _buscar_en_zip(ruta_zip: Path) -> Path | None:
    """Extrae el zip junto a sí mismo y devuelve la primera carpeta que tenga los tres splits, o None."""
    destino = ruta_zip.parent / f"{ruta_zip.stem}_extraido"
    _extraer_zip(ruta_zip, destino)
    for carpeta in [destino, *sorted(p for p in destino.rglob("*") if p.is_dir())]:
        if _tiene_splits(carpeta):
            return carpeta
    return None


def localizar_datos(cfg: Config) -> Path:
    """Encuentra la carpeta del dataset: DATOS_DIR si está definido; si no, rutas habituales y dataset.zip."""
    if cfg.datos_dir:
        ruta = Path(cfg.datos_dir)
        if ruta.is_dir() and _tiene_splits(ruta):
            return ruta
        if ruta.is_file() and ruta.suffix == ".zip" and (encontrada := _buscar_en_zip(ruta)):
            return encontrada
        raise FileNotFoundError(f"DATOS_DIR={cfg.datos_dir} no contiene train/dev/test.jsonl")
    for c in ("data/dataset", "dataset", "../data/dataset", "/content/data/dataset", "/content/dataset"):
        if _tiene_splits(Path(c)):
            return Path(c)
    for z in ("dataset.zip", "data/dataset.zip", "/content/dataset.zip"):
        if Path(z).is_file() and (encontrada := _buscar_en_zip(Path(z))):
            return encontrada
    raise FileNotFoundError(
        "No encontré train/dev/test.jsonl. Pon la carpeta en data/dataset/, deja dataset.zip junto al notebook "
        "(en Colab: colab upload -s <sesión> dataset.zip /content/dataset.zip) o define DATOS_DIR.")


def leer_jsonl(ruta: Path) -> list[dict]:
    """Lee un archivo JSON Lines y devuelve una lista de diccionarios (ignora líneas vacías)."""
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def validar_instancias(filas: list[dict], split: str) -> None:
    """Comprueba que cada instancia tenga los campos obligatorios y una etiqueta conocida; si no, lanza ValueError."""
    errores = []
    for f in filas:
        faltan = [c for c in CAMPOS_OBLIGATORIOS if c not in f]
        if faltan:
            errores.append(f"{f.get('instance_id', '?')}: faltan {faltan}")
        elif f["label"] not in ETIQ2ID:
            errores.append(f"{f['instance_id']}: etiqueta desconocida '{f['label']}'")
    if errores:
        raise ValueError(f"{split}: {len(errores)} instancias inválidas. Primeras: {errores[:5]}")


def submuestra_estratificada(filas: list[dict], n: int, semilla: int) -> list[dict]:
    """Toma n instancias repartidas por clase (round-robin), para que el modo mini vea todas las clases."""
    if n <= 0 or n >= len(filas):
        return list(filas)
    rng = random.Random(semilla)
    por_clase: dict[str, list] = {}
    for f in filas:
        por_clase.setdefault(f["label"], []).append(f)
    for lista in por_clase.values():
        rng.shuffle(lista)
    salida = []
    while len(salida) < n:
        for etiqueta in sorted(por_clase):
            if por_clase[etiqueta] and len(salida) < n:
                salida.append(por_clase[etiqueta].pop())
    rng.shuffle(salida)
    return salida


def resumen_split(filas: list[dict]) -> dict:
    """Cuenta instancias por clase y por fuente de un split."""
    return {"n": len(filas),
            "clases": dict(Counter(f["label"] for f in filas).most_common()),
            "fuentes": dict(Counter(f["annotation_metadata"].get("source_dataset", "?") for f in filas).most_common())}


def cargar_datos(cfg: Config) -> dict:
    """Carga y valida train, dev y test; en modo mini toma submuestras estratificadas. Imprime un resumen."""
    carpeta = localizar_datos(cfg)
    datos = {}
    for s in SPLITS:
        filas = leer_jsonl(carpeta / f"{s}.jsonl")
        validar_instancias(filas, s)
        datos[s] = filas
    if cfg.modo == "mini":
        datos["train"] = submuestra_estratificada(datos["train"], cfg.mini_train, cfg.semilla)
        datos["dev"] = submuestra_estratificada(datos["dev"], cfg.mini_dev, cfg.semilla)
        datos["test"] = submuestra_estratificada(datos["test"], cfg.mini_dev, cfg.semilla)
    print(f"  carpeta: {carpeta}")
    for s in SPLITS:
        r = resumen_split(datos[s])
        clases = ", ".join(f"{k} {v}" for k, v in r["clases"].items())
        fuentes = ", ".join(f"{k} {v}" for k, v in r["fuentes"].items())
        print(f"  {s:<6}{r['n']:>5} · {clases} · fuentes: {fuentes}")
    return datos


# Texto de entrada y tokenización (celda 5) ---------------------------------------------------------------------------

# La cabeza lee el último token, así que la secuencia termina en una pregunta fija y lo que se recorta es la
# evidencia del citado, nunca el contexto ni la pregunta. El texto va en inglés, como el preentrenamiento.
PLANTILLA = {
    "citante": "Citing section: {seccion}\nCitation context: {contexto}\n",
    "citante_sin_seccion": "Citation context: {contexto}\n",
    "citado_A": "Cited paper: {titulo}. {abstract}\n",
    "citado_B": "Cited paper excerpts:\n{fragmentos}\n",
    "fragmento": "[{seccion}] {texto}",
    "pregunta": "Citation function:",
}


def partes_texto(inst: dict, cfg: Config) -> tuple[str, str]:
    """Devuelve (texto del citante, texto del citado) según la configuración A o B."""
    contexto = inst["citation_context"].strip()
    seccion = (inst["citing_paper"].get("rhetorical_section") or "").strip()
    if cfg.incluir_seccion and seccion:
        citante = PLANTILLA["citante"].format(seccion=seccion, contexto=contexto)
    else:
        citante = PLANTILLA["citante_sin_seccion"].format(contexto=contexto)
    citado = inst["cited_paper"]
    if cfg.configuracion == "A":
        texto_citado = PLANTILLA["citado_A"].format(titulo=(citado.get("title") or "").strip(),
                                                   abstract=(citado.get("abstract") or "").strip())
    else:
        fragmentos = sorted(citado.get("retrieved_chunks") or [], key=lambda c: c.get("rank", 0))
        texto_citado = PLANTILLA["citado_B"].format(fragmentos="\n".join(
            PLANTILLA["fragmento"].format(seccion=(c.get("section_heading") or "").strip(), texto=c["text"].strip())
            for c in fragmentos))
    return citante, texto_citado


def _especiales_iniciales(tok) -> list[int]:
    """Tokens que el tokenizador antepone por su cuenta (por ejemplo BOS en Llama 3)."""
    con = tok("a")["input_ids"]
    sin = tok("a", add_special_tokens=False)["input_ids"]
    for i in range(len(con) - len(sin) + 1):
        if con[i:i + len(sin)] == sin:
            return con[:i]
    return []


def tokenizar_split(filas: list[dict], tok, cfg: Config) -> list[dict]:
    """Tokeniza un split: [especiales iniciales] + citante + citado + pregunta, sin pasar de max_longitud.

    Si no cabe, recorta el texto del citado; si el contexto solo ya excede el presupuesto, conserva su final.
    Cada ítem guarda su índice, ids, etiqueta, id de instancia, fuente y si se recortó.
    """
    partes = [partes_texto(f, cfg) for f in filas]
    ids_citante = tok([p[0] for p in partes], add_special_tokens=False)["input_ids"]
    ids_citado = tok([p[1] for p in partes], add_special_tokens=False)["input_ids"]
    ids_pregunta = tok(PLANTILLA["pregunta"], add_special_tokens=False)["input_ids"]
    inicio = _especiales_iniciales(tok)
    presupuesto = cfg.max_longitud - len(inicio) - len(ids_pregunta)
    items = []
    for k, (f, a, b) in enumerate(zip(filas, ids_citante, ids_citado)):
        truncado = len(a) + len(b) > presupuesto
        if len(a) > presupuesto:          # contexto excepcionalmente largo: se conserva su final
            a, b = a[-presupuesto:], []
        elif truncado:
            b = b[:presupuesto - len(a)]
        items.append({"indice": k, "input_ids": inicio + a + b + ids_pregunta, "etiqueta": ETIQ2ID[f["label"]],
                      "instance_id": f["instance_id"], "fuente": f["annotation_metadata"].get("source_dataset", "?"),
                      "truncado": truncado})
    return items


def resumen_tokens(items: list[dict]) -> dict:
    """Estadísticas de longitud en tokens de un split: media, p95, máximo y % de ejemplos recortados."""
    largos = np.array([len(x["input_ids"]) for x in items])
    return {"n": len(items), "media": float(largos.mean()), "p95": float(np.percentile(largos, 95)),
            "max": int(largos.max()), "truncados_pct": 100.0 * float(np.mean([x["truncado"] for x in items]))}


# Lotes (celda 6) -----------------------------------------------------------------------------------------------------

class LotesPorLongitud(torch.utils.data.Sampler):
    """Agrupa ejemplos de longitud parecida para rellenar menos.

    Entrenamiento: baraja, ordena por longitud dentro de bloques de `factor` lotes y baraja los lotes; el lote más
    largo va primero para que un problema de memoria aparezca en el paso 1. Evaluación: ordena por longitud.
    """

    def __init__(self, longitudes, tam_lote: int, mezclar: bool, semilla: int = 0, factor: int = 50):
        """Guarda las longitudes de los ejemplos, el tamaño de lote y si hay que barajar (con qué semilla)."""
        super().__init__()
        self.longitudes, self.tam, self.mezclar, self.semilla, self.factor = list(longitudes), tam_lote, mezclar, semilla, factor
        self.epoca = 0

    def __iter__(self):
        """Genera las listas de índices de cada lote; en entrenamiento, con un orden distinto en cada época."""
        idx = list(range(len(self.longitudes)))
        if not self.mezclar:
            idx.sort(key=lambda i: self.longitudes[i], reverse=True)
            yield from (idx[i:i + self.tam] for i in range(0, len(idx), self.tam))
            return
        rng = random.Random(self.semilla + self.epoca)
        self.epoca += 1
        rng.shuffle(idx)
        lotes = []
        bloque = self.tam * self.factor
        for i in range(0, len(idx), bloque):
            orden = sorted(idx[i:i + bloque], key=lambda j: self.longitudes[j], reverse=True)
            lotes += [orden[k:k + self.tam] for k in range(0, len(orden), self.tam)]
        rng.shuffle(lotes)
        mayor = max(range(len(lotes)), key=lambda k: max(self.longitudes[j] for j in lotes[k]))
        lotes[0], lotes[mayor] = lotes[mayor], lotes[0]
        yield from lotes

    def __len__(self):
        """Número de lotes por época."""
        return math.ceil(len(self.longitudes) / self.tam)


class DatosTokenizados(torch.utils.data.Dataset):
    """Dataset de PyTorch sobre la lista de ítems ya tokenizados."""

    def __init__(self, items: list[dict]):
        """Guarda la lista de ítems."""
        self.items = items

    def __len__(self):
        """Número de ítems."""
        return len(self.items)

    def __getitem__(self, i):
        """Devuelve el ítem i tal cual; el relleno lo hace colar()."""
        return self.items[i]


def colar(lote: list[dict], pad_id: int, multiplo: int) -> dict:
    """Arma los tensores del lote con relleno dinámico a la derecha hasta el múltiplo de `multiplo` más cercano."""
    largo = max(len(x["input_ids"]) for x in lote)
    largo = ((largo + multiplo - 1) // multiplo) * multiplo
    ids = torch.full((len(lote), largo), pad_id, dtype=torch.long)
    mascara = torch.zeros((len(lote), largo), dtype=torch.long)
    for i, x in enumerate(lote):
        n = len(x["input_ids"])
        ids[i, :n] = torch.tensor(x["input_ids"], dtype=torch.long)
        mascara[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mascara,
            "labels": torch.tensor([x["etiqueta"] for x in lote], dtype=torch.long),
            "indices": torch.tensor([x["indice"] for x in lote], dtype=torch.long)}


def crear_dataloader(items: list[dict], tam_lote: int, mezclar: bool, cfg: Config, opt: dict, pad_id: int):
    """Crea el DataLoader de un split con lotes agrupados por longitud y relleno dinámico."""
    muestreador = LotesPorLongitud([len(x["input_ids"]) for x in items], tam_lote, mezclar, cfg.semilla)
    return torch.utils.data.DataLoader(
        DatosTokenizados(items), batch_sampler=muestreador,
        collate_fn=functools.partial(colar, pad_id=pad_id, multiplo=opt["multiplo_padding"]),
        num_workers=cfg.num_workers, pin_memory=opt["pin_memory"], persistent_workers=cfg.num_workers > 0)


def mover(lote: dict, opt: dict) -> dict:
    """Mueve los tensores del lote al dispositivo de entrenamiento (de forma asíncrona en CUDA)."""
    return {k: v.to(opt["dispositivo"], non_blocking=opt["non_blocking"]) if torch.is_tensor(v) else v
            for k, v in lote.items()}
