"""Guardado de la corrida: adaptador, tokenizador, JSONs y predicciones en una carpeta y su zip (celda 11)."""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file

from configuracion import Config
from variantes_lora import VarianteLoRA


def _a_json(obj):
    """Convierte a algo serializable los objetos que json no sabe escribir (dtype, Path, números de numpy)."""
    if isinstance(obj, (torch.dtype, Path)):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)


def _reponer_modulos_guardados(modelo, carpeta_adaptador: Path) -> None:
    """La conversión PiSSA/OLoRA → LoRA de PEFT 0.21 omite los modules_to_save (la cabeza entrenada).

    Sin ellos el adaptador no se puede recargar; aquí se vuelven a escribir en el mismo archivo.
    """
    archivo = carpeta_adaptador / "adapter_model.safetensors"
    convertido = load_file(archivo)
    faltan = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(modelo).items()
              if k not in convertido and ".lora_" not in k}
    if faltan:
        convertido.update(faltan)
        save_file(convertido, archivo, metadata={"format": "pt"})


def guardar_corrida(cfg: Config, modelo, tok, var: VarianteLoRA, dir_init: Path | None, archivos: dict) -> Path:
    """Guarda adaptador, tokenizador, JSONs y predicciones en una carpeta por corrida y la empaqueta en un zip.

    PiSSA y OLoRA se guardan convertidos a LoRA estándar, para cargarlos sobre la base original.
    """
    base = Path(cfg.salidas_dir)
    carpeta = base / cfg.nombre_corrida
    carpeta.mkdir(parents=True, exist_ok=True)
    conversion = str(dir_init) if (var.muta_base and dir_init is not None) else None
    modelo.save_pretrained(carpeta / "adaptador", path_initial_model_for_weight_conversion=conversion)
    if conversion:
        _reponer_modulos_guardados(modelo, carpeta / "adaptador")
    tok.save_pretrained(carpeta / "adaptador")
    for nombre, contenido in archivos.items():
        with open(carpeta / nombre, "w", encoding="utf-8") as f:
            if nombre.endswith(".jsonl"):
                f.writelines(json.dumps(fila, ensure_ascii=False, default=_a_json) + "\n" for fila in contenido)
            else:
                json.dump(contenido, f, ensure_ascii=False, indent=2, default=_a_json)
    zip_corrida = Path(shutil.make_archive(str(base / cfg.nombre_corrida), "zip", root_dir=carpeta))
    shutil.copyfile(zip_corrida, base / "ultima_corrida.zip")
    return zip_corrida
