#!/usr/bin/env python3
"""Create private local Granite docker compose env; never print the shared token."""
import argparse
import json
import secrets
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
home = Path.home()


def is_base(path: Path) -> bool:
    return (path / "config.json").is_file() and (path / "tokenizer.json").is_file() and (
        (path / "model.safetensors.index.json").is_file() or (path / "model.safetensors").is_file()
    )


def locate_base():
    candidates = [
        repo / "deploy-assets/granite_base",
        repo.parent / "EvidenCite/deploy-assets/granite_base",
        home / "EvidenCite/deploy-assets/granite_base",
    ]
    cache = home / ".cache/huggingface/hub/models--ibm-granite--granite-4.2-8b/snapshots"
    if cache.exists():
        candidates += sorted(cache.iterdir(), reverse=True)
    return next((p.resolve() for p in candidates if is_base(p)), None)


parser = argparse.ArgumentParser()
parser.add_argument("--base", help="Ruta al directorio original de Granite 4.2 8B (safetensors)")
args = parser.parse_args()
base = Path(args.base).expanduser().resolve() if args.base else locate_base()
if base is None or not is_base(base):
    raise SystemExit("[STOP] No encontré Granite base. Usa --base /ruta/al/modelo (config.json + safetensors + tokenizer.json). No se descargará de nuevo automáticamente.")
backup = home / "evidencite-mlflow-backup/test-revisado"
a = backup / "granite_A/artifacts"
b = backup / "granite_B/artifacts"
for p in (a, b):
    for f in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json"):
        if not (p / f).is_file():
            raise SystemExit(f"[STOP] Falta {p / f}")

env = home / ".config/evidencite/granite-local.env"
env.parent.mkdir(parents=True, exist_ok=True)
if env.exists():
    existing = {}
    for line in env.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                existing[k] = json.loads(v)
            except json.JSONDecodeError:
                existing[k] = v
    token = existing.get("GRANITE_SHARED_TOKEN", "")
    if len(token) < 32:
        raise SystemExit("[STOP] El archivo existente no tiene un token válido. No lo sobreescribo.")
else:
    token = secrets.token_hex(32)

values = {
    "GRANITE_BASE_DIR": str(base),
    "GRANITE_A_DIR": str(a),
    "GRANITE_B_DIR": str(b),
    "GRANITE_SHARED_TOKEN": token,
}
env.write_text("".join(f"{k}={json.dumps(v)}\n" for k, v in values.items()), encoding="utf-8")
env.chmod(0o600)
print("[OK] Config local privada:", env)
print("[OK] Base encontrada:", base)
print("[SAFE] Token creado/reutilizado y no mostrado")
