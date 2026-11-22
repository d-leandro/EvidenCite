#!/usr/bin/env python3
"""Preflight for EvidenCite AWS deployment. Pure file inspection: no network/AWS/Docker.

--report: CI informational check, exit 0
--strict: deployment gate, exit nonzero until the full stack is represented
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def present(path: str) -> bool:
    return (ROOT / path).is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    blockers: list[str] = []
    cautions: list[str] = []

    ecs_path = ROOT / "infra/evidencite/ecs.tf"
    ecs = ecs_path.read_text(encoding="utf-8") if ecs_path.is_file() else ""
    compose_path = ROOT / "compose.yaml"
    compose = compose_path.read_text(encoding="utf-8") if compose_path.is_file() else ""

    if not re.search(r"(?m)^\s{2}granite:\s*$", compose):
        blockers.append("El Compose publicado no declara granite; faltan cambios locales por subir.")

    if not re.search(r'resource\s+"aws_(?:ecs_task_definition|ecs_service|instance)"\s+"granite"', ecs):
        blockers.append("Terraform aún no declara la ejecución de Granite en AWS.")

    # The current Dockerfile copies these paths into the CI-produced image.
    for item in (
        "deploy-assets/scibert_A/config.json",
        "deploy-assets/scibert_B/config.json",
        "deploy-assets/tfidf_A/tfidf_pipeline.joblib",
        "deploy-assets/tfidf_B/tfidf_pipeline.joblib",
    ):
        if not present(item):
            blockers.append(f"Artefacto requerido para docker build ausente: {item}")

    if "GEMINI_API_KEY" not in ecs:
        blockers.append("La definición ECS no inyecta GEMINI_API_KEY desde un secreto.")

    if "EVIDENCITE_METRICS_DB" not in ecs:
        cautions.append("No se ve la ruta de SQLite de métricas en ECS: revisar persistencia.")
    if ":latest" in ecs:
        cautions.append("Terraform referencia etiquetas ECR :latest: conviene usar un SHA inmutable en producción.")
    if not present("app/static/training.js"):
        cautions.append("La pestaña Entrenamiento no está aún publicada en esta copia del repo.")

    print("=== EvidenCite - preparación para AWS (SIN crear recursos) ===")
    for message in blockers:
        print("[BLOCKER]", message)
    for message in cautions:
        print("[NOTE]", message)
    if not blockers:
        print("[OK] No se detectaron bloqueos en estas verificaciones estáticas.")
    print("Estas verificaciones no sustituyen una revisión de arquitectura, permisos ni costos.")
    if blockers and args.strict:
        print("[STOP] Despliegue bloqueado para evitar publicar un EvidenCite incompleto.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
