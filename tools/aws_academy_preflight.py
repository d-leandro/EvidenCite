#!/usr/bin/env python3
"""Read-only AWS Academy preflight. Never launches, stops or changes resources."""
from __future__ import annotations

import json
import os
import subprocess
import sys

REGION = os.environ.get("AWS_REGION", "us-east-1")
INSTANCE_TYPE = os.environ.get("EVIDENCITE_INSTANCE_TYPE", "t3.large")
PROFILE = os.environ.get("EVIDENCITE_INSTANCE_PROFILE", "LabInstanceProfile")


def aws(*args: str) -> tuple[dict | list | None, str | None]:
    cmd = ["aws", *args, "--region", REGION, "--output", "json"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
    except FileNotFoundError:
        return None, "AWS CLI no está instalado."
    except subprocess.TimeoutExpired:
        return None, "Consulta AWS agotó el tiempo."
    if res.returncode != 0:
        # Don't print CLI stderr: it could contain local credential details.
        return None, "Acceso denegado, credenciales vencidas o recurso no disponible."
    try:
        return json.loads(res.stdout), None
    except json.JSONDecodeError:
        return None, "Respuesta inesperada de AWS."


def main() -> int:
    print("=== EvidenCite EC2: AWS Academy preflight (solo lectura) ===")
    ident, error = aws("sts", "get-caller-identity")
    if error:
        print("[STOP] STS:", error)
        return 2
    print("[OK] Cuenta:", ident.get("Account"), "Región:", REGION)

    vpcs, error = aws("ec2", "describe-vpcs", "--filters", "Name=isDefault,Values=true")
    if error or not vpcs.get("Vpcs"):
        print("[STOP] Falta VPC predeterminada o permiso para consultarla.")
        return 2
    print("[OK] VPC predeterminada:", vpcs["Vpcs"][0]["VpcId"])

    offering, error = aws("ec2", "describe-instance-type-offerings", "--location-type", "region",
                          "--filters", f"Name=instance-type,Values={INSTANCE_TYPE}")
    if error or not offering.get("InstanceTypeOfferings"):
        print("[STOP] La región no ofrece", INSTANCE_TYPE, "o faltan permisos.")
        return 2
    print("[OK] Tipo disponible en la región:", INSTANCE_TYPE)
    print("[WARN] Disponibilidad no significa permiso del Learner Lab; consulte límites del laboratorio.")

    metadata, error = aws("ec2", "describe-instance-types", "--instance-types", INSTANCE_TYPE)
    if error:
        print("[WARN] No se pudieron verificar memoria y arquitectura.")
    else:
        item = metadata["InstanceTypes"][0]
        mem = item["MemoryInfo"]["SizeInMiB"] / 1024
        archs = item["ProcessorInfo"]["SupportedArchitectures"]
        print(f"[INFO] {item['VCpuInfo']['DefaultVCpus']} vCPU; {mem:g} GiB RAM; arquitecturas: {', '.join(archs)}")
        if mem < 8 or "x86_64" not in archs:
            print("[STOP] Se requiere CPU x86_64 y al menos 8 GiB de RAM.")
            return 2

    prof, error = aws("iam", "get-instance-profile", "--instance-profile-name", PROFILE)
    if error:
        print("[WARN] No pude verificar IAM", PROFILE, "; confirmar desde consola.")
    else:
        print("[OK] Perfil IAM:", prof.get("InstanceProfile", {}).get("InstanceProfileName"))
    print("[NEXT] Revisar cuota EC2 y presupuesto de AWS Academy antes de terraform apply.")
    print("[SAFE] No se han creado ni modificado recursos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
