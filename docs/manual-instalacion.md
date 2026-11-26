---
layout: default
title: Manual de instalación — EvidenCite
---

# Manual de instalación y despliegue — EvidenCite

## 1. Alcance

Esta guía describe el despliegue actual de EvidenCite con **una instancia Amazon EC2**, contenedores Docker, FastAPI, Nginx y Caddy. Terraform administra la infraestructura, AWS Systems Manager (SSM) ejecuta comandos remotos y Parameter Store protege la credencial de Gemini. S3 guarda los artefactos de TF-IDF y SciBERT y el estado remoto de Terraform.

Los manuales del proyecto previo IntenCite utilizaban ECS/Fargate, ECR y ALB; **esos componentes no forman parte del despliegue AWS actual**. No uses instrucciones de la infraestructura anterior para desplegar ni destruir EvidenCite.

## 2. Arquitectura

```text
Usuario (navegador)
    |
    v HTTPS (443)
AWS EC2 — Caddy (certificado HTTPS automático)
    |
    v
Nginx Gateway — rutas / y /api/
    +--> Frontend estático (Nginx)
    +--> FastAPI (puerto 8000, red Docker)
           +--> TF-IDF A/B — artefactos locales descargados de S3
           +--> SciBERT A/B — artefactos locales descargados de S3
           +--> Gemini A/B — API externa, credencial en SSM Parameter Store
           +--> Granite A/B — NO desplegado en esta EC2 de 8 GiB
```

Terraform configura la EC2, un Security Group que permite 80/443, el disco gp3 cifrado y el rol IAM `LabInstanceProfile`. **No se expone SSH**. El administrador ejecuta comandos por SSM Run Command. La IP pública es dinámica y se incorpora al dominio `evidencite.<ip-con-guiones>.sslip.io`.

## 3. Archivos y responsabilidades

| Archivo / carpeta | Finalidad |
|---|---|
| `infra/evidencite/` | Terraform de la EC2 actual. |
| `compose.aws.yaml` | Servicios API, frontend, gateway y Caddy en AWS. |
| `compose.yaml` | Desarrollo local; puede requerir pesos y adaptadores no incluidos en el repositorio. |
| `compose.cloud.yaml` | Referencia de despliegue alternativo para Granite; no lo usa la pila EC2 actual. |
| `infra/evidencite/bootstrap.sh` | Configura Docker, Compose y Buildx al iniciar una EC2 nueva. |
| `tools/academy_rebuild.sh` | Crea o reutiliza la instancia, ejecuta el despliegue SSM y comprueba HTTPS. |
| `tools/ec2_academy_deploy.sh` | Recupera secretos y modelos, construye contenedores y verifica modelos. |
| `.github/workflows/deploy.yml` | Despliegue EC2 activado manualmente. |
| `.github/workflows/destroy.yml` | Destrucción EC2 con confirmación explícita. |
| `.github/workflows/validate.yml` | Comprobaciones automáticas de IaC y Compose. |

## 4. Requisitos

- Repositorio `d-leandro/EvidenCite` y acceso a la rama de despliegue (`feature/app-infrastructure` mientras no esté fusionada con `main`).
- AWS Academy Learner Lab con AWS CLI autenticado y Terraform 1.11 o superior.
- Región `us-east-1`, VPC predeterminada, `LabInstanceProfile` con permisos suficientes y cuotas apropiadas.
- Bucket persistente de modelos `evidencite-demo-<account-id>-models` con las carpetas `tfidf_A`, `tfidf_B`, `scibert_A`, `scibert_B`.
- Bucket de estado remoto `evidencite-tfstate-<account-id>`.
- Parámetro seguro `/evidencite/prod/gemini-api-key` en Parameter Store, accesible desde el rol de EC2.
- Para GitHub Actions: secretos `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` de la sesión AWS Academy activa. **No se deben copiar secretos al repositorio.**

Las credenciales temporales de AWS Academy pueden caducar. Renueva los secretos de Actions antes de desplegar o destruir si la sesión cambió.

## 5. Desarrollo local

```bash
git clone https://github.com/d-leandro/EvidenCite.git
cd EvidenCite
docker compose -f compose.yaml config --quiet
```

El archivo `compose.yaml` local contiene una configuración Granite que utiliza artefactos externos y rutas de la máquina del desarrollador. No ejecutes `docker compose up` hasta verificar los modelos, volúmenes y credenciales requeridos por tu ambiente. En Windows con WSL2, activa Docker Desktop → Settings → Resources → WSL Integration.

Para comprobar la sintaxis de AWS Compose, sin levantar contenedores ni exponer credenciales reales:

```bash
EVIDENCITE_DOMAIN=evidencite.example.com \
GEMINI_API_KEY=placeholder \
docker compose -f compose.aws.yaml config --quiet
```

## 6. Estado persistente y seguridad

El estado Terraform usa **una clave histórica intencional**, que se conserva aunque la carpeta haya cambiado de nombre:

```text
s3://evidencite-tfstate-<account-id>/evidencite/ec2-demo.tfstate
```

**No cambies esa clave ni la reemplaces por un nuevo estado sin una migración explícita.** El bucket de modelos, sus protecciones y el parámetro SSM se administran fuera de la pila EC2 efímera. Antes de destruir revisa el estado y confirma que no incluya recursos persistentes de S3/SSM. No publiques el archivo `.env` ni el contenido de Parameter Store.

## 7. Despliegue desde terminal

En la raíz del repositorio, trabajando en `feature/app-infrastructure` con el commit aprobado ya publicado:

```bash
export AWS_PROFILE=evidencite-lab
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 AWS_PAGER=''
aws sts get-caller-identity
python3 tools/aws_academy_preflight.py
```

Ejecuta el despliegue **solo tras verificar presupuesto, instancia y credenciales**:

```bash
bash tools/academy_rebuild.sh --approve-costs
```

El script inicializa Terraform con el estado remoto existente, crea o reutiliza EC2, espera a SSM, descarga el commit aprobado del repositorio, obtiene modelos/credenciales, construye contenedores y comprueba HTTPS `/health`. El dominio se calcula a partir de la IP actual de EC2.

### Comprobación funcional

```bash
terraform -chdir=infra/evidencite output -raw public_url
# Sustituye URL por la salida anterior:
curl --fail --silent --show-error 'https://URL/health'
curl --fail --silent --show-error 'https://URL/api/v1/models'
```

En `/api/v1/models` deben figurar disponibles TF-IDF A/B, SciBERT A/B y Gemini A/B; Granite continuará no disponible hasta configurar su servicio de inferencia separado.

## 8. GitHub Actions

Desde GitHub → Actions selecciona **Deploy EvidenCite AWS EC2** y ejecuta el workflow manual con la confirmación exacta `DEPLOY_EVIDENCITE`. Este proceso crea o reutiliza la EC2, instala la app por SSM y comprueba la URL pública. Es necesaria la configuración actualizada de secretos AWS Academy.

El workflow **Validate EvidenCite AWS** verifica Terraform, configuración Compose y sintaxis sin desplegar recursos; se dispara al modificar rutas relevantes en `main` o ramas `feature/**`.

**Precaución:** los workflows de despliegue y destrucción deben probarse contra la cuenta de laboratorio antes de considerarse operativos en una entrega. No son garantía de éxito si cambian permisos, cuotas o credenciales.

## 9. Detener, actualizar y destruir

### Detener la instancia

Detener EC2 ahorra cómputo, pero **el volumen EBS continúa facturándose** y, al encender nuevamente, puede cambiar la IP pública. Vuelve a ejecutar el script de despliegue para recalcular el dominio y reconfigurar Caddy. Si necesitas URL estable para una evaluación de pocos días, es preferible mantener encendida la instancia base durante esa ventana.

### Actualizar el código

Actualiza y publica la rama aprobada. El script actual exige que el commit local corresponda al commit remoto de `feature/app-infrastructure`; evita ejecutar el deploy desde un commit distinto. Luego ejecuta nuevamente `academy_rebuild.sh` o el workflow de despliegue cuando esté habilitado.

### Destruir la pila EC2

Respalda primero las métricas que quieras conservar: `/srv/evidencite/metrics/` reside en el volumen raíz, configurado para eliminarse con la instancia.

Workflow manual **Destroy EvidenCite AWS EC2**, confirmación exacta `DESTROY_EVIDENCITE`. Verifica que el estado Terraform apunta únicamente a EC2 y sus recursos efímeros. Alternativamente, tras inicializar el backend remoto correcto y verificar su contenido:

```bash
terraform -chdir=infra/evidencite state list
terraform -chdir=infra/evidencite destroy -var='allow_paid_resources=true'
```

Los buckets S3 persistentes y el parámetro SSM no deben destruirse con esta pila. El workflow exige una comprobación del estado antes del `destroy`.

## 10. Problemas frecuentes

| Síntoma | Diagnóstico |
|---|---|
| `docker` no responde en WSL2 | Inicia Docker Desktop y habilita su integración WSL; valida `docker version`. |
| `ExpiredToken` / permisos AWS | Reinicia el laboratorio y actualiza credenciales locales o Secrets de Actions. |
| Terraform muestra recursos inesperados | Verifica que el backend apunte a `evidencite/ec2-demo.tfstate`; no hagas `apply` o `destroy` hasta investigar. |
| SSM no alcanza `Online` | Revisa agente SSM, rol de EC2, red de salida y permisos. |
| `/health` no responde | Revisa el comando remoto SSM, contenedores, gateway y certificado Caddy. |
| Un modelo aparece como no disponible | Comprueba el contenido S3, los montajes del contenedor o el acceso a Gemini. |
| `429 Too Many Requests` | Nginx aplica un límite por IP pública; varios evaluadores pueden compartir IP. |
| El enlace `sslip.io` dejó de funcionar | Tras reiniciar EC2, recalcula la URL con la nueva IP y despliega de nuevo. |
| El costo supera lo esperado | Revisa EC2, EBS, IPv4 pública, S3, transferencias y APIs de proveedores por separado. |

## 11. Granite y alcance actual

Granite 4.2 8B + LoRA A/B existe en el registro de modelos, pero la pila `compose.aws.yaml` actual **no despliega su servicio**. Debe integrarse en una etapa posterior, dimensionando recursos de inferencia y presupuesto. No afirmes que Granite está operativo en AWS antes de probarlo realmente.

## 12. Lista de verificación para entrega

- [ ] `Validate EvidenCite AWS` en verde sobre el commit aprobado.
- [ ] Secrets AWS Academy vigentes.
- [ ] Deploy EC2 exitoso y dominio HTTPS accesible.
- [ ] `/health` devuelve respuesta correcta.
- [ ] Seis configuraciones A/B disponibles (TF-IDF, SciBERT, Gemini).
- [ ] Ejemplo guiado y extracción de PDF probados desde navegador.
- [ ] Documentación publicada en GitHub Pages.
- [ ] Granite A/B desplegado o claramente indicado como no disponible.
- [ ] Prueba de `destroy` verificada sin eliminar S3/SSM.

[Volver al índice](index.html) · [Manual de usuario](manual-usuario.html)
