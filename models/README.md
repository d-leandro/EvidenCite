# Modelos y artefactos de inferencia de EvidenCite

Este directorio documenta y, cuando su tamaño lo permite, distribuye los **artefactos entrenados para la aplicación**, sin publicar datos restringidos, secretos ni decenas de gigabytes de pesos.

> **Importante:** los archivos con extensión `.joblib`, `.pkl` o similares deben cargarse solo desde fuentes de confianza. Pueden ejecutar código durante la deserialización. Los artefactos de este proyecto fueron producidos por el equipo y deben corresponder a las versiones documentadas en `pyproject.toml` y los Dockerfiles.

## Inventario de los ocho clasificadores

| Identificador | Familia | Archivos esperados / formato | Distribución |
|---|---|---|---|
| `tfidf-a` | TF-IDF + LogReg A | `tfidf_A/tfidf_pipeline.joblib`, `input_representation.json` | **Incluir en Git** si pasan revisión de tamaño (script de preparación) |
| `tfidf-b` | TF-IDF + LogReg B | `tfidf_B/tfidf_pipeline.joblib`, `input_representation.json` | **Incluir en Git** si pasan revisión de tamaño (script de preparación) |
| `scibert-a` | SciBERT A | `config.json`, `model.safetensors`, tokenizer y configuración | Checkpoint de entrenamiento en **S3 privado** de EvidenCite |
| `scibert-b` | SciBERT B | Igual que A | Checkpoint de entrenamiento en **S3 privado** de EvidenCite |
| `gemini-a` | Gemini 3.5 Flash-Lite A | Sin checkpoint local; inferencia vía API | Servicio externo, requiere credencial propia |
| `gemini-b` | Gemini 3.5 Flash-Lite B | Sin checkpoint local; inferencia vía API | Servicio externo, requiere credencial propia |
| `granite-a` | Granite 4.2 8B + LoRA A | Base original, `adapter_model.safetensors`, `adapter_config.json`, tokenizer | Base pública de IBM + LoRA del equipo (S3 privado y copia local autorizada) |
| `granite-b` | Granite 4.2 8B + LoRA B | Igual que A | Base pública de IBM + LoRA del equipo (S3 privado y copia local autorizada) |

**No equivale** enlazar el modelo base de Hugging Face a distribuir el checkpoint entrenado: los adaptadores LoRA y los pesos finetuned de SciBERT forman parte de los resultados del equipo y tienen su propio almacenamiento/versionado.

### Referencias

- [Modelo base IBM Granite 4.2 8B en Hugging Face](https://huggingface.co/ibm-granite/granite-4.2-8b) (no incluye los adaptadores del proyecto).
- [SciBERT original, modelo de base](https://huggingface.co/allenai/scibert_scivocab_uncased) (no incluye el ajuste de EvidenCite).
- Checkpoints entrenados del equipo, bucket S3 **privado**: `s3://evidencite-demo-<account-id>-models/{tfidf_A,tfidf_B,scibert_A,scibert_B,granite_A,granite_B}/`. Este URI es **un catálogo de ubicación, no un enlace público de descarga**. Se requieren credenciales/autorización AWS Academy.

Se puede publicar posteriormente una copia de checkpoints grandes en una organización o repositorio Hugging Face con control de acceso y licencia aprobada por el equipo. **No existe todavía un repositorio público de pesos finetuned verificado y por eso no inventamos uno.**

## Preparar los modelos ligeros para la entrega

En WSL desde la raíz del repositorio:

```bash
export AWS_PROFILE=evidencite-lab
export AWS_REGION=us-east-1
bash tools/prepare_delivery_models.sh
```

El script:

1. **Prefiere** artefactos existentes en `deploy-assets/tfidf_A` y `deploy-assets/tfidf_B`.
2. Si faltan, intenta recuperarlos del bucket privado S3 asociado a la cuenta actual, **sin modificar el bucket**.
3. Verifica `tfidf_pipeline.joblib` y `input_representation.json`.
4. Impone un límite de **45 MiB por archivo** para evitar rechazos de GitHub y no publicar pesos masivos por accidente.
5. Copia solo archivos permitidos a `models/tfidf_A` y `models/tfidf_B`, calcula SHA-256 y sugiere el `git add -f` explícito.

**El script no hace commit, no modifica el despliegue y nunca publica secretos o datos.** Si un artefacto excede el límite, no debe subirse a Git por fuerza: usa almacenamiento externo autorizado y registra hash, tamaño y acceso aquí.

Para verificar los archivos que se hayan incluido:

```bash
find models/tfidf_A models/tfidf_B -maxdepth 1 -type f -printf '%p %s bytes\n'
sha256sum models/tfidf_*/tfidf_pipeline.joblib
```

## Reproducibilidad y versiones

- Los pesos grandes no se copian al repositorio de GitHub por su tamaño, uso de memoria y condiciones de acceso.
- Los modelos de producción de AWS se descargan desde S3 mediante `tools/ec2_academy_deploy.sh`.
- Granite local monta la base y LoRA A/B como volúmenes de solo lectura con `compose.granite-local.yaml`.
- La aplicación puede verificar disponibilidad de cada modelo mediante `/api/v1/models`.
- Los detalles de entrenamiento y los hiperparámetros están en `notebooks/`, `configs/`, el panel **Entrenamiento** y `docs/manual-instalacion.md`.

**Pendiente de coordinación del grupo:** confirmar si se autoriza publicar checkpoints grandes y muestras de datos en un repositorio externo con licencia y un esquema de acceso definido. No subir copias a servicios personales sin esa aprobación.
