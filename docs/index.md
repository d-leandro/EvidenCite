---
layout: default
title: Documentación de EvidenCite
---

# EvidenCite — Documentación

**Clasificador de funciones retóricas de citas científicas**, con dos representaciones del artículo citado: **Configuración A** (título y resumen) y **Configuración B** (fragmentos de evidencia recuperados del texto completo).

Este sitio acompaña a la aplicación para que evaluadores y desarrolladores puedan reproducir un análisis, comprender las limitaciones de cada modelo e instalar el sistema.

## Manuales

- **[Manual de usuario](manual-usuario.html):** ejemplo guiado, búsqueda de artículos, selección de modelos, carga de PDF, interpretación de resultados, métricas y solución de problemas.
- **[Manual de instalación y despliegue](manual-instalacion.html):** arquitectura, Docker Compose, Terraform, AWS EC2, S3, Systems Manager, GitHub Actions, validaciones y destrucción segura.

## Inicio rápido para evaluadores

1. Abre la aplicación EvidenCite que te haya facilitado el equipo del proyecto.
2. En **Análisis de Citas**, despliega **¿Necesitas un ejemplo?** y selecciona **Cargar ejemplo**.
3. Confirma el artículo *Attention Is All You Need* (Vaswani et al., 2017; arXiv:1706.03762) después de recuperarlo con la búsqueda.
4. Selecciona un modelo disponible en Configuración A o B. En B, recupera los tres fragmentos de evidencia antes de clasificar.
5. Ejecuta **Clasificar intención de cita**, revisa las nueve probabilidades y compara resultados si lo deseas.

El ejemplo contiene un **contexto ilustrativo redactado para esta demostración**, no una cita textual de otra publicación. [PDF oficial del artículo de referencia](https://arxiv.org/pdf/1706.03762).

## Modelos

| Familia | Configuración A | Configuración B | Estado en despliegue EC2 actual |
|---|---|---|---|
| TF-IDF + Regresión logística | Sí | Sí | Disponible |
| SciBERT | Sí | Sí | Disponible |
| Gemini 3.5 Flash-Lite | Sí | Sí | Disponible con API key válida |
| Granite 4.2 8B + LoRA | Previsto | Previsto | No desplegado en EC2 de 8 GiB |

La disponibilidad exacta depende del estado de la API: `GET /api/v1/models`. La aplicación solo ofrece al usuario modelos marcados como disponibles.

## Taxonomía

EvidenCite utiliza nueve etiquetas: **Background, Gap, Basis, Comparison, Application, Improvement, Evidence, Originator y Further Reading**. Consulta el manual de usuario para sus definiciones y limitaciones.

## Arquitectura publicada en AWS

`Navegador → Caddy (HTTPS) → Nginx (gateway) → Frontend / FastAPI → modelos locales o Gemini`. El host es una EC2 administrada con Terraform y SSM; los artefactos TF-IDF y SciBERT se conservan en S3. **No utiliza ECS, ECR ni ALB** en el despliegue de evaluación actual.

## Repositorio

[Código fuente de EvidenCite](https://github.com/d-leandro/EvidenCite) · [Artículo de ejemplo en arXiv](https://arxiv.org/abs/1706.03762)
