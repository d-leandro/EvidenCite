---
layout: default
title: Manual de usuario — EvidenCite
---

# Manual de usuario — EvidenCite

## 1. Propósito y alcance

EvidenCite identifica la **función retórica que cumple una referencia académica** en el contexto de un texto científico en inglés. Distingue nueve clases y permite comparar modelos usando distintas fuentes de información sobre el artículo citado.

La predicción es una **herramienta de apoyo**, no una comprobación de la veracidad de un artículo ni un sustituto de la interpretación de un experto. Las probabilidades mostradas no deben interpretarse automáticamente como garantías de exactitud.

## 2. Interfaz

La pantalla principal dispone de un selector de configuración (A/B), selector de modelo, entrada de contexto, cita objetivo, métodos para identificar o cargar el artículo citado y un botón de clasificación. También incluye:

- **Análisis de Citas:** entrada y resultado de una predicción.
- **Tiempos y costos:** mediciones de inferencia disponibles.
- **Entrenamiento:** registros y visualizaciones experimentales publicados en la aplicación.
- **Ayuda:** acceso a los manuales de usuario e instalación.

La disponibilidad de modelos se consulta en `GET /api/v1/models`; los que no estén disponibles no pueden utilizarse.

## 3. Prueba rápida: ejemplo guiado

1. En la sección de entrada despliega **¿Necesitas un ejemplo?**.
2. Lee el contexto ilustrativo relacionado con *Attention Is All You Need* (Vaswani et al., 2017).
3. Pulsa **Cargar ejemplo**. Se completan el contexto y la cita objetivo, y se coloca el identificador `1706.03762` en la búsqueda del artículo.
4. Pulsa **Buscar** y, cuando aparezcan los metadatos, selecciona **Confirmar artículo**.
5. Selecciona Configuración A o B y sigue las instrucciones correspondientes.
6. Presiona **Clasificar intención de cita** y revisa la etiqueta y el desglose de probabilidades.

El texto de contexto se redactó para demostrar el flujo; **no procede de un artículo citante real**. Puedes abrir el [PDF original de Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762) para comprobar el procesamiento de documentos.

### 3.1 Contexto de muestra

> Previous sequence-to-sequence architectures relied heavily on recurrent neural networks. Vaswani et al. (2017) introduced the Transformer architecture, replacing recurrence with self-attention mechanisms. Subsequent studies expanded upon this approach.

**Cita objetivo:** `Vaswani et al. (2017)`  
**Artículo citado:** *Attention Is All You Need*  
**arXiv:** `1706.03762`

## 4. Configuraciones A y B

### Configuración A — Título y resumen

La clasificación utiliza el contexto citante, la referencia objetivo y **título + resumen** del artículo citado. Puedes buscar el artículo por título exacto o identificador de arXiv y confirmarlo; también puedes proporcionar los metadatos manualmente o extraerlos mediante el PDF.

Antes de clasificar, comprueba que el título y el resumen correspondan al artículo realmente citado.

### Configuración B — Evidencia local

Además del contexto y la referencia objetivo, requiere **tres fragmentos relevantes** del texto del artículo citado. Tras buscar y confirmar un artículo con identificador arXiv, utiliza **Recuperar los 3 fragmentos**. El sistema recupera el texto y ordena fragmentos con SciBERT. Otra opción es subir el PDF del artículo para extraer la evidencia.

La Configuración B no se completa únicamente con el título y resumen; es necesario obtener fragmentos válidos antes de predecir. La recuperación puede tardar dependiendo del documento y del servicio.

## 5. Cómo analizar una cita propia

1. Elige **A** o **B** según la evidencia que quieras utilizar.
2. Selecciona un modelo **disponible** en la lista superior.
3. Introduce un contexto escrito en inglés, con al menos 20 caracteres, que contenga claramente la referencia objetivo.
4. Verifica la **cita objetivo** que detecta el formulario, o corrígela manualmente.
5. Busca y confirma el artículo citado, completa sus metadatos (A) o recupera fragmentos (B).
6. Pulsa **Clasificar intención de cita**. Espera el resultado antes de repetir la operación.

### Búsqueda por arXiv

Para el artículo de ejemplo, escribe `1706.03762` o `Attention Is All You Need` y pulsa **Buscar**. Verifica los resultados y confirma el documento. En modo B, después recupera los tres fragmentos.

### Procesar PDF

Usa **Sube el PDF del artículo** para cargar un archivo de hasta **20 MB**. Antes de procesarlo confirma la casilla que indica que el PDF corresponde a la cita objetivo: **EvidenCite no verifica automáticamente esa correspondencia**.

En A se obtienen título y resumen; en B se buscan fragmentos de evidencia. La extracción de texto puede tomar varios minutos y depende de la calidad del PDF. No subas documentos restringidos o confidenciales sin autorización.

## 6. Interpretación de resultados

La aplicación muestra la función principal predicha, un indicador de confianza y la distribución de probabilidades entre las nueve clases. Estas cantidades son **salidas del modelo**, no un dictamen definitivo de corrección. Las diferentes familias pueden producir resultados distintos ante el mismo contexto.

| Etiqueta | Interpretación |
|---|---|
| **Background** | Antecedentes, definiciones o contextualización general del problema. |
| **Gap** | Limitación, problema pendiente o necesidad de investigación. |
| **Basis** | Fundamento teórico o metodológico sobre el que se construye el trabajo citante. |
| **Comparison** | Contraste de enfoques, métodos, resultados o rendimiento. |
| **Application** | Uso directo de una herramienta, dato, método o algoritmo del trabajo citado. |
| **Improvement** | Extensión, modificación, refinamiento u optimización de un trabajo previo. |
| **Evidence** | Hechos o resultados empíricos citados para sustentar una afirmación. |
| **Originator** | Reconocimiento de quien introdujo un concepto o método original. |
| **Further Reading** | Orientación hacia materiales adicionales para profundizar. |

## 7. Modelos y disponibilidad

La API registra `tfidf-a/b`, `scibert-a/b`, `gemini-a/b` y `granite-a/b`. La instancia AWS actual se preparó para **TF-IDF, SciBERT y Gemini** en A/B. **Granite se incorporará posteriormente** y no debe interpretarse como disponible mientras la API no lo confirme.

Gemini necesita comunicación con un proveedor externo, por lo que su latencia y disponibilidad pueden diferir de las de los modelos locales. La selección de modelo no altera automáticamente las entradas de A/B que exige la aplicación.

## 8. Métricas y entrenamiento

**Tiempos y costos** muestra ejecuciones registradas, latencias y costos estimados cuando existe una tarifa configurada. Los datos pueden depender de si la sesión utiliza caché y de qué componente haya medido la latencia. Las métricas registradas no garantizan capacidad ni costo idénticos bajo carga elevada.

La sección **Entrenamiento** presenta información de los experimentos y modelos cuando se encuentra integrada. No equivale a entrenar modelos desde el navegador.

## 9. Preguntas frecuentes y errores

- **No aparecen modelos:** puede que la API o sus artefactos no estén disponibles. Recarga la página y comprueba el estado del servidor.
- **No deja clasificar en A:** verifica que haya título y resumen.
- **No deja clasificar en B:** primero recupera los tres fragmentos.
- **No encuentra el artículo:** utiliza el ID de arXiv o el título completo y confirma los metadatos antes de continuar.
- **El PDF no se procesa:** comprueba que sea PDF válido, que pese menos de 20 MB y que el servidor esté disponible.
- **Error HTTP 429:** existe un límite de solicitudes de predicción para proteger el servicio. Espera antes de intentar nuevamente; varios usuarios pueden compartir la misma IP pública.
- **La URL dejó de responder:** la EC2 puede haberse detenido o cambiado de dirección IP. Solicita la URL actual al equipo del proyecto.
- **El resultado parece incorrecto:** revisa el contexto, la cita objetivo y que los metadatos o fragmentos realmente pertenezcan al artículo citado.

## 10. Uso directo de la API

Si el despliegue es accesible, consulta `GET /health`, `GET /api/v1/models` y `GET /api/v1/labels` para comprobar el servicio. La documentación interactiva de FastAPI y las demás rutas dependen de la configuración pública del gateway.

## 11. Limitaciones y buenas prácticas

El prototipo trabaja principalmente con citas científicas en inglés. El rendimiento depende de la calidad del contexto, de la correspondencia entre referencia y documento citado, de la distribución de clases y del modelo. Los ejemplos inventados solo demuestran el funcionamiento del sistema: **no constituyen evidencia cuantitativa de su desempeño**. Evita consultas automatizadas masivas durante las jornadas de evaluación.

[Volver al índice](index.html) · [Manual de instalación](manual-instalacion.html)
