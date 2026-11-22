const form = document.getElementById("prediction-form");
const contextInput = document.getElementById("context");
const targetInput = document.getElementById("target-citation");
const modelSelector = document.getElementById("model-selector");
const submitButton = document.getElementById("submit-button");
const resultSection = document.getElementById("result-section");
const errorElement = document.getElementById("error-message");

const API = "/api/v1";

// Indicadores de inferencia.
const progressEl = document.createElement("p");
progressEl.id = "prediction-progress";
progressEl.className = "prediction-progress hidden";
progressEl.setAttribute("role", "timer");
progressEl.setAttribute("aria-live", "off");
submitButton.closest(".form-footer")
    .insertAdjacentElement("afterend", progressEl);

const latencyEl = document.createElement("p");
latencyEl.id = "prediction-latency";
latencyEl.className = "prediction-latency hidden";
resultSection.appendChild(latencyEl);

// Configuration A/B controls the visible model selection.
let uiConfiguration = "A";
let uiModels = [];
const lastModelForConfiguration = {A: null, B: null};

const configurationButtons =
    document.querySelectorAll("[data-config-choice]");

modelSelector.addEventListener("change", () => {
    const model = uiModels.find(item => item.id === modelSelector.value);

    if (model && (model.configuration === "A" ||
                  model.configuration === "B")) {
        lastModelForConfiguration[model.configuration] = model.id;
    }
});

const nombresModelosEs = {
    "tfidf-a": "TF-IDF + Regresión logística (A)",
    "tfidf-b": "TF-IDF + Regresión logística (B)",
    "scibert-a": "SciBERT (Configuración A)",
    "scibert-b": "SciBERT (Configuración B)"
};

function setUiConfiguration(mode) {
    if (mode !== "A" && mode !== "B") return;

    const currentModel = uiModels.find(
        item => item.id === modelSelector.value
    );

    if (currentModel?.configuration === uiConfiguration) {
        lastModelForConfiguration[uiConfiguration] = currentModel.id;
    }

    const previousMode = uiConfiguration;
    uiConfiguration = mode;
    document.body.dataset.configuration = mode;

    for (const button of configurationButtons) {
        const active = button.dataset.configChoice === mode;
        button.classList.toggle("active", active);
        button.setAttribute("aria-pressed", String(active));
    }

    document.getElementById("configuration-indicator").textContent =
        `CONFIGURACIÓN ${mode}`;

    document.getElementById("configuration-description").textContent =
        mode === "A"
            ? "Necesitas el contexto de la cita, la cita objetivo, el título y el resumen del artículo citado."
            : "Necesitas el contexto de la cita, la cita objetivo y tres fragmentos relevantes del artículo citado.";

    const available = uiModels.filter(
        model => model.available && model.configuration === mode
    );

    modelSelector.replaceChildren();

    for (const model of available) {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = nombresModelosEs[model.id] || model.name.replace(/[\u2013\u2014]/g, ":").replace("Configuration", "Configuración");
        modelSelector.appendChild(option);
    }

    const preferred = available.find(
        model => model.id === lastModelForConfiguration[mode]
    );

    const selected = preferred || available[0];

    if (selected) {
        modelSelector.value = selected.id;
        lastModelForConfiguration[mode] = selected.id;
        modelSelector.disabled = false;
        submitButton.disabled = false;
        modelSelector.dispatchEvent(
            new Event("change", {bubbles: true})
        );
    } else {
        modelSelector.disabled = true;
        submitButton.disabled = true;
        showError("No hay modelos disponibles para esta configuración.");
    }

    if (previousMode !== mode) {
        resultSection.classList.add("hidden");
        errorElement.classList.add("hidden");
    }
}

function loadConfigurationModels(models) {
    uiModels = models;
    setUiConfiguration(uiConfiguration);
}

configurationButtons.forEach(button => {
    button.addEventListener("click", () => {
        setUiConfiguration(button.dataset.configChoice);
    });
});


const modelConfigurations = new Map();
const configAFields = document.getElementById("config-a-fields");
const citedTitleInput = document.getElementById("cited-title");
const citedAbstractInput = document.getElementById("cited-abstract");

function currentMode() {
    return modelConfigurations.get(modelSelector.value) === "A"
        ? "A" : "B";
}

function updateEvidenceMode() {
    const isA = currentMode() === "A";

    configAFields.classList.toggle("hidden", !isA);

    const button = document.getElementById("retrieve-evidence-button");
    const status = document.getElementById("evidence-status");

    if (button) button.classList.toggle("hidden", isA);
    if (status) status.classList.toggle("hidden", isA);

    const paperHelp = document.querySelector(
        ".paper-lookup > .paper-help"
    );
    if (paperHelp) {
        paperHelp.textContent = isA
            ? "Busca el artículo para obtener su título y resumen."
            : "Busca y confirma el artículo para recuperar los 3 fragmentos más relevantes.";
    }

    const pdfHelp = document.querySelector(
        ".pdf-upload-panel > .paper-help"
    );
    if (pdfHelp) {
        pdfHelp.textContent = isA
            ? "Sube un PDF para extraer el título y resumen. Revísalos antes de clasificar."
            : "Sube un PDF para recuperar los 3 fragmentos más relevantes.";
    }

    if (!isA && selectedPaper?.arxiv_id && !button) {
        mountEvidenceRetriever();
    }

    syncPdfStateForMode();
}

modelSelector.addEventListener("change", updateEvidenceMode);

async function apiRequest(path, options = {}) {
    const response = await fetch(`${API}${path}`, options);
    const data = await response.json();

    if (!response.ok) {
        const detail = data.detail;
        const message = Array.isArray(detail)
            ? detail.map(item => item.msg).join("; ")
            : detail || "No se pudo completar la solicitud.";
        throw new Error(message);
    }
    return data;
}

async function initialize() {
    try {
        const [modelsData, labelsData] = await Promise.all([
            apiRequest("/models"),
            apiRequest("/labels"),
        ]);

        modelSelector.replaceChildren();

        for (const model of modelsData.models) {
            modelConfigurations.set(model.id, model.configuration);
            const option = document.createElement("option");
            option.value = model.id;
            option.textContent = model.available
                ? model.name
                : `${model.name} (No disponible)`;
            option.disabled = !model.available;
            modelSelector.appendChild(option);
        }

        const firstAvailable = modelsData.models.find(m => m.available);
        if (firstAvailable) {
            modelSelector.value = firstAvailable.id;
            updateEvidenceMode();
        } else {
            submitButton.disabled = true;
            showError("No hay modelos disponibles en este momento.");
        }

        loadConfigurationModels(modelsData.models);

        const taxonomy = document.getElementById("taxonomy-list");
        taxonomy.replaceChildren();

        for (const label of labelsData.labels) {
            const item = document.createElement("div");
            item.className = "taxonomy-item";

            const title = document.createElement("strong");
            title.textContent = `● ${label}`;

            item.appendChild(title);
            taxonomy.appendChild(item);
        }
    } catch (error) {
        submitButton.disabled = true;
        showError(error.message);
    }
}

function showError(message) {
    errorElement.textContent = message;
    errorElement.classList.remove("hidden");
}


const citationPattern =
    /\(([^()]*\b(?:19|20)\d{2}[a-z]?[^()]*)\)|\[(\d+(?:\s*[,;-]\s*\d+)*)\]|\b([A-Z][\w'-]+(?:\s+et\s+al\.?)?\s*\((?:19|20)\d{2}[a-z]?\))/;

let manuallySelected = false;

targetInput.addEventListener("input", () => {
    manuallySelected = true;
});

contextInput.addEventListener("input", () => {
    const text = contextInput.value;
    const current = targetInput.value.trim();

    if (manuallySelected && current && text.includes(current)) {
        return;
    }

    const match = text.match(citationPattern);
    targetInput.value = match
        ? (match[1] || match[2] || match[3]).trim()
        : "";

    manuallySelected = false;
});

contextInput.addEventListener("input", () => {
    document.getElementById("character-count").textContent =
        `${contextInput.value.length} / 5000`;
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorElement.classList.add("hidden");
    resultSection.classList.add("hidden");

    submitButton.disabled = true;
    submitButton.textContent = "Analizando cita...";
    let progressInterval = null;

    try {
        const citingContext = contextInput.value.trim();
        const targetCitation = targetInput.value.trim();

        const mode = currentMode();
        const selectedModelId = modelSelector.value;

        if (mode === "A" &&
            (!citedTitleInput.value.trim() ||
             !citedAbstractInput.value.trim())) {
            throw new Error(
                "Configuration A requires a cited paper title and abstract."
            );
        }

        if (mode === "B" && !retrievedEvidence?.cited_chunks?.length) {
            throw new Error(
                "Configuration B requires retrieved evidence chunks."
            );
        }

        // Step 1: Prepare target citation and mask secondary citations.
        const prepared = await apiRequest("/prepare-context", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                citing_context: citingContext,
                target_citation: targetCitation,
            }),
        });

        // Step 2: Run selected classifier.
        const predictionStartedAt = performance.now();

        if (selectedModelId) {
            const updateProgress = () => {
                const elapsed = Math.floor(
                    (performance.now() - predictionStartedAt) / 1000
                );
                const minutes = String(
                    Math.floor(elapsed / 60)
                ).padStart(2, "0");
                const seconds = String(
                    elapsed % 60
                ).padStart(2, "0");

                progressEl.textContent =
                    `${modelSelector.selectedOptions[0]?.textContent || selectedModelId}: ${minutes}:${seconds}. ` +
                    (selectedModelId.startsWith("granite-")
                        ? "En CPU puede tardar más de un minuto."
                        : "Analizando la cita...");
            };

            progressEl.classList.remove("hidden");
            updateProgress();
            progressInterval = setInterval(updateProgress, 1000);
        }

        const prediction = await apiRequest("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                model_id: selectedModelId,
                citing_context: prepared.citing_context,
                target_citation: targetCitation,
                cited_title: mode === "A"
                    ? citedTitleInput.value.trim()
                    : selectedPaper?.title ?? null,
                cited_abstract: mode === "A"
                    ? citedAbstractInput.value.trim()
                    : selectedPaper?.abstract ?? null,
                cited_arxiv_id: selectedPaper?.arxiv_id ?? null,
                cited_chunks: mode === "B"
                    ? retrievedEvidence?.cited_chunks ?? null
                    : null,
                cited_chunk_sections: mode === "B"
                    ? retrievedEvidence?.cited_chunk_sections ?? null
                    : null,
            }),
        });

        const requestElapsedMs =
            performance.now() - predictionStartedAt;

        renderPrediction(prediction, requestElapsedMs);
        // The metric exists on the API already; only update browser-observed time.
        if (prediction.run_id) {
            apiRequest("/metrics/client", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    run_id: prediction.run_id,
                    client_ms: requestElapsedMs,
                }),
            }).catch(() => {});
        }
    } catch (error) {
        showError(error.message);
    } finally {
        if (progressInterval !== null) {
            clearInterval(progressInterval);
        }
        progressEl.classList.add("hidden");
        progressEl.textContent = "";

        submitButton.disabled = false;
        submitButton.textContent = "✦ CLASIFICAR INTENCIÓN DE CITA";
    }
});

function renderPrediction(data, requestElapsedMs) {
    document.getElementById("prediction").textContent =
        data.predicted_label.toUpperCase();

    const confidence = data.confidence * 100;

    document.getElementById("confidence").textContent =
        `${confidence.toFixed(1)}%`;

    document.getElementById("confidence-bar").style.width =
        `${confidence}%`;

    const probabilities = document.getElementById("probabilities");
    probabilities.replaceChildren();

    for (const [label, probability] of Object.entries(data.probabilities)) {
        const percentage = probability * 100;

        const row = document.createElement("div");
        row.className = "probability-row";

        const name = document.createElement("span");
        name.className = "probability-label";
        name.textContent = label;

        const track = document.createElement("div");
        track.className = "probability-track";

        const fill = document.createElement("div");
        fill.className =
            `probability-fill fill-${label.toLowerCase().replace(/\s+/g, "-")}`;
        fill.style.width = `${percentage}%`;
        track.appendChild(fill);

        const value = document.createElement("span");
        value.className = "probability-value";
        value.textContent = `${percentage.toFixed(1)}%`;

        row.append(name, track, value);
        probabilities.appendChild(row);
    }

    let warning = document.getElementById("prediction-warning");
    if (!warning) {
        warning = document.createElement("p");
        warning.id = "prediction-warning";
        warning.className = "field-description";
        resultSection.appendChild(warning);
    }

    warning.textContent = data.is_demo
        ? "DEMO MODE : Synthetic prediction. Scores are not scientifically meaningful."
        : "";

    const modelLatencyMs = Number(data.latency_ms);

    if (Number.isFinite(modelLatencyMs) &&
        modelLatencyMs >= 0) {
        const serverLatency = Number(data.server_ms);
        latencyEl.textContent =
            `Modelo (reportado): ${(modelLatencyMs / 1000).toFixed(1)} s` +
            (Number.isFinite(serverLatency)
                ? ` | API: ${(serverLatency / 1000).toFixed(1)} s`
                : "") +
            ` | HTTP: ${(requestElapsedMs / 1000).toFixed(1)} s`;
        latencyEl.classList.remove("hidden");
    } else {
        latencyEl.classList.add("hidden");
    }

    resultSection.classList.remove("hidden");
    resultSection.scrollIntoView({behavior: "smooth", block: "nearest"});
}

initialize();


// Cited paper search and confirmation
let selectedPaper = null;
let retrievedEvidence = null;

const paperQuery = document.getElementById("paper-query");
const paperButton = document.getElementById("paper-search-button");
const paperResult = document.getElementById("paper-result");

paperQuery.addEventListener("input", () => {
    selectedPaper = null;
    paperResult.replaceChildren();
    paperResult.classList.add("hidden");
});

paperButton.addEventListener("click", async () => {
    resetEvidence();
    const query = paperQuery.value.trim();
    if (!query) return;

    paperButton.disabled = true;
    paperButton.textContent = "BUSCANDO...";
    paperResult.classList.remove("hidden");
    paperResult.textContent = "Buscando el artículo citado...";
    selectedPaper = null;

    try {
        const paper = await apiRequest(
            `/papers/lookup?query=${encodeURIComponent(query)}`
        );

        const title = document.createElement("h3");
        title.textContent = paper.title;

        const authors = document.createElement("p");
        authors.textContent = paper.authors?.join(", ") || "Autores no disponibles";

        const details = document.createElement("p");
        details.textContent =
            `Fuente: ${paper.source} | arXiv: ${paper.arxiv_id || "N/A"}`;

        const abstract = document.createElement("p");
        abstract.textContent = paper.abstract || "Resumen no disponible.";

        const confirm = document.createElement("button");
        confirm.type = "button";
        confirm.textContent = "CONFIRMAR ARTÍCULO";

        confirm.addEventListener("click", () => {
            selectedPaper = paper;
            citedTitleInput.value = paper.title || "";
            citedAbstractInput.value = paper.abstract || "";

            if (currentMode() === "B") {
                mountEvidenceRetriever();
            } else {
                resetEvidence();
                updateEvidenceMode();
            }
            confirm.textContent = "✓ ARTÍCULO CONFIRMADO";
            confirm.disabled = true;
        });

        paperResult.replaceChildren(
            title, authors, details, abstract, confirm
        );
    } catch (error) {
        paperResult.textContent = `Error durante la búsqueda: ${error.message}`;
    } finally {
        paperButton.disabled = false;
        paperButton.textContent = "BUSCAR";
    }
});


// Real SciBERT evidence retrieval
function resetEvidence() {
    retrievedEvidence = null;

    document.getElementById("pdf-upload-result")?.replaceChildren();

    const status = document.getElementById("evidence-status");
    if (status) status.replaceChildren();

    const button = document.getElementById("retrieve-evidence-button");
    if (button) {
        button.disabled = false;
        button.textContent = "RECUPERAR LOS 3 FRAGMENTOS";
    }
}

contextInput.addEventListener("input", resetEvidence);
targetInput.addEventListener("input", resetEvidence);
paperQuery.addEventListener("input", resetEvidence);

function mountEvidenceRetriever() {
    const existingButton =
        document.getElementById("retrieve-evidence-button");
    const existingStatus =
        document.getElementById("evidence-status");

    // Reuse the existing controls and evidence when switching A/B.
    if (
        existingButton &&
        existingStatus &&
        paperResult.contains(existingButton)
    ) {
        const isA = currentMode() === "A";

        existingButton.classList.toggle("hidden", isA);
        existingStatus.classList.toggle("hidden", isA);
        return;
    }

    // Only replace stale controls. Do not discard retrieved evidence.
    existingButton?.remove();
    existingStatus?.remove();

    const button = document.createElement("button");
    button.id = "retrieve-evidence-button";
    button.type = "button";
    button.textContent = "RECUPERAR LOS 3 FRAGMENTOS";
    button.disabled = !selectedPaper?.arxiv_id;

    const status = document.createElement("div");
    status.id = "evidence-status";

    if (!selectedPaper?.arxiv_id) {
        status.textContent = "La recuperación del texto completo requiere un identificador arXiv.";
    }

    button.addEventListener("click", async () => {
        const context = contextInput.value.trim();
        const target = targetInput.value.trim();
        const arxivId = selectedPaper?.arxiv_id;

        if (context.length < 20) {
            status.textContent = "Primero escribe el contexto de la cita.";
            return;
        }

        resetEvidence();
        button.disabled = true;
        button.textContent = "RECUPERANDO...";
        status.textContent =
            "Extrayendo el artículo y ordenando los fragmentos con SciBERT...";

        try {
            const data = await apiRequest("/evidence", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    citing_context: context,
                    target_citation: target || null,
                    cited_arxiv_id: arxivId,
                    cited_title: selectedPaper?.title ?? null,
                    cited_abstract: selectedPaper?.abstract ?? null,
                }),
            });

            // Never reuse evidence if the citation changed during retrieval.
            if (
                contextInput.value.trim() !== context ||
                targetInput.value.trim() !== target ||
                selectedPaper?.arxiv_id !== arxivId
            ) {
                resetEvidence();
                return;
            }

            if (!data.cited_chunks?.length) {
                throw new Error("No se encontraron fragmentos de evidencia.");
            }

            retrievedEvidence = data;

            const container = document.createElement("div");
            container.className = "evidence-results";

            const heading = document.createElement("h3");
            heading.textContent = "EVIDENCIA RECUPERADA: 3 FRAGMENTOS";
            container.appendChild(heading);

            data.cited_chunks.forEach((chunk, index) => {
                const item = document.createElement("article");
                item.className = "evidence-chunk";

                const title = document.createElement("h4");
                title.textContent =
                    `FRAGMENTO ${index + 1}: ` +
                    (data.cited_chunk_sections?.[index] || "Sección no identificada");

                const paragraph = document.createElement("p");
                paragraph.textContent = chunk;

                item.append(title, paragraph);
                container.appendChild(item);
            });

            status.replaceChildren(container);
            button.textContent =
                `✓ ${data.cited_chunks.length} FRAGMENTOS LISTOS`;
        } catch (error) {
            status.textContent = `Error al recuperar la evidencia: ${error.message}`;
            button.disabled = false;
            button.textContent = "REINTENTAR RECUPERACIÓN";
        }
    });

    paperResult.append(button, status);
}


// Uploaded PDF evidence retrieval
const pdfInput = document.getElementById("pdf-file");
const pdfUploadButton = document.getElementById("pdf-upload-button");
const pdfUploadResult = document.getElementById("pdf-upload-result");
const pdfAcknowledgement =
    document.getElementById("pdf-match-confirmation");

// Cache kept only in the current browser session.
let pdfCache = null;
let pdfProcessing = false;

function pdfCacheValid() {
    return Boolean(
        pdfCache &&
        pdfInput.files?.[0] === pdfCache.file &&
        contextInput.value.trim() === pdfCache.context &&
        targetInput.value.trim() === pdfCache.target
    );
}

function renderPdfEvidence(cache) {
    const container = document.createElement("div");
    container.className = "evidence-results";

    const heading = document.createElement("h3");
    heading.textContent =
        `PDF PROCESADO: ${cache.title || cache.file.name}`;
    container.appendChild(heading);

    cache.chunks.forEach((chunk, index) => {
        const item = document.createElement("article");
        item.className = "evidence-chunk";

        const title = document.createElement("h4");
        title.textContent =
            `FRAGMENTO ${index + 1}: ` +
            (cache.sections?.[index] || "Sección no identificada");

        const paragraph = document.createElement("p");
        paragraph.textContent = chunk;

        item.append(title, paragraph);
        container.appendChild(item);
    });

    pdfUploadResult.replaceChildren(container);
}

function syncPdfStateForMode() {
    if (pdfProcessing || !pdfCacheValid()) return;

    const mode = currentMode();

    selectedPaper = {
        title: pdfCache.title,
        abstract: pdfCache.abstract,
        arxiv_id: null,
    };

    if (mode === "A") {
        // Preserve any manual edits made after PDF extraction.
        if (!citedTitleInput.value.trim()) {
            citedTitleInput.value = pdfCache.title || "";
        }
        if (!citedAbstractInput.value.trim()) {
            citedAbstractInput.value = pdfCache.abstract || "";
        }

        pdfUploadResult.textContent =
            "Título y resumen disponibles. Revisa los campos antes de clasificar.";
        pdfUploadButton.textContent = "✓ DATOS DISPONIBLES";
    } else if (pdfCache.chunks?.length) {
        retrievedEvidence = {
            cited_title: pdfCache.title,
            cited_abstract: pdfCache.abstract,
            cited_chunks: pdfCache.chunks,
            cited_chunk_sections: pdfCache.sections,
        };

        renderPdfEvidence(pdfCache);
        pdfUploadButton.textContent = "✓ EVIDENCIA DISPONIBLE";
    } else {
        retrievedEvidence = null;

        pdfUploadResult.textContent =
            "El PDF ya está identificado. Para usar la configuración B, " +
            "recupera los tres fragmentos relevantes del mismo archivo. " +
            "No necesitas seleccionarlo nuevamente.";

        pdfUploadButton.textContent = "RECUPERAR FRAGMENTOS";
    }

    pdfUploadButton.disabled = !pdfAcknowledgement.checked;
}

pdfUploadButton.disabled = true;

pdfAcknowledgement.addEventListener("change", () => {
    pdfUploadButton.disabled =
        pdfProcessing || !pdfAcknowledgement.checked;
});

for (const field of [contextInput, targetInput]) {
    field.addEventListener("input", () => {
        pdfCache = null;
        pdfAcknowledgement.checked = false;
        pdfUploadButton.disabled = true;
    });
}

// Searching for another article invalidates the PDF cache.
paperQuery.addEventListener("input", () => {
    pdfCache = null;
});

pdfInput.addEventListener("change", () => {
    pdfCache = null;
    pdfAcknowledgement.checked = false;
    pdfUploadButton.disabled = true;

    resetEvidence();
    selectedPaper = null;

    paperResult.replaceChildren();
    paperResult.classList.add("hidden");
    paperQuery.value = "";
});

pdfUploadButton.addEventListener("click", async () => {
    if (pdfProcessing || !pdfAcknowledgement.checked) return;

    const file = pdfInput.files?.[0];
    const context = contextInput.value.trim();
    const target = targetInput.value.trim();
    const mode = currentMode();

    if (!file) {
        pdfUploadResult.textContent = "Primero selecciona un PDF.";
        return;
    }

    if (file.size > 20 * 1024 * 1024) {
        pdfUploadResult.textContent =
            "El tamaño máximo del PDF es 20 MB.";
        return;
    }

    if (context.length < 20) {
        pdfUploadResult.textContent =
            "Primero escribe el contexto de la cita.";
        return;
    }

    // Reuse existing PDF information without another HTTP request.
    if (pdfCacheValid()) {
        if (mode === "A" || pdfCache.chunks?.length) {
            syncPdfStateForMode();
            return;
        }
    }

    const previousCache = pdfCacheValid() ? pdfCache : null;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("citing_context", context);

    if (target) {
        formData.append("target_citation", target);
    }

    pdfProcessing = true;
    pdfUploadButton.disabled = true;
    pdfUploadButton.textContent = "PROCESANDO...";

    pdfUploadResult.textContent = mode === "A"
        ? "Extrayendo el título y resumen con Docling. " +
          "Esto puede tardar unos minutos. Mantén esta página abierta."
        : "Procesando el PDF y recuperando evidencia con SciBERT. " +
          "Esto puede tardar unos minutos. Mantén esta página abierta.";

    try {
        const data = await apiRequest(
            mode === "A" ? "/metadata/pdf" : "/evidence/pdf",
            {
                method: "POST",
                body: formData,
            }
        );

        // A mode switch alone does not invalidate the response.
        // A different file or citation context does.
        if (
            pdfInput.files?.[0] !== file ||
            contextInput.value.trim() !== context ||
            targetInput.value.trim() !== target
        ) {
            pdfUploadResult.textContent =
                "El archivo o el contexto cambió durante el procesamiento. " +
                "Inténtalo nuevamente.";
            return;
        }

        pdfCache = {
            file,
            context,
            target,
            title: data.cited_title || previousCache?.title || "",
            abstract: data.cited_abstract || previousCache?.abstract || "",
            chunks: mode === "B"
                ? (data.cited_chunks || [])
                : (previousCache?.chunks || []),
            sections: mode === "B"
                ? (data.cited_chunk_sections || [])
                : (previousCache?.sections || []),
        };

        // Populate A metadata even when the PDF was processed in B.
        citedTitleInput.value = pdfCache.title;
        citedAbstractInput.value = pdfCache.abstract;

        selectedPaper = {
            title: pdfCache.title,
            abstract: pdfCache.abstract,
            arxiv_id: null,
        };

        if (pdfCache.chunks.length) {
            retrievedEvidence = {
                cited_title: pdfCache.title,
                cited_abstract: pdfCache.abstract,
                cited_chunks: pdfCache.chunks,
                cited_chunk_sections: pdfCache.sections,
            };
        }

        paperQuery.value = "";
        paperResult.replaceChildren();
        paperResult.classList.add("hidden");

    } catch (error) {
        pdfUploadResult.textContent =
            `Error al procesar el PDF: ${error.message}`;
        pdfUploadButton.textContent = "REINTENTAR PDF";
    } finally {
        pdfProcessing = false;

        if (pdfCacheValid()) {
            syncPdfStateForMode();
        } else {
            pdfUploadButton.disabled = !pdfAcknowledgement.checked;
        }
    }
});
