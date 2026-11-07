const form = document.getElementById("prediction-form");
const contextInput = document.getElementById("context");
const targetInput = document.getElementById("target-citation");
const modelSelector = document.getElementById("model-selector");
const submitButton = document.getElementById("submit-button");
const resultSection = document.getElementById("result-section");
const errorElement = document.getElementById("error-message");

const API = "/api/v1";

async function apiRequest(path, options = {}) {
    const response = await fetch(`${API}${path}`, options);
    const data = await response.json();

    if (!response.ok) {
        const detail = data.detail;
        const message = Array.isArray(detail)
            ? detail.map(item => item.msg).join("; ")
            : detail || "Request failed";
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
            const option = document.createElement("option");
            option.value = model.id;
            option.textContent = model.available
                ? model.name
                : `${model.name} (Unavailable)`;
            option.disabled = !model.available;
            modelSelector.appendChild(option);
        }

        const firstAvailable = modelsData.models.find(m => m.available);
        if (firstAvailable) {
            modelSelector.value = firstAvailable.id;
        } else {
            submitButton.disabled = true;
            showError("No inference models are currently available.");
        }

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
    submitButton.textContent = "ANALYZING...";

    try {
        const citingContext = contextInput.value.trim();
        const targetCitation = targetInput.value.trim();

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
        const prediction = await apiRequest("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                model_id: modelSelector.value,
                citing_context: prepared.citing_context,
                target_citation: targetCitation,
                cited_title: selectedPaper?.title ?? null,
                cited_abstract: selectedPaper?.abstract ?? null,
                cited_arxiv_id: selectedPaper?.arxiv_id ?? null,
                cited_chunks: retrievedEvidence?.cited_chunks ?? null,
                cited_chunk_sections:
                    retrievedEvidence?.cited_chunk_sections ?? null,
            }),
        });

        renderPrediction(prediction);
    } catch (error) {
        showError(error.message);
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = "✦ CLASSIFY CITATION";
    }
});

function renderPrediction(data) {
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
        ? "DEMO MODE — Synthetic prediction. Scores are not scientifically meaningful."
        : "";

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
    paperButton.textContent = "SEARCHING...";
    paperResult.classList.remove("hidden");
    paperResult.textContent = "Searching for cited paper...";
    selectedPaper = null;

    try {
        const paper = await apiRequest(
            `/papers/lookup?query=${encodeURIComponent(query)}`
        );

        const title = document.createElement("h3");
        title.textContent = paper.title;

        const authors = document.createElement("p");
        authors.textContent = paper.authors?.join(", ") || "Unknown authors";

        const details = document.createElement("p");
        details.textContent =
            `Source: ${paper.source} | arXiv: ${paper.arxiv_id || "N/A"}`;

        const abstract = document.createElement("p");
        abstract.textContent = paper.abstract || "Abstract unavailable.";

        const confirm = document.createElement("button");
        confirm.type = "button";
        confirm.textContent = "CONFIRM THIS PAPER";

        confirm.addEventListener("click", () => {
            selectedPaper = paper;
            mountEvidenceRetriever();
            confirm.textContent = "✓ PAPER SELECTED";
            confirm.disabled = true;
        });

        paperResult.replaceChildren(
            title, authors, details, abstract, confirm
        );
    } catch (error) {
        paperResult.textContent = `Search failed: ${error.message}`;
    } finally {
        paperButton.disabled = false;
        paperButton.textContent = "SEARCH";
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
        button.textContent = "RETRIEVE TOP 3 EVIDENCE";
    }
}

contextInput.addEventListener("input", resetEvidence);
targetInput.addEventListener("input", resetEvidence);
paperQuery.addEventListener("input", resetEvidence);

function mountEvidenceRetriever() {
    resetEvidence();

    const button = document.createElement("button");
    button.id = "retrieve-evidence-button";
    button.type = "button";
    button.textContent = "RETRIEVE TOP 3 EVIDENCE";
    button.disabled = !selectedPaper?.arxiv_id;

    const status = document.createElement("div");
    status.id = "evidence-status";

    if (!selectedPaper?.arxiv_id) {
        status.textContent = "Full-text retrieval requires an arXiv ID.";
    }

    button.addEventListener("click", async () => {
        const context = contextInput.value.trim();
        const target = targetInput.value.trim();
        const arxivId = selectedPaper?.arxiv_id;

        if (context.length < 20) {
            status.textContent = "Enter a citation context first.";
            return;
        }

        resetEvidence();
        button.disabled = true;
        button.textContent = "RETRIEVING...";
        status.textContent =
            "Extracting the article and ranking chunks with SciBERT...";

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
                throw new Error("No evidence chunks were returned.");
            }

            retrievedEvidence = data;

            const container = document.createElement("div");
            container.className = "evidence-results";

            const heading = document.createElement("h3");
            heading.textContent = "RETRIEVED EVIDENCE — TOP 3";
            container.appendChild(heading);

            data.cited_chunks.forEach((chunk, index) => {
                const item = document.createElement("article");
                item.className = "evidence-chunk";

                const title = document.createElement("h4");
                title.textContent =
                    `TOP ${index + 1} — ` +
                    (data.cited_chunk_sections?.[index] || "Unknown section");

                const paragraph = document.createElement("p");
                paragraph.textContent = chunk;

                item.append(title, paragraph);
                container.appendChild(item);
            });

            status.replaceChildren(container);
            button.textContent =
                `✓ ${data.cited_chunks.length} CHUNKS READY`;
        } catch (error) {
            status.textContent = `Retrieval failed: ${error.message}`;
            button.disabled = false;
            button.textContent = "RETRY RETRIEVAL";
        }
    });

    paperResult.append(button, status);
}


// Uploaded PDF evidence retrieval
const pdfInput = document.getElementById("pdf-file");
const pdfUploadButton = document.getElementById("pdf-upload-button");
const pdfUploadResult = document.getElementById("pdf-upload-result");

pdfInput.addEventListener("change", () => {
    resetEvidence();
    selectedPaper = null;
    paperResult.replaceChildren();
    paperResult.classList.add("hidden");
    paperQuery.value = "";
});

pdfUploadButton.addEventListener("click", async () => {
    const file = pdfInput.files?.[0];
    const context = contextInput.value.trim();
    const target = targetInput.value.trim();

    if (!file) {
        pdfUploadResult.textContent = "Select a PDF first.";
        return;
    }

    if (file.size > 20 * 1024 * 1024) {
        pdfUploadResult.textContent = "Maximum PDF size is 20 MB.";
        return;
    }

    if (context.length < 20) {
        pdfUploadResult.textContent = "Enter a citation context first.";
        return;
    }

    resetEvidence();
    selectedPaper = null;
    paperResult.replaceChildren();
    paperResult.classList.add("hidden");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("citing_context", context);

    if (target) {
        formData.append("target_citation", target);
    }

    pdfUploadButton.disabled = true;
    pdfUploadButton.textContent = "PROCESSING...";
    pdfUploadResult.textContent =
        "Extracting PDF with Docling and retrieving evidence with SciBERT...";

    try {
        const data = await apiRequest("/evidence/pdf", {
            method: "POST",
            body: formData,
        });

        if (
            pdfInput.files?.[0] !== file ||
            contextInput.value.trim() !== context ||
            targetInput.value.trim() !== target
        ) {
            pdfUploadResult.textContent =
                "Inputs changed during processing. Please retry.";
            return;
        }

        retrievedEvidence = data;
        selectedPaper = {
            title: data.cited_title,
            abstract: data.cited_abstract,
            arxiv_id: null,
        };

        paperQuery.value = "";

        const container = document.createElement("div");
        container.className = "evidence-results";

        const heading = document.createElement("h3");
        heading.textContent =
            `PDF PROCESSED — ${data.cited_title || file.name}`;
        container.appendChild(heading);

        data.cited_chunks.forEach((chunk, index) => {
            const item = document.createElement("article");
            item.className = "evidence-chunk";

            const title = document.createElement("h4");
            title.textContent =
                `TOP ${index + 1} — ` +
                (data.cited_chunk_sections?.[index] || "Unknown section");

            const paragraph = document.createElement("p");
            paragraph.textContent = chunk;

            item.append(title, paragraph);
            container.appendChild(item);
        });

        pdfUploadResult.replaceChildren(container);
        pdfUploadButton.textContent = "✓ PDF PROCESSED";

    } catch (error) {
        pdfUploadResult.textContent =
            `PDF processing failed: ${error.message}`;
        pdfUploadButton.textContent = "RETRY PDF";
    } finally {
        pdfUploadButton.disabled = false;
    }
});
