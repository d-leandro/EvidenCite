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
