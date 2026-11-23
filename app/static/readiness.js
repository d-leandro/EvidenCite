(() => {
    "use strict";

    const form = document.getElementById("prediction-form");
    const context = document.getElementById("context");
    const target = document.getElementById("target-citation");
    const title = document.getElementById("cited-title");
    const abstract = document.getElementById("cited-abstract");
    const model = document.getElementById("model-selector");
    const submit = document.getElementById("submit-button");
    const footer = form?.querySelector(".form-footer");

    if (!form || !context || !target || !title ||
        !abstract || !model || !submit || !footer) {
        console.error("EvidenCite: faltan elementos de validación.");
        return;
    }

    const panel = document.createElement("section");
    panel.className = "readiness-panel";
    panel.setAttribute("aria-label", "Requisitos para clasificar");

    panel.innerHTML = `
        <div class="readiness-header">
            <strong>Preparación del análisis</strong>
            <span id="readiness-count">0 de 3 requisitos</span>
        </div>
        <div class="readiness-progress" aria-hidden="true">
            <span></span><span></span><span></span>
        </div>
        <div class="readiness-items">
            <div class="readiness-item" data-requirement="context">
                <span class="readiness-icon"></span>
                <span class="readiness-name">Contexto y cita objetivo</span>
                <span class="readiness-state">Pendiente</span>
            </div>
            <div class="readiness-item" data-requirement="paper">
                <span class="readiness-icon"></span>
                <span class="readiness-name">Información del artículo</span>
                <span class="readiness-state">Pendiente</span>
            </div>
            <div class="readiness-item" data-requirement="model">
                <span class="readiness-icon"></span>
                <span class="readiness-name">Modelo disponible</span>
                <span class="readiness-state">Pendiente</span>
            </div>
        </div>
        <p id="readiness-help" class="readiness-help"></p>
    `;

    footer.before(panel);

    function markRequired(element, fallbackLabel) {
        if (!element) return;

        const label = document.querySelector(
            `label[for="${element.id}"]`
        );

        if (label && !label.querySelector(".required-mark")) {
            const mark = document.createElement("span");
            mark.className = "required-mark";
            mark.textContent = " *";
            mark.setAttribute("aria-hidden", "true");
            label.appendChild(mark);
        }

        element.setAttribute("aria-required", "true");

        if (!label && fallbackLabel) {
            element.setAttribute("aria-label", fallbackLabel);
        }
    }

    markRequired(context, "Contexto de la cita, obligatorio");
    markRequired(target, "Cita objetivo, obligatoria");
    markRequired(title, "Título del artículo, obligatorio");
    markRequired(abstract, "Resumen del artículo, obligatorio");

    const rows = Object.fromEntries(
        [...panel.querySelectorAll("[data-requirement]")]
            .map(row => [row.dataset.requirement, row])
    );

    const progress = [
        ...panel.querySelectorAll(".readiness-progress span")
    ];

    const count = document.getElementById("readiness-count");
    const help = document.getElementById("readiness-help");

    function getMode() {
        return document.body.dataset.configuration === "B"
            ? "B" : "A";
    }

    function getEvidence() {
        return typeof retrievedEvidence !== "undefined"
            ? retrievedEvidence
            : null;
    }

    function validate() {
        const mode = getMode();

        const contextReady =
            context.value.trim().length >= 20 &&
            target.value.trim().length > 0;

        const evidence = getEvidence();

        const paperReady = mode === "A"
            ? title.value.trim().length > 0 &&
              abstract.value.trim().length > 0
            : Array.isArray(evidence?.cited_chunks) &&
              evidence.cited_chunks.length >= 3 &&
              evidence.cited_chunks.slice(0, 3).every(
                  chunk => typeof chunk === "string" &&
                           chunk.trim().length > 0
              );

        const selected = model.selectedOptions[0];

        const modelReady =
            Boolean(model.value && selected && !selected.disabled) &&
            !model.disabled &&
            model.value.toLowerCase().endsWith(
                `-${mode.toLowerCase()}`
            );

        const requirements = {
            context: contextReady,
            paper: paperReady,
            model: modelReady,
        };

        const completed = Object.values(requirements)
            .filter(Boolean).length;

        for (const [key, ready] of Object.entries(requirements)) {
            const row = rows[key];

            row.classList.toggle("complete", ready);

            const state = row.querySelector(".readiness-state");
            const next = ready ? "Completo" : "Pendiente";

            if (state.textContent !== next) {
                state.textContent = next;
            }
        }

        progress.forEach((segment, index) => {
            segment.classList.toggle("complete", index < completed);
        });

        const progressText = `${completed} de 3 requisitos`;
        if (count.textContent !== progressText) {
            count.textContent = progressText;
        }

        rows.paper.querySelector(".readiness-name").textContent =
            mode === "A"
                ? "Título y resumen del artículo"
                : "Tres fragmentos de evidencia";

        const messages = [];

        if (!contextReady) {
            messages.push("Completa el contexto y la cita objetivo.");
        }

        if (!paperReady) {
            messages.push(mode === "A"
                ? "Completa el título y el resumen del artículo."
                : "Recupera los tres fragmentos del artículo.");
        }

        if (!modelReady) {
            messages.push("Selecciona un modelo disponible.");
        }

        const nextHelp = completed === 3
            ? "Todo listo. Puedes clasificar la intención de cita."
            : messages.join(" ");

        if (help.textContent !== nextHelp) {
            help.textContent = nextHelp;
        }

        help.classList.toggle("ready", completed === 3);

        const analyzing =
            /analizando|analyzing/i.test(submit.textContent);

        const shouldDisable = completed !== 3 || analyzing;

        if (submit.disabled !== shouldDisable) {
            submit.disabled = shouldDisable;
        }
    }

    for (const field of [context, target, title, abstract]) {
        field.addEventListener("input", validate);
        field.addEventListener("change", validate);
    }

    model.addEventListener("change", validate);

    // Detect changes made by asynchronous search, PDF and retrieval.
    const observer = new MutationObserver(validate);

    for (const id of [
        "paper-result",
        "pdf-upload-result",
        "configuration-switch",
        "model-selector",
        "submit-button"
    ]) {
        const element = document.getElementById(id);

        if (element) {
            observer.observe(element, {
                childList: true,
                subtree: true,
                attributes: element === submit
            });
        }
    }

    validate();
})();
