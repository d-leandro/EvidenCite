(() => {
    "use strict";

    const lookup = document.querySelector(".paper-lookup");
    const title = document.getElementById("cited-title");
    const abstract = document.getElementById("cited-abstract");
    const paperResult = document.getElementById("paper-result");
    const pdfResult = document.getElementById("pdf-upload-result");
    const pdfButton = document.getElementById("pdf-upload-button");

    if (!lookup || !title || !abstract ||
        !paperResult || !pdfResult || !pdfButton) {
        console.error("EvidenCite: faltan elementos de entrada.");
        return;
    }

    const searchLabel = lookup.querySelector(
        'label[for="paper-query"]'
    );
    const searchRow = lookup.querySelector(".paper-search-row");
    const searchHelp = lookup.querySelector(
        ":scope > .paper-help"
    );
    const pdfPanel = lookup.querySelector(".pdf-upload-panel");

    if (!searchLabel || !searchRow || !pdfPanel) {
        console.error("EvidenCite: estructura de búsqueda incompleta.");
        return;
    }

    const selector = document.createElement("div");
    selector.className = "source-method-selector";

    selector.innerHTML = `
        <div class="source-method-heading">
            <strong>Información del artículo citado</strong>
            <p>
                Elige cómo proporcionar los datos. La información
                ya obtenida se conserva al cambiar de opción.
            </p>
        </div>

        <div class="source-method-options" role="group"
             aria-label="Método de entrada del artículo">
            <button type="button" data-source-method="manual"
                    aria-controls="source-panel-manual">
                <span class="source-method-symbol">✎</span>
                <span class="source-method-name">Escribir manualmente</span>
                <span class="source-method-ready" hidden>✓ Listo</span>
                <span class="source-method-chevron">⌄</span>
            </button>

            <button type="button" data-source-method="search"
                    aria-controls="source-panel-search">
                <span class="source-method-symbol">⌕</span>
                <span class="source-method-name">Buscar artículo</span>
                <span class="source-method-ready" hidden>✓ Listo</span>
                <span class="source-method-chevron">⌄</span>
            </button>

            <button type="button" data-source-method="pdf"
                    aria-controls="source-panel-pdf">
                <span class="source-method-symbol">▤</span>
                <span class="source-method-name">Subir PDF</span>
                <span class="source-method-ready" hidden>✓ Listo</span>
                <span class="source-method-chevron">⌄</span>
            </button>
        </div>
    `;

    const searchPanel = document.createElement("div");
    searchPanel.id = "source-panel-search";
    searchPanel.className = "source-method-panel";
    searchPanel.append(
        searchLabel,
        searchRow
    );
    if (searchHelp) searchPanel.append(searchHelp);
    searchPanel.append(paperResult);

    const pdfSourcePanel = document.createElement("div");
    pdfSourcePanel.id = "source-panel-pdf";
    pdfSourcePanel.className = "source-method-panel";
    pdfSourcePanel.append(pdfPanel);

    const manualPanel = document.createElement("div");
    manualPanel.id = "source-panel-manual";
    manualPanel.className = "source-method-panel";

    // Move the original fields without recreating them.
    // Their values and event listeners remain intact.
    const configAFields = document.getElementById("config-a-fields");

    if (!configAFields) {
        console.error("No se encontraron los campos de configuración A.");
        return;
    }

    const helper = configAFields.querySelector(".paper-help");

    if (helper) {
        helper.textContent =
            "Escribe o edita el título y el resumen del artículo. " +
            "Si antes buscaste o procesaste un PDF, " +
            "los datos obtenidos se conservan.";
    }

    manualPanel.append(configAFields);

    lookup.prepend(selector);
    lookup.append(searchPanel, pdfSourcePanel, manualPanel);

    const buttons = [
        ...selector.querySelectorAll("[data-source-method]")
    ];

    const panels = {
        manual: manualPanel,
        search: searchPanel,
        pdf: pdfSourcePanel,
    };

    let activeMethod = "manual";
    let previousMode = "A";

    const history = {A: "manual", B: null};

    function mode() {
        return document.body.dataset.configuration === "B"
            ? "B" : "A";
    }

    function selectMethod(method) {
        if (!(method in panels)) return;
        if (mode() === "B" && method === "manual") {
            method = "search";
        }

        activeMethod = method;
        history[mode()] = method;

        for (const button of buttons) {
            const selected =
                button.dataset.sourceMethod === method;

            button.classList.toggle("active", selected);
            button.setAttribute("aria-expanded", String(selected));
        }

        for (const [name, panel] of Object.entries(panels)) {
            panel.classList.toggle("hidden", name !== method);
        }

        updateStatus();
    }

    function availableEvidence() {
        return typeof retrievedEvidence !== "undefined"
            ? retrievedEvidence
            : null;
    }

    function availablePaper() {
        return typeof selectedPaper !== "undefined"
            ? selectedPaper
            : null;
    }

    function updateStatus() {
        const currentMode = mode();
        const paper = availablePaper();
        const evidence = availableEvidence();

        const hasChunks =
            Array.isArray(evidence?.cited_chunks) &&
            evidence.cited_chunks.length >= 3;

        const hasMetadata =
            Boolean(title.value.trim() && abstract.value.trim());

        const confirmedSearch = [
            ...paperResult.querySelectorAll("button")
        ].some(button =>
            button.disabled &&
            /ARTÍCULO CONFIRMADO|PAPER SELECTED/i
                .test(button.textContent)
        );

        const cachedPdf =
            typeof pdfCacheValid === "function" &&
            pdfCacheValid() &&
            typeof pdfCache !== "undefined"
                ? pdfCache
                : null;

        const ready = {
            manual: currentMode === "A" && hasMetadata,
            search: currentMode === "A"
                ? confirmedSearch && hasMetadata
                : Boolean(paper?.arxiv_id && hasChunks),
            pdf: Boolean(cachedPdf) && (
                currentMode === "A"
                    ? Boolean(cachedPdf.title && cachedPdf.abstract)
                    : cachedPdf.chunks?.length >= 3
            ),
        };

        for (const button of buttons) {
            const method = button.dataset.sourceMethod;
            const indicator = button.querySelector(
                ".source-method-ready"
            );

            button.classList.toggle("complete", !!ready[method]);
            indicator.hidden = !ready[method];
        }
    }

    function syncMode() {
        const current = mode();

        for (const button of buttons) {
            button.classList.toggle(
                "hidden",
                current === "B" &&
                button.dataset.sourceMethod === "manual"
            );
        }

        if (current !== previousMode) {
            const nextMethod = history[current] ||
                (activeMethod === "manual" ? "search" : activeMethod);

            previousMode = current;
            selectMethod(nextMethod);
        } else {
            updateStatus();
        }
    }

    for (const button of buttons) {
        button.addEventListener("click", () => {
            selectMethod(button.dataset.sourceMethod);
        });
    }

    for (const field of [title, abstract]) {
        field.addEventListener("input", updateStatus);
        field.addEventListener("change", updateStatus);
    }

    const observer = new MutationObserver(syncMode);

    observer.observe(document.body, {
        attributes: true,
        attributeFilter: ["data-configuration"],
    });

    for (const element of [paperResult, pdfResult, pdfButton]) {
        observer.observe(element, {
            childList: true,
            subtree: true,
            characterData: true,
            attributes: true,
            attributeFilter: ["disabled"],
        });
    }

    syncMode();
    selectMethod("manual");
})();
