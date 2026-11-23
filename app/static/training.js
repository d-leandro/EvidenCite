// EvidenCite: referencia científica fija = test-revisado.
// Snapshot trazable de MLflow; esta pantalla no ejecuta entrenamiento ni consulta MLflow.
(() => {
    const tab = document.getElementById('training-tab');
    if (!tab) return;
    tab.replaceChildren();
    const panel = document.createElement('div');
    panel.className = 'metrics-panel panel';
    panel.innerHTML = `
        <h2 class="metrics-heading">Entrenamiento y evaluación · test-revisado</h2>
        <p class="metrics-intro"><strong>Conjunto de referencia: test-revisado.</strong> Todas las métricas, clasificaciones y gráficas principales usan exclusivamente este test y sus nueve clases. No se comparan con labeled-v2.</p>
        <p id="training-source" class="training-note">Cargando resultados de MLflow…</p>

        <div class="training-toolbar">
            <label for="training-family">Familia de modelo</label>
            <select id="training-family"><option value="all">Todas</option></select>
            <label for="training-configuration">Configuración</label>
            <select id="training-configuration"><option value="all">A y B</option><option value="A">A</option><option value="B">B</option></select>
        </div>
        <div class="training-overview">
            <div class="training-stat"><small>Runs principales visibles</small><strong id="training-run-count">—</strong></div>
            <div class="training-stat"><small>Mejor macro-F1 · test-revisado</small><strong id="training-best-f1">—</strong></div>
            <div class="training-stat"><small>Entrenamiento efectivo acumulado</small><strong id="training-total-time">—</strong></div>
        </div>
        <h3 class="metrics-subheading">Macro-F1 en test-revisado · seis modelos principales</h3>
        <div id="training-f1-chart" class="training-chart" aria-label="Macro-F1 en test-revisado"></div>
        <h3 class="metrics-subheading">Tiempo efectivo de entrenamiento (métrica A7)</h3>
        <div id="training-wall-chart" class="training-chart" aria-label="Tiempo efectivo de entrenamiento"></div>
        <h3 class="metrics-subheading">Duración completa del run de MLflow</h3>
        <div id="training-time-chart" class="training-chart" aria-label="Duración de runs"></div>
        <h3 class="metrics-subheading">Seis runs principales · hiperparámetros efectivos</h3>
        <div class="training-table-scroll"><table class="training-table"><thead><tr>
            <th>Run</th><th>Familia</th><th>Config.</th><th>Entrenamiento</th><th>Run completo</th><th>Macro-F1</th><th>GPU registrada</th><th>USD equivalente*</th><th>Parámetros MLflow</th>
        </tr></thead><tbody id="training-history"></tbody></table></div>
        <p class="training-note">* USD equivalente es una <strong>simulación de alquiler</strong> por la duración completa del run, no el costo pagado ni una tarifa de Colab: T4 / AWS g4dn.xlarge US$0,526/h; A100 80 GB / RunPod US$2,19/h; B200 / RunPod US$7,99/h (referencias aproximadas de noviembre de 2026). No incluye otros cargos. En TF-IDF, T4 identifica el entorno, no implica entrenamiento con GPU.</p>

        <h3 class="metrics-subheading">Evaluación por API · experimento 13 · test-revisado</h3>
        <p class="training-note">Ocho runs históricos: Gemini 3.8 Flash y GPT-4.1 mini, A/B, zero-shot/few-shot. <strong>No hay entrenamiento local para estas APIs.</strong> Gemini 3.5 Flash-Lite es el modelo operativo y sus mediciones están en «Tiempos y costos», no en estos resultados históricos.</p>
        <div class="training-toolbar">
            <label for="training-api-family">Proveedor experimental</label>
            <select id="training-api-family"><option value="all">Gemini y GPT</option></select>
            <label for="training-api-configuration">Configuración API</label>
            <select id="training-api-configuration"><option value="all">A y B</option><option value="A">A</option><option value="B">B</option></select>
        </div>
        <div class="training-overview">
            <div class="training-stat"><small>Runs API visibles</small><strong id="training-api-count">—</strong></div>
            <div class="training-stat"><small>Mejor macro-F1 · test-revisado</small><strong id="training-api-best">—</strong></div>
            <div class="training-stat"><small>Test de referencia</small><strong class="training-source-small">test-revisado</strong></div>
        </div>
        <div id="training-api-chart" class="training-chart" aria-label="Macro-F1 API en test-revisado"></div>
        <div class="training-table-scroll"><table class="training-table"><thead><tr>
            <th>Run</th><th>Modelo API</th><th>Config.</th><th>Estrategia</th><th>Macro-F1</th><th>Costo A7 (USD)*</th><th>Tokens entrada / salida</th><th>Latencia A7</th><th>Parámetros MLflow</th>
        </tr></thead><tbody id="training-api-history"></tbody></table></div>
        <p class="training-note">* Costo API A7 = estimación registrada por MLflow, no una factura auditada. Se presenta una sola vez para cada configuración del experimento 13. Los runs de API usan latencia media por instancia, no la duración redondeada del registro.</p>

        <details class="training-archive" id="training-archive">
            <summary>Historial adicional (17 runs) · excluido de la comparación principal</summary>
            <p class="training-note">Este historial se conserva para trazabilidad; <strong>ninguna de sus métricas alimenta los KPI o las gráficas superiores</strong>. Las corridas labeled-v2 corresponden a otro hash de test. No deben mezclarse con test-revisado.</p>
            <h3 class="metrics-subheading">Alternativa en test-revisado (1 run)</h3>
            <div class="training-table-scroll"><table class="training-table"><thead><tr><th>Run</th><th>Test</th><th>Macro-F1</th><th>GPU</th><th>Duración</th><th>Parámetros</th></tr></thead><tbody id="training-other-reference"></tbody></table></div>
            <h3 class="metrics-subheading">Modelos históricos en labeled-v2 (8 runs)</h3>
            <div class="training-table-scroll"><table class="training-table"><thead><tr><th>Run</th><th>Test</th><th>Macro-F1 histórico</th><th>GPU</th><th>Duración</th><th>Parámetros</th></tr></thead><tbody id="training-old-models"></tbody></table></div>
            <h3 class="metrics-subheading">Evaluaciones API históricas en labeled-v2 (8 runs)</h3>
            <div class="training-table-scroll"><table class="training-table"><thead><tr><th>Run</th><th>Test</th><th>Macro-F1 histórico</th><th>Latencia A7</th><th>Parámetros</th></tr></thead><tbody id="training-old-api"></tbody></table></div>
            <p class="training-note">Las estimaciones de costos A7 de API pueden repetirse entre los experimentos 12 y 13 para la misma configuración; este historial no las suma ni las factura nuevamente. Los seis runs eliminados de MLflow permanecen excluidos.</p>
        </details>
        <p class="training-note">El macro-F1 evalúa el rendimiento de clasificación; no es la confianza de una predicción. La duración del run y el tiempo efectivo de entrenamiento no son equivalentes.</p>
    `;
    tab.appendChild(panel);

    const $ = selector => panel.querySelector(selector);
    const familyFilter = $('#training-family');
    const configFilter = $('#training-configuration');
    const apiFamilyFilter = $('#training-api-family');
    const apiConfigFilter = $('#training-api-configuration');
    const sourceEl = $('#training-source');
    let runs = [], apiRuns = [], otherRuns = [], oldRuns = [], oldApiRuns = [];

    const text = (node, value) => { node.textContent = String(value); };
    const appendCell = (row, value) => {
        const td = document.createElement('td');
        text(td, value);
        row.appendChild(td);
        return td;
    };
    const numeric = value => value == null || value === '' || !Number.isFinite(Number(value))
        ? null : Number(value);
    const durationValue = run => {
        const ms = numeric(run.duration_ms);
        if (ms != null) return ms / 60000;
        return numeric(run.duration_minutes);
    };
    const durationLabel = run => {
        const value = durationValue(run);
        return value == null ? 'N/D' : `${run.duration_is_approximate ? '≈ ' : ''}${value.toFixed(1)} min`;
    };
    const trainingTime = run => {
        const seconds = numeric(run.metrics?.a7_train_wall_time_sec);
        return seconds == null ? null : seconds / 60;
    };
    const trainingTimeLabel = run => trainingTime(run) == null ? 'N/D' : `${trainingTime(run).toFixed(2)} min`;
    const f1 = run => numeric(run.test_macro_f1_all_9);
    const f1Label = run => f1(run) == null ? 'N/D' : `${(f1(run) * 100).toFixed(2)} %`;
    const name = run => `${run.family || run.model_id || 'Otro'} ${run.configuration || ''}${run.shot_mode ? ` (${run.shot_mode})` : ''}`.trim();

    function drawChart(container, items, metric, maxValue, formatter) {
        container.replaceChildren();
        if (!items.length) { text(container, 'No hay resultados para este filtro.'); return; }
        for (const run of items) {
            const value = metric(run);
            const row = document.createElement('div'); row.className = 'training-chart-row';
            const title = document.createElement('span'); text(title, name(run));
            const track = document.createElement('div'); track.className = 'training-track';
            const fill = document.createElement('div'); fill.className = 'training-fill';
            fill.style.width = value == null ? '0%' : `${Math.min(100, Math.max(0, value / maxValue * 100))}%`;
            track.appendChild(fill);
            const label = document.createElement('strong'); text(label, formatter(run));
            row.append(title, track, label); container.appendChild(row);
        }
    }

    // Equivalencia de infraestructura: NO factura de Colab o costo efectivamente cobrado.
    const gpuRates = [
        {pattern:/\bB200\b/i, rate:7.99, label:'RunPod B200 Secure Cloud'},
        {pattern:/A100-SXM4-80GB|A100.*SXM/i, rate:2.19, label:'RunPod A100 80 GB Secure Cloud'},
        {pattern:/\bT4\b/i, rate:0.526, label:'AWS g4dn.xlarge us-east-1'}
    ];
    function equivalentCost(run) {
        const minutes = durationValue(run);
        const rate = gpuRates.find(item => item.pattern.test(run.gpu || ''));
        return rate && minutes != null ? {amount: minutes / 60 * rate.rate, label:rate.label} : null;
    }
    function fillParamsCell(cell, run) {
        const entries = Object.entries(run.params || {});
        if (!entries.length) { text(cell, 'N/D'); return; }
        const details = document.createElement('details'); details.className = 'training-details';
        const summary = document.createElement('summary'); text(summary, `Ver ${entries.length} parámetros`);
        const dl = document.createElement('dl');
        for (const [key, value] of entries) {
            const dt = document.createElement('dt'); text(dt, key);
            const dd = document.createElement('dd'); text(dd, value);
            dl.append(dt, dd);
        }
        details.append(summary, dl); cell.appendChild(details);
    }
    function populateSelect(select, values, allLabel) {
        select.replaceChildren();
        const all = document.createElement('option'); all.value = 'all'; text(all, allLabel); select.appendChild(all);
        for (const value of values) {
            const opt = document.createElement('option'); opt.value = value; text(opt, value); select.appendChild(opt);
        }
    }
    function showEmpty(tbody, colSpan, message) {
        if (tbody.children.length) return;
        const row = document.createElement('tr');
        const cell = appendCell(row, message); cell.colSpan = colSpan; tbody.appendChild(row);
    }
    function renderMain() {
        const filtered = runs.filter(run =>
            (familyFilter.value === 'all' || run.family === familyFilter.value) &&
            (configFilter.value === 'all' || run.configuration === configFilter.value)
        );
        const ranked = filtered.filter(run => f1(run) != null).sort((a,b) => f1(b) - f1(a));
        const effective = filtered.map(trainingTime).filter(value => value != null);
        text($('#training-run-count'), filtered.length);
        text($('#training-best-f1'), ranked.length ? f1Label(ranked[0]) : 'N/D');
        text($('#training-total-time'), effective.length ? `${effective.reduce((a,b) => a+b, 0).toFixed(1)} min` : 'N/D');
        drawChart($('#training-f1-chart'), ranked, f1, 1, f1Label);
        const wall = filtered.filter(run => trainingTime(run) != null).sort((a,b) => trainingTime(a) - trainingTime(b));
        drawChart($('#training-wall-chart'), wall, trainingTime,
            Math.max(1, ...wall.map(trainingTime)), trainingTimeLabel);
        const timed = filtered.filter(run => durationValue(run) != null).sort((a,b) => durationValue(a)-durationValue(b));
        drawChart($('#training-time-chart'), timed, durationValue,
            Math.max(1, ...timed.map(durationValue)), durationLabel);
        const tbody = $('#training-history'); tbody.replaceChildren();
        for (const run of filtered) {
            const row = document.createElement('tr');
            appendCell(row, run.run_name || 'Sin nombre');
            appendCell(row, run.family || 'N/D');
            appendCell(row, run.configuration || 'N/D');
            appendCell(row, trainingTimeLabel(run));
            appendCell(row, durationLabel(run));
            appendCell(row, f1Label(run));
            appendCell(row, run.gpu || 'N/D');
            const cost = equivalentCost(run);
            const costCell = appendCell(row, cost ? `≈ US$${cost.amount.toFixed(2)}` : 'N/D');
            if (cost) costCell.title = `Solo escenario: ${cost.label}`;
            fillParamsCell(appendCell(row, ''), run);
            tbody.appendChild(row);
        }
        showEmpty(tbody, 9, 'No hay corridas principales con los filtros seleccionados.');
    }
    function renderApi() {
        const filtered = apiRuns.filter(run =>
            (apiFamilyFilter.value === 'all' || run.family === apiFamilyFilter.value) &&
            (apiConfigFilter.value === 'all' || run.configuration === apiConfigFilter.value)
        );
        const ranked = filtered.filter(run => f1(run) != null).sort((a,b) => f1(b)-f1(a));
        text($('#training-api-count'), filtered.length);
        text($('#training-api-best'), ranked.length ? f1Label(ranked[0]) : 'N/D');
        drawChart($('#training-api-chart'), ranked, f1, 1, f1Label);
        const tbody = $('#training-api-history'); tbody.replaceChildren();
        for (const run of filtered) {
            const row = document.createElement('tr');
            appendCell(row, run.run_name || 'Sin nombre');
            appendCell(row, run.family || 'N/D');
            appendCell(row, run.configuration || 'N/D');
            appendCell(row, run.shot_mode || 'N/D');
            appendCell(row, f1Label(run));
            const cost = numeric(run.metrics?.a7_api_cost_usd);
            const costCell = appendCell(row, cost == null ? 'N/D' : `US$${cost.toFixed(2)}`);
            const gross = numeric(run.metrics?.a7_api_gross_cost_usd);
            if (gross != null) costCell.title = `Costo bruto A7 sin descuentos de cache: US$${gross.toFixed(2)}`;
            const input = numeric(run.metrics?.a7_total_input_tokens);
            const output = numeric(run.metrics?.a7_total_output_tokens);
            appendCell(row, input == null || output == null ? 'N/D' :
                `${input.toLocaleString('es-CO')} / ${output.toLocaleString('es-CO')}`);
            const latency = numeric(run.metrics?.a7_latency_ms);
            appendCell(row, latency == null ? 'N/D' : `${(latency/1000).toFixed(2)} s/instancia`);
            fillParamsCell(appendCell(row, ''), run);
            tbody.appendChild(row);
        }
        showEmpty(tbody, 9, 'No hay evaluaciones API con los filtros seleccionados.');
    }
    function renderArchives() {
        function fillModels(tbodySelector, archived) {
            const tbody = $(tbodySelector); tbody.replaceChildren();
            for (const run of archived) {
                const row = document.createElement('tr');
                appendCell(row, run.run_name || 'Sin nombre');
                appendCell(row, run.dataset_tag || 'N/D');
                appendCell(row, f1Label(run));
                appendCell(row, run.gpu || 'N/D');
                appendCell(row, durationLabel(run));
                fillParamsCell(appendCell(row,''), run);
                tbody.appendChild(row);
            }
        }
        fillModels('#training-other-reference', otherRuns);
        fillModels('#training-old-models', oldRuns);
        const tbody = $('#training-old-api'); tbody.replaceChildren();
        for (const run of oldApiRuns) {
            const row = document.createElement('tr');
            appendCell(row, run.run_name || 'Sin nombre');
            appendCell(row, run.dataset_tag || 'N/D');
            appendCell(row, f1Label(run));
            const latency = numeric(run.metrics?.a7_latency_ms);
            appendCell(row, latency == null ? 'N/D' : `${(latency/1000).toFixed(2)} s/instancia`);
            fillParamsCell(appendCell(row,''), run);
            tbody.appendChild(row);
        }
    }
    async function load() {
        try {
            const response = await fetch('/static/mlflow-training.json', {cache:'no-store'});
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            if (data.schema_version !== 4 || data.reference_test !== 'test-revisado' ||
                !Array.isArray(data.runs) || !Array.isArray(data.api_runs) ||
                !Array.isArray(data.other_reference_runs) || !Array.isArray(data.historical_runs) ||
                !Array.isArray(data.historical_api_runs)) throw new Error('Snapshot MLflow v4 inválido');
            // Defensa: aunque el archivo se cambie por otro, el tablero principal NUNCA mezcla datasets.
            runs = data.runs.filter(r => r.dataset_tag === 'test-revisado' && !r.is_alternate &&
                r.params?.model_family !== 'llm_api');
            apiRuns = data.api_runs.filter(r => r.dataset_tag === 'test-revisado' &&
                r.params?.model_family === 'llm_api');
            otherRuns = data.other_reference_runs;
            oldRuns = data.historical_runs;
            oldApiRuns = data.historical_api_runs;
            populateSelect(familyFilter, [...new Set(runs.map(r => r.family).filter(Boolean))].sort(), 'Todas');
            populateSelect(apiFamilyFilter, [...new Set(apiRuns.map(r => r.family).filter(Boolean))].sort(), 'Gemini y GPT');
            const when = data.exported_at_utc ? new Date(data.exported_at_utc).toLocaleString('es-CO') : 'N/D';
            text(sourceEl, `Fuente: MLflow #12 y #13 · exportado ${when} · referencia fija: test-revisado. ` +
                `${runs.length} runs de modelos entrenados + ${apiRuns.length} evaluaciones API en el panel principal. ` +
                `${otherRuns.length+oldRuns.length+oldApiRuns.length} ejecuciones adicionales archivadas.`);
            renderMain(); renderApi(); renderArchives();
        } catch (error) {
            text(sourceEl, `No fue posible cargar resultados de MLflow: ${error.message}`);
        }
    }
    familyFilter.addEventListener('change', renderMain);
    configFilter.addEventListener('change', renderMain);
    apiFamilyFilter.addEventListener('change', renderApi);
    apiConfigFilter.addEventListener('change', renderApi);
    load();
})();
