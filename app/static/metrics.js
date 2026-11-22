// Dashboard de inferencia. La pestaña de entrenamiento se alimentará de MLflow.
(() => {
    const buttons = [...document.querySelectorAll('.tab[data-tab]')];
    const panels = [...document.querySelectorAll('.tab-content')];
    const modelFilter = document.getElementById('metrics-model');
    const configFilter = document.getElementById('metrics-config');
    const status = document.getElementById('metrics-status');
    let allRuns = [];

    const formatMs = ms => ms == null || !Number.isFinite(+ms)
        ? 'N/D' : `${(+ms / 1000).toFixed(2)} s`;
    const fmtUsd = value => value == null || !Number.isFinite(+value)
        ? 'N/D' : `$${Number(value).toFixed(6)}`;
    const short = id => ({'tfidf':'TF-IDF', 'scibert':'SciBERT',
        'gemini':'Gemini', 'granite':'Granite'})[id.split('-')[0]] || id;
    const avg = (items, field) => {
        const values = items.map(x => x[field]).filter(x => x != null && Number.isFinite(+x));
        return values.length ? values.reduce((a, b) => a + +b, 0) / values.length : null;
    };
    const text = (node, value) => { node.textContent = value; };
    const cell = (row, value) => {
        const td = document.createElement('td');
        td.textContent = value;
        row.appendChild(td);
    };

    function filtered() {
        return allRuns.filter(run =>
            (modelFilter.value === 'all' || run.model_id === modelFilter.value) &&
            (configFilter.value === 'all' || run.configuration === configFilter.value)
        );
    }

    function render() {
        const runs = filtered();
        const successes = runs.filter(x => x.status === 'ok' && !x.is_demo);
        const withCosts = runs.filter(x => x.cost_usd != null);
        text(document.getElementById('metric-count'), String(runs.length));
        text(document.getElementById('metric-ok'), String(successes.length));
        text(document.getElementById('metric-mean'), formatMs(avg(successes, 'server_ms')));
        text(document.getElementById('metric-cost'),
            withCosts.length ? fmtUsd(withCosts.reduce((sum, x) => sum + x.cost_usd, 0)) : 'N/D');
        text(document.getElementById('metric-cost-note'),
            withCosts.length ? `Costo calculado para ${withCosts.length} de ${runs.length} ejecuciones` :
            'Sin tarifas verificadas todavía');

        const byModel = new Map();
        for (const run of successes) {
            const group = byModel.get(run.model_id) || [];
            group.push(run);
            byModel.set(run.model_id, group);
        }
        const chart = document.getElementById('metrics-chart');
        chart.replaceChildren();
        const series = [...byModel.entries()].map(([id, group]) => ({
            id, count: group.length, ms: avg(group, 'server_ms')
        })).sort((a, b) => a.ms - b.ms);
        const max = Math.max(1, ...series.map(x => x.ms || 0));
        if (!series.length) {
            text(chart, 'Todavía no hay predicciones registradas para estos filtros.');
        }
        for (const item of series) {
            const row = document.createElement('div');
            row.className = 'metrics-bar-row';
            const name = document.createElement('span');
            name.textContent = `${short(item.id)} ${item.id.endsWith('-a') ? 'A' : 'B'} (${item.count})`;
            const track = document.createElement('div');
            track.className = 'metrics-bar-track';
            const fill = document.createElement('div');
            fill.className = 'metrics-bar-fill';
            fill.style.width = `${Math.max(1, 100 * item.ms / max)}%`;
            track.appendChild(fill);
            const value = document.createElement('strong');
            value.textContent = formatMs(item.ms);
            row.append(name, track, value);
            chart.appendChild(row);
        }
        const tbody = document.getElementById('metrics-rows');
        tbody.replaceChildren();
        if (!runs.length) {
            const tr = document.createElement('tr');
            const td = document.createElement('td');
            td.colSpan = 9;
            td.textContent = 'Sin registros todavía.';
            tr.appendChild(td);
            tbody.appendChild(tr);
        }
        for (const run of runs.slice(0, 200)) {
            const row = document.createElement('tr');
            const reportedMs = run.inference_ms;
            const suspect = reportedMs != null && reportedMs > run.server_ms + 100;
            cell(row, new Date(run.created_at).toLocaleString('es-CO'));
            cell(row, run.model_id);
            cell(row, run.status === 'ok' ? 'Correcta' : 'Error');
            const confidence = run.confidence == null
                ? null : Number(run.confidence);
            const label = run.predicted_label || '-';
            const confidenceText = confidence != null &&
                Number.isFinite(confidence)
                ? ` (${(confidence * 100).toFixed(1).replace('.', ',')}%)`
                : '';
            cell(row, label + confidenceText);
            cell(row, formatMs(run.server_ms));
            cell(row, formatMs(reportedMs) + (suspect ? ' ⚠' : ''));
            cell(row, formatMs(run.client_ms));
            cell(row, run.input_tokens == null ? 'N/D' :
                `${run.input_tokens} / ${run.output_tokens ?? '?'}`);
            cell(row, fmtUsd(run.cost_usd));
            tbody.appendChild(row);
        }
    }

    async function refresh() {
        status.textContent = 'Actualizando métricas...';
        try {
            const response = await fetch('/api/v1/metrics/runs?limit=1000', {cache: 'no-store'});
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            allRuns = (await response.json()).runs || [];
            const ids = [...new Set(allRuns.map(x => x.model_id))].sort();
            const previous = modelFilter.value;
            modelFilter.replaceChildren();
            const every = document.createElement('option');
            every.value = 'all'; every.textContent = 'Todos los modelos';
            modelFilter.appendChild(every);
            for (const id of ids) {
                const opt = document.createElement('option');
                opt.value = id; opt.textContent = id;
                modelFilter.appendChild(opt);
            }
            modelFilter.value = ids.includes(previous) ? previous : 'all';
            render();
            status.textContent = `${allRuns.length} registros cargados. Se muestran los 200 más recientes en la tabla.`;
        } catch (err) {
            status.textContent = `No fue posible cargar métricas: ${err.message}`;
        }
    }

    for (const button of buttons) {
        button.addEventListener('click', () => {
            for (const item of buttons) {
                const active = item === button;
                item.classList.toggle('active', active);
                item.setAttribute('aria-selected', String(active));
            }
            for (const panel of panels) {
                panel.classList.toggle('active', panel.id === `${button.dataset.tab}-tab`);
            }
            if (button.dataset.tab === 'metrics') refresh();
        });
    }
    modelFilter.addEventListener('change', render);
    configFilter.addEventListener('change', render);
    document.getElementById('metrics-refresh').addEventListener('click', refresh);
})();
