/* ==========================================================================
   EvidenCite — Cajón / Overlay Oculto de Q&A (Tecla Q · 9 Pestañas)
   Invisible durante los 15 min principales; acceso instantáneo para preguntas
   ========================================================================== */

window.EvidenQA = (function () {
  const D = window.EVIDENCITE_DATA;
  let isOpen = false;
  let activeTab = "a6-01";

  const TABS = [
    { id: "a6-01", label: "A6-01 Calibración" },
    { id: "a4-04", label: "A4-04 Ablación" },
    { id: "a4-05", label: "A4-05 Confusiones" },
    { id: "a5-04", label: "A5-04 Vecindad/NLL" },
    { id: "tab5",  label: "Tabla V Signif." },
    { id: "tab6",  label: "Tabla VI Estratos" },
    { id: "a7-02", label: "A7-02 Pensam./Caché" },
    { id: "a1-04", label: "A1-04 Ec. 2 NLI" },
    { id: "audit", label: "Auditoría Forense" },
    { id: "conc3", label: "CONC-03 IAG" }
  ];

  function renderContent(id) {
    if (id === "a6-01") {
      return `
        <h3 class="qa-heading">A6-01 · Calibración Probabilística y Revisión Selectiva (10 bins · Gold N = 2.798)</h3>
        <p class="qa-summary">
          Gemini 0s-B funciona como un semáforo confiable (<strong>0,829 al acertar vs. 0,737 al fallar</strong>, Spearman ρ = +0,4150, p = 6,48×10⁻¹¹⁷), permitiendo automatizar alta confianza y derivar casos inciertos a revisión humana. GPT-4.1-mini sobreestima su confianza incluso al fallar (ECE = 0,3360).
        </p>
        <div id="qa-cal-svg" style="width:100%;max-width:750px;margin:0 auto;"></div>`;
    }

    if (id === "a4-04") {
      return `
        <h3 class="qa-heading">A4-04 · Ablación Clasificador vs. Generativo (Qwen3-0.6B) y Costo Local</h3>
        <p class="qa-summary">
          Comparar cabeza lineal de clasificación frente a generación causal con prompt demostró desempeño indistinguible (±0,02 Macro-F1), pero el modo generativo disparó VRAM y latencia sin beneficio.
        </p>
        <div class="qa-kpi-row">
          <div class="qa-kpi"><strong>6 → 85 GB VRAM</strong><span>Config A (Config B exigió gradient checkpointing en B200 180 GB)</span></div>
          <div class="qa-kpi"><strong>2,9–4,2× latencia</strong><span>7,9–9,2 ms generativo vs. 1,9–3,2 ms clasificador</span></div>
          <div class="qa-kpi"><strong>~$12,7 USD</strong><span>Entrenamiento de las 6 definitivas (Granite $12,44 · SciBERT 767 COP · TF-IDF 28 COP)</span></div>
        </div>
        <table class="qa-table">
          <thead>
            <tr><th>Variante Qwen3-0.6B</th><th>Auto Macro-F1</th><th>Gold Macro-F1</th><th>Latencia lote (ms)</th><th>VRAM pico</th></tr>
          </thead>
          <tbody>
            <tr class="is-hi"><td>Clasificador · Config A</td><td class="num">0,746</td><td class="num">0,456</td><td class="num">1,9 ms</td><td class="num">6 GB</td></tr>
            <tr><td>Generativo · Config A</td><td class="num">0,737</td><td class="num">0,434</td><td class="num">7,9 ms</td><td class="num">85 GB</td></tr>
            <tr class="is-hi"><td>Clasificador · Config B</td><td class="num">0,737</td><td class="num">0,442</td><td class="num">3,2 ms</td><td class="num">32 GB</td></tr>
            <tr><td>Generativo · Config B</td><td class="num">0,742</td><td class="num">0,436</td><td class="num">9,2 ms</td><td class="num">34 GB (grad. ckpt.)</td></tr>
          </tbody>
        </table>
        <div class="qa-note-box">
          <strong>Aclaración de alcance:</strong> ~$12,7 USD cubre únicamente las 6 corridas definitivas de Tabla III; excluye R1/R2 (92 corridas, 47,82 h GPU totales) y cómputo continuo en producción.
        </div>`;
    }

    if (id === "a4-05") {
      return `
        <h3 class="qa-heading">A4-05 · Patrones de Confusión por Familia Local (Figs. 12–14, Anexo A)</h3>
        <p class="qa-summary">
          Las tres familias locales reproducen el mismo sesgo heredado de las cuotas de entrenamiento casi uniformes: alta diagonal en <em>Further Reading</em> y colapso de recuperación en <em>Background</em>.
        </p>
        <div class="qa-kpi-row">
          <div class="qa-kpi"><strong>0,21 – 0,29</strong><span>Recall de Background en A4 (cuando es 42,89% del Gold)</span></div>
          <div class="qa-kpi"><strong>0,76 → 0,18</strong><span>Caída de F1 en Gap (dev automático vs. Gold humano)</span></div>
          <div class="qa-kpi"><strong>0,85 – 0,95</strong><span>Recall sostenido en Further Reading (marcadores léxicos unívocos)</span></div>
        </div>
        <table class="qa-table">
          <thead>
            <tr><th>Familia Local</th><th>Macro-F1 A → B (Gold)</th><th>Efecto diferencial B − A por clase</th></tr>
          </thead>
          <tbody>
            <tr><td><strong>TF-IDF + RL</strong></td><td class="num">0,366 → 0,332 (−0,034)</td><td>Config B reduce exhaustividad en las 9 clases (hasta −0,10 en Basis y Originator).</td></tr>
            <tr><td><strong>SciBERT 110M</strong></td><td class="num">0,462 → 0,433 (−0,029)</td><td>B baja Background (0,29→0,21) y Application (0,63→0,53); sube Originator (0,64→0,70). En Config A dispersa Background hacia Gap (0,18) y Originator (0,11).</td></tr>
            <tr class="is-hi"><td><strong>Granite-4.2-8B LoRA</strong></td><td class="num">0,435 → 0,441 (+0,006)</td><td>B sube Application (0,55→0,69) y Further Reading (0,89→0,95), pero baja Basis (0,78→0,59) e Improvement (0,63→0,31); Background casi inmóvil (0,21→0,23).</td></tr>
          </tbody>
        </table>`;
    }

    if (id === "a5-04") {
      return `
        <h3 class="qa-heading">A5-04 · Matriz de Vecindad Retórica en Δ⁸ (Tabla VII) y Auditoría JSON</h3>
        <p class="qa-summary">
          Para evitar copia rígida de plantillas en 9-Shot (p. ej. 0,92/0,01), sintetizamos determinísticamente (semilla SHA-256) la masa residual entre vecinos retóricos primarios (58–70%) y secundarios (20–28%).
        </p>
        <div class="qa-kpi-row">
          <div class="qa-kpi"><strong>4× menos ceros</strong><span>Probabilidades nulas exactas (26,9–40,9% → 6,5–10,8%)</span></div>
          <div class="qa-kpi"><strong>5,68 → 2,05</strong><span>Caída de NLL en GPT (Gemini: 1,23 → 1,07)</span></div>
          <div class="qa-kpi"><strong>99,991% válido</strong><span>22.382 / 22.384 JSON válidos; 2 fallos truncados (+1 tok), ΔF1 ≤ 0,0003</span></div>
        </div>
        <table class="qa-table">
          <thead>
            <tr><th>Clase Real (c*)</th><th>Banda Dominante [L; U]</th><th>Vecinos Primarios N₁</th><th>Vecinos Secundarios N₂</th></tr>
          </thead>
          <tbody>
            ${D.qaTablaVII.map((r) => `
              <tr>
                <td><strong>${r.cls}</strong></td>
                <td class="num">${r.band}</td>
                <td>${r.n1}</td>
                <td>${r.n2}</td>
              </tr>`).join("")}
          </tbody>
        </table>`;
    }

    if (id === "tab5") {
      return `
        <h3 class="qa-heading">Tabla V · Pruebas Pareadas de Significancia (Bootstrap B = 10.000 y McNemar)</h3>
        <p class="qa-summary">
          Ajuste múltiple Holm-Bonferroni (α = 0,05, M = 12 contrastes) sobre el conjunto de prueba Gold (N = 2.798).
        </p>
        <table class="qa-table">
          <thead>
            <tr><th>Eje</th><th>Contraste</th><th>F1 Base</th><th>F1 Comp.</th><th>Δ Macro-F1</th><th>p (Boot)</th><th>Holm B</th><th>p (McNemar)</th><th>Holm M</th></tr>
          </thead>
          <tbody>
            ${D.qaTablaV.map((r) => `
              <tr class="${r.holmBoot || r.pair.includes("GPT 0-Shot") ? "is-hi" : ""}">
                <td>${r.axis}</td>
                <td>${r.pair}</td>
                <td class="num">${r.base}</td>
                <td class="num">${r.comp}</td>
                <td class="num">${r.delta}</td>
                <td class="num">${r.pBoot}</td>
                <td><span class="${r.holmBoot ? "qa-tag-yes" : "qa-tag-no"}">${r.holmBoot ? "SÍ" : "NO"}</span></td>
                <td class="num">${r.pMcN}</td>
                <td><span class="${r.holmMcN ? "qa-tag-yes" : "qa-tag-no"}">${r.holmMcN ? "SÍ" : "NO"}</span></td>
              </tr>`).join("")}
          </tbody>
        </table>`;
    }

    if (id === "tab6") {
      return `
        <h3 class="qa-heading">Tabla VI · Exactitud por Estrato Previo A1/A2 y Acotamiento de Circularidad A3</h3>
        <p class="qa-summary">
          Desglose de exactitud sobre el Gold humano (N = 2.798). Muestra que Gemini 0s-B supera a GPT 0s-B por +8 a +20 pp en <strong>todos</strong> los estratos, incluso en MultiCite puro y donde el panel A2 acertó.
        </p>
        <table class="qa-table">
          <thead>
            <tr><th>Estrato Previo A1/A2</th><th>n (%)</th><th>GPT 0s-B</th><th>GPT 9s-B</th><th>Gemini 0s-B</th><th>Gemini 9s-B</th></tr>
          </thead>
          <tbody>
            ${D.qaTablaVI.map((r) => `
              <tr class="${r.hi ? "is-hi" : ""}">
                <td><strong>${r.stratum}</strong></td>
                <td class="num">${r.n}</td>
                <td class="num">${r.gpt0b}</td>
                <td class="num">${r.gpt9b}</td>
                <td class="num"><strong>${r.gem0b}</strong></td>
                <td class="num">${r.gem9b}</td>
              </tr>`).join("")}
          </tbody>
        </table>
        <div class="qa-note-box">
          <strong>Respuesta a circularidad del 32% en A3:</strong> En 894/901 desempates, los 3 LLMs árbitros (Claude, GPT, Gemini) solo pudieron escoger entre las dos etiquetas propuestas por revisores humanos. Además, en <strong>MultiCite puro (+8,88 pp)</strong> y donde <strong>A2 acertó (+8,25 pp)</strong> la ventaja de Gemini se sostiene ampliamente.
        </div>`;
    }

    if (id === "a7-02") {
      return `
        <h3 class="qa-heading">A7-02 · Anatomía del Pensamiento Latente y Ahorro por Prompt Caching</h3>
        <p class="qa-summary">
          El pensamiento latente en Gemini 0s-B gobernó la cola de latencia sin mejorar la exactitud, demostrando que la ventaja de Gemini sobre GPT-4.1-mini <strong>no</strong> depende del canal de pensamiento.
        </p>
        <div class="qa-kpi-row">
          <div class="qa-kpi"><strong>68,74% vs. 67,79%</strong><span>Exactitud Gemini SIN vs. CON pensamiento (p = 0,609, n.s.)</span></div>
          <div class="qa-kpi"><strong>3,89 s → 6,14 s</strong><span>Aumento de latencia p95 al activar pensamiento (ρ = +0,5567)</span></div>
          <div class="qa-kpi"><strong>$55,14 → $44,49</strong><span>Costo neto global de las 22.384 llamadas (−19,31% por caché)</span></div>
        </div>
        <div class="qa-two-col">
          <div>
            <table class="qa-table">
              <thead><tr><th>Métrica Gemini 0s-B</th><th>Sin pensamiento</th><th>Con pensamiento</th></tr></thead>
              <tbody>
                <tr><td>Llamadas (n = 2.798)</td><td class="num">1.820 (65,05%)</td><td class="num">978 (34,95%)</td></tr>
                <tr class="is-hi"><td>Exactitud Gold</td><td class="num">68,74%</td><td class="num">67,79% (p=0,609)</td></tr>
                <tr><td>Latencia p95</td><td class="num">3,89 s</td><td class="num">6,14 s</td></tr>
              </tbody>
            </table>
          </div>
          <div>
            <table class="qa-table">
              <thead><tr><th>Prompt Caching</th><th>0-Shot</th><th>9-Shot</th></tr></thead>
              <tbody>
                <tr><td><strong>OpenAI (GPT-4.1-mini)</strong></td><td>Inactivo (&lt;1.024 tok prefijo)</td><td class="num">&gt;99,3% hit · −59–60,8% costo</td></tr>
                <tr><td><strong>Vertex AI (Gemini)</strong></td><td>Implícito bajo</td><td class="num">4,3–13,4% reutilización</td></tr>
                <tr class="is-hi"><td><strong>Config B en Gemini 0s</strong></td><td colspan="2">−19,4% llamadas con pensamiento (1.214→978) y −17,7% tokens</td></tr>
              </tbody>
            </table>
          </div>
        </div>`;
    }

    if (id === "a1-04") {
      return `
        <h3 class="qa-heading">A1-04 · Cota de Co-ocurrencia en Δ⁸ (Ec. 2) y Calibración NLI</h3>
        <p class="qa-summary">
          En funciones retóricas co-localizadas (p. ej. <em>Improvement</em> + <em>Basis</em>), la probabilidad superior queda acotada por <strong>P₁ = (1 − ε + Δ) / 2</strong>. Con masa residual ε ≈ 0,19 y margen claro Δ = 0,35, se obtiene <strong>P₁ ≤ 0,580 &lt; 0,70</strong>, bloqueando el umbral preliminar.
        </p>
        <div class="qa-kpi-row">
          <div class="qa-kpi"><strong>6,6% → 16,1%</strong><span>Admisión NLI (2,44× en muestra estratificada N = 1.000)</span></div>
          <div class="qa-kpi"><strong>1 → 13 (13×)</strong><span>Captura de Improvement por cada 1.000 candidatos</span></div>
          <div class="qa-kpi"><strong>2 → 11 (5,5×)</strong><span>Captura de Basis por cada 1.000 candidatos</span></div>
        </div>
        <table class="qa-table">
          <thead><tr><th>Régimen NLI</th><th>Parámetros (τ, θ_conf, θ_margin)</th><th>Aceptación</th><th>Improvement/1k</th><th>Comparison/1k</th><th>Basis/1k</th></tr></thead>
          <tbody>
            <tr><td>R1 Preliminar</td><td class="num">(0,7 · 0,70 · 0,15)</td><td class="num">6,6%</td><td class="num">1</td><td class="num">5</td><td class="num">2</td></tr>
            <tr class="is-hi"><td>R4 Calibrado Oficial</td><td class="num">(0,6 · 0,50 · 0,15)</td><td class="num">16,1%</td><td class="num">13</td><td class="num">10</td><td class="num">11</td></tr>
          </tbody>
        </table>`;
    }

    if (id === "audit") {
      return `
        <h3 class="qa-heading">Auditoría Forense — Flancos Técnicos Verificados</h3>
        <p class="qa-summary">
          Respuestas exactas y verificadas directamente en código y datos ante las preguntas más incisivas del jurado.
        </p>
        <table class="qa-table">
          <thead><tr><th>Flanco / Pregunta probable</th><th>Dato auditado exacto y explicación técnica</th></tr></thead>
          <tbody>
            <tr class="is-hi">
              <td><strong>Solapamiento por artículo CITADO (<code>cited_paper_id</code>)</strong></td>
              <td><code>citing_paper_id</code> tiene <strong>leakage = 0</strong> (8.336 citantes disjuntos, estándar MultiCite/ACL-ARC). En <code>cited_paper_id</code> se repiten <strong>621 artículos únicos</strong>, cubriendo <strong>2.110 / 2.798 (75,41%)</strong> del test (papers seminales citados con distintas funciones). Pese a ver el 75,4% de los citados en train, A4 no superó 0,46.</td>
            </tr>
            <tr>
              <td><strong>¿Por qué SciBERT A (7,2 ms) parece más lento que B (6,6 ms)?</strong></td>
              <td>Nota † de Tabla III: <code>SciBERT A</code> se ejecutó en <strong>Colab T4</strong> (24,65 min) y <code>SciBERT B</code> en <strong>Colab A100</strong> (7,12 min). Además, la latencia por lotes de A4 no es equiparable al p50 HTTP individual de A5.</td>
            </tr>
            <tr>
              <td><strong>Granite y Gemini en EC2 vs. Artículo</strong></td>
              <td>El worker <code>api/workers/granite_service.py</code> ejecuta Granite-4.2-8B en <strong>CPU INT8</strong> con <em>hot-swap</em> de adaptadores LoRA A/B y exige <strong>14 GiB RAM</strong>; la EC2 de AWS Academy es <code>t3.large</code> (8 GiB). Para la app pública se usa <code>gemini-3.5-flash-lite</code> sin pensamiento (contención de costos) separando en UI los runs históricos de MLflow.</td>
            </tr>
            <tr>
              <td><strong>Calibración ausente (<code>---</code>) en A4</strong></td>
              <td>Limitación declarada (Sección IV-C): las corridas finales de A4 guardaron solo <code>argmax</code> por instancia y no la distribución Softmax completa, por lo que ECE/Brier/NLL se reportaron únicamente sobre las 8 variantes API.</td>
            </tr>
            <tr>
              <td><strong>Cambios en MultiCite Gold (35,7% = 201/563)</strong></td>
              <td>MultiCite era originalmente multi-etiqueta en 7 clases; al proyectar a 9 clases exclusivas de Lyu et al., 98 pasaron a <em>Background</em>, 39 a <em>Application</em>, 24 a <em>Comparison</em> y 30 migraron a las 4 clases inexistentes en MultiCite.</td>
            </tr>
          </tbody>
        </table>`;
    }

    // CONC-03
    return `
      <h3 class="qa-heading">CONC-03 · Declaración sobre el Uso de Inteligencia Artificial Generativa</h3>
      <p class="qa-summary">
        Cumplimiento de los lineamientos institucionales de la Universidad de los Andes sobre transparencia y rigor científico en el uso de IAG.
      </p>
      <div class="qa-kpi-row">
        <div class="qa-kpi"><strong>AI-DLC</strong><span>Ciclo asistido por IA (Google Antigravity y Claude) con revisión humana en cada hito</span></div>
        <div class="qa-kpi"><strong>Verificación real</strong><span>Pruebas unitarias pytest, runs trazables en MLflow/DVC y auditoría directa sobre JSONL/CSV</span></div>
        <div class="qa-kpi"><strong>Privacidad</strong><span>100% literatura científica pública; sin datos personales ni credenciales expuestas</span></div>
      </div>
      <div class="qa-note-box">
        La IAG apoyó especificación (<em>Spec-First</em>), codificación modular, depuración, visualización y edición LaTeX, sin reemplazar el diseño experimental, el criterio de curaduría humana (A3) ni la validación numérica de resultados.
      </div>`;
  }

  function renderTabs() {
    const tabBar = document.getElementById("qa-tabs");
    const body = document.getElementById("qa-body");
    if (!tabBar || !body) return;

    tabBar.innerHTML = TABS.map((t) => `
      <button class="qa-tab ${t.id === activeTab ? "is-active" : ""}" role="tab" aria-selected="${t.id === activeTab}" data-qa-tab="${t.id}">
        ${t.label}
      </button>`).join("");

    tabBar.querySelectorAll("[data-qa-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        activeTab = btn.getAttribute("data-qa-tab");
        renderTabs();
      });
    });

    body.innerHTML = renderContent(activeTab);
    if (activeTab === "a6-01") {
      const mount = document.getElementById("qa-cal-svg");
      if (mount && window.EvidenCharts) {
        window.EvidenCharts.renderA6Calibration(mount, D.a6Calibration);
      }
    }
    body.scrollTop = 0;
  }

  function open(tabId) {
    const modal = document.getElementById("qa-modal");
    if (!modal) return;
    if (tabId) activeTab = tabId;
    isOpen = true;
    renderTabs();
    modal.classList.add("is-open");
  }

  function close() {
    const modal = document.getElementById("qa-modal");
    if (!modal) return;
    isOpen = false;
    modal.classList.remove("is-open");
  }

  function toggle() {
    if (isOpen) close();
    else open();
  }

  function init() {
    const modal = document.getElementById("qa-modal");
    const closeBtn = document.getElementById("qa-close");
    if (closeBtn) closeBtn.addEventListener("click", close);
    if (modal) {
      modal.addEventListener("click", (e) => {
        if (e.target === modal) close();
      });
    }
  }

  return {
    init,
    open,
    close,
    toggle,
    isOpen: () => isOpen
  };
})();
