/* ==========================================================================
   EvidenCite — Controlador de Escenas por Beat (Estado Determinista)
   ========================================================================== */

window.EvidenScenes = (function () {
  const D = window.EVIDENCITE_DATA;
  const C = window.EvidenCharts;

  // Persistent controller caches for smooth morphs without destroying SVG DOM
  const controllers = {
    a1Mode: null,
    a1Dag: null,
    a2Panel: null,
    a3Morph: null,
    a2a3Mode: null,
    a4Mode: null,
    fig11: null,
    a5a7Mode: null,
    costChart: null,
    a8Flow: null
  };

  function pulseText(node, nextText) {
    if (!node || !nextText || node.textContent === nextText) return;
    const hadText = Boolean(node.dataset.hydrated);
    node.textContent = nextText;
    node.dataset.hydrated = "1";
    if (!hadText) return;
    node.classList.remove("stage-text-swap");
    void node.offsetWidth;
    node.classList.add("stage-text-swap");
  }

  function setHeader(prefix, title, caption, foot) {
    pulseText(document.getElementById(`stage-${prefix}-title`), title);
    pulseText(document.getElementById(`stage-${prefix}-caption`), caption);
    pulseText(document.getElementById(`stage-${prefix}-foot`), foot);
  }

  const lastSteps = {};
  let currentDir = "next";

  // Smooth directional Cross-fade + Scroll-Lift when switching between distinct visuals inside a chapter.
  // Keeps persistent SVG controllers intact whenever the sceneKey stays the same.
  function swapStage(canvas, sceneKey, renderFn) {
    if (!canvas) return null;
    const existing = canvas.querySelector(":scope > .stage-layer.is-active");
    if (canvas.dataset.sceneKey === sceneKey && existing) {
      return existing;
    }

    const isFirstPaint = !canvas.dataset.sceneKey;
    canvas.dataset.sceneKey = sceneKey;

    if (canvas._leaveTimer) {
      clearTimeout(canvas._leaveTimer);
      canvas._leaveTimer = null;
    }
    canvas.querySelectorAll(":scope > .stage-layer.is-leaving").forEach((n) => n.remove());

    if (existing && !isFirstPaint) {
      existing.classList.remove("is-active", "is-entering");
      existing.dataset.dir = currentDir;
      existing.classList.add("is-leaving");
      existing.setAttribute("aria-hidden", "true");
      // Strip HTML ids on leaving ghost so queries never collide with the incoming layer
      existing.querySelectorAll("[id]").forEach((el) => {
        if (!el.closest("defs")) el.removeAttribute("id");
      });
      canvas._leaveTimer = setTimeout(() => {
        existing.remove();
        canvas._leaveTimer = null;
      }, 390);
    } else if (existing) {
      existing.remove();
    }

    const nextLayer = document.createElement("div");
    nextLayer.className = isFirstPaint ? "stage-layer is-active" : "stage-layer is-active is-entering";
    if (!isFirstPaint) nextLayer.dataset.dir = currentDir;
    canvas.appendChild(nextLayer);
    if (renderFn) renderFn(nextLayer);
    return nextLayer;
  }

  /* ---- CTX (Beats 1–2) ---- */
  function renderCtx(step) {
    const canvas = document.getElementById("stage-ctx-canvas");
    if (!canvas) return;

    if (step === 0) {
      setHeader(
        "ctx",
        "El problema y las tres brechas frente al estado del arte",
        "Tabla I · Ejemplo real inst_04783",
        "EvidenCite es el único banco con 9 clases, paridad 1:1 Config A/B y 100% de test revisado manualmente."
      );
      swapStage(canvas, "ctx-0", (layer) => {
        layer.innerHTML = `
        <div class="editorial-panel is-ctx-merged">
          <div class="quote-showcase">
            <div class="quote-meta">ARTÍCULO CITANTE (2212.08018) · VENTANA m = 3 ORACIONES</div>
            <p class="quote-text">
              "The procedure builds upon the recent work by Hopkins, Kamath and Majid
              <span class="cite-mark">&lt;cite&gt;Hopkins, Kamath &amp; Majid, 2022&lt;/cite&gt;</span>
              on pure DP mean estimation. First, we exploit their algorithm to precondition the unknown covariance matrix…
              uses the result of <span class="othercit-mark">[OTHERCIT]</span> to improve the conditioning of Σ."
            </p>
          </div>
          <div class="pi-grid">
            <div class="pi-item">
              <div class="pi-code">BRECHA 1 · GRANULARIDAD</div>
              <h4 class="pi-title">3–7 clases gruesas</h4>
              <p class="pi-desc">Colapsan funciones cercanas (<em>Basis</em>, <em>Application</em>, <em>Improvement</em>).</p>
            </div>
            <div class="pi-item">
              <div class="pi-code">BRECHA 2 · EVIDENCIA</div>
              <h4 class="pi-title">Solo título y resumen</h4>
              <p class="pi-desc">Oculta el pasaje técnico interno del citado.</p>
            </div>
            <div class="pi-item">
              <div class="pi-code">BRECHA 3 · RUIDO</div>
              <h4 class="pi-title">Sesgo LLMs 7B–8B</h4>
              <p class="pi-desc">Sin test 100% Gold distorsionan evaluación.</p>
            </div>
          </div>
          <table class="sota-table">
            <thead>
              <tr>
                <th>Referencia / Sistema</th>
                <th>Clases</th>
                <th>Evidencia del citado</th>
                <th>Test Gold 9 cl.</th>
                <th>Calibración y costo</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>SciCite / ACL-ARC</strong> (Cohan; Jurgens)</td>
                <td class="mono">3 / 6</td>
                <td>Ninguna (solo citante)</td>
                <td><span class="check-no">—</span></td>
                <td><span class="check-no">—</span></td>
              </tr>
              <tr>
                <td><strong>MultiCite</strong> (Lauscher et al., 2022)</td>
                <td class="mono">7</td>
                <td>Ninguna en CFC (F1 0,00–0,04)</td>
                <td><span class="check-no">Parcial (7 cl.)</span></td>
                <td><span class="check-no">—</span></td>
              </tr>
              <tr>
                <td><strong>Gu et al. / SPECTER2</strong> (2022–2023)</td>
                <td class="mono">Ranking</td>
                <td>Global (Título + Resumen)</td>
                <td><span class="check-no">—</span></td>
                <td><span class="check-no">—</span></td>
              </tr>
              <tr>
                <td><strong>CiteFusion</strong> (Paolini et al., 2025)</td>
                <td class="mono">3 / 6</td>
                <td>Ninguna (encabezado citante)</td>
                <td><span class="check-no">—</span></td>
                <td><span class="check-no">—</span></td>
              </tr>
              <tr class="is-ours">
                <td>EvidenCite (Este trabajo)</td>
                <td class="mono">9 (Δ⁸)</td>
                <td>Paridad 1:1 Config A vs. B (Top-3 + sec.)</td>
                <td><span class="check-yes">✓ 2.798 Gold</span></td>
                <td><span class="check-yes">✓ ECE · Costo</span></td>
              </tr>
            </tbody>
          </table>
        </div>`;
      });
    } else {
      setHeader(
        "ctx",
        "Preguntas de investigación (PI1–PI3) y hoja de ruta",
        "Diseño experimental en 8 actividades",
        "Al cierre de la sustentación volveremos a estas tres tarjetas con su veredicto empírico."
      );
      swapStage(canvas, "ctx-1", (layer) => {
        layer.innerHTML = `
        <div class="editorial-panel">
          <div class="pi-grid">
            <div class="pi-item">
              <div class="pi-code">PI1 · EVIDENCIA LOCAL</div>
              <h4 class="pi-title">¿Cuánto aporta la evidencia local (Config B)?</h4>
              <p class="pi-desc">¿En qué medida y en qué secciones del citante los pasajes locales superan al título y resumen global?</p>
            </div>
            <div class="pi-item">
              <div class="pi-code">PI2 · ETIQUETADO Y TECHO</div>
              <h4 class="pi-title">¿Qué limita al fine-tuning?</h4>
              <p class="pi-desc">¿Cómo condiciona el ruido automático a A4 y cómo cambia la comparación entre familias tras A3?</p>
            </div>
            <div class="pi-item">
              <div class="pi-code">PI3 · PRECISIÓN Y COSTO</div>
              <h4 class="pi-title">¿Cuál es el mejor balance?</h4>
              <p class="pi-desc">¿Qué compromiso existe entre Macro-F1, calibración probabilística, latencia y costo en A4 vs. A5?</p>
            </div>
          </div>
          <div class="roadmap-strip">
            <div class="roadmap-node"><strong>A1</strong><span>Datos 1:1</span></div>
            <div class="roadmap-node"><strong>A2</strong><span>Panel 7B–8B</span></div>
            <div class="roadmap-node"><strong>A3</strong><span>Gold manual</span></div>
            <div class="roadmap-node"><strong>A4</strong><span>Fine-tuning</span></div>
            <div class="roadmap-node"><strong>A5</strong><span>LLMs API</span></div>
            <div class="roadmap-node"><strong>A6</strong><span>Calibración</span></div>
            <div class="roadmap-node"><strong>A7</strong><span>Costos</span></div>
            <div class="roadmap-node"><strong>A8</strong><span>AWS + Demo</span></div>
          </div>
        </div>`;
      });
    }
  }

  /* ---- A1 (Beats 3–6) ---- */
  let a1RevealTimer = null;

  function buildA1AB(layer) {
    const ex = D.inst04783;
    layer.innerHTML = `
      <div class="ab-container">
        <div class="ab-citing">
          <div class="ab-label-row">
            <span>VENTANA CITANTE FIJA · <strong class="mono">${ex.id}</strong> (${ex.citingId})</span>
            <span class="ab-pill gold">Gold A3: ${ex.goldLabel}</span>
          </div>
          <p>
            "The procedure builds upon the recent work by Hopkins, Kamath and Majid
            <span class="cite-mark">&lt;cite&gt;Samuel B Hopkins, Gautam Kamath, and Mahbod Majid, 2022&lt;/cite&gt;</span>
            on pure DP mean estimation. First, we exploit their algorithm to precondition the unknown covariance matrix such that it is approximately identity.
            A key step uses the result of <span class="othercit-mark">[OTHERCIT]</span> to improve the conditioning of Σ."
          </p>
        </div>

        <div class="ab-a-block">
          <div class="ab-label-row">
            <span>EVIDENCIA DEL CITADO (${ex.citedId})</span>
            <span class="ab-pill cfg-a">Config A · Vista Global (Título + Resumen)</span>
          </div>
          <div class="ab-global-box">
            <h4 class="ab-global-title">${ex.citedTitle}</h4>
            <p class="ab-global-abs">"${ex.citedAbstract}"</p>
          </div>
        </div>

        <div class="ab-b-staged" id="ab-b-staged">
          <div class="ab-label-row">
            <span>MISMA CITA · PARIDAD 1:1 (RANKING TOP-3)</span>
            <span class="ab-pill cfg-b">Config B · Top-3 Chunks ≤110 pal. + Encabezados Autorales</span>
          </div>
          <div class="ab-chunks">
            ${ex.chunks.map((ch) => `
              <div class="ab-chunk">
                <div class="ab-chunk-head">
                  <span class="ab-chunk-sec">[${ch.heading}]</span>
                  <span class="ab-chunk-score">R${ch.rank} · cos ${ch.score}</span>
                </div>
                <p>"${ch.text}"</p>
              </div>`).join("")}
          </div>
        </div>
      </div>`;
  }

  function renderA1(step) {
    const canvas = document.getElementById("stage-a1-canvas");
    if (!canvas) return;

    if (a1RevealTimer) {
      clearTimeout(a1RevealTimer);
      a1RevealTimer = null;
    }

    if (step === 0) {
      const layer = swapStage(canvas, "a1-ab", (el) => {
        controllers.a1Mode = "ab";
        controllers.a1Dag = null;
        buildA1AB(el);
      });
      setHeader(
        "a1",
        "Paridad 1:1 — Config A (global) vs. Config B (Top-3 local)",
        "Fig. 1 · inst_04783 · cosenos 0,894 / 0,889 / 0,888",
        "Misma ventana citante fija (m = 3); Config B aparece debajo con pasajes ≤110 pal. + encabezados autorales."
      );
      const bBlock = layer && layer.querySelector("#ab-b-staged");
      if (bBlock) {
        bBlock.classList.remove("is-revealed");
        // Timed entrance underneath Config A so the audience reads Config A first (~1,35s)
        a1RevealTimer = setTimeout(() => {
          bBlock.classList.add("is-revealed");
          a1RevealTimer = null;
        }, 1350);
      }
      return;
    }

    swapStage(canvas, "a1-dag", (el) => {
      controllers.a1Mode = "dag";
      controllers.a1Dag = C.createMedallionDAG(el);
    });
    const dagStage = step - 1; // 0, 1, 2
    controllers.a1Dag.update(dagStage);

    if (dagStage === 0) {
      setHeader("a1", "Construcción del Dataset — Capas 1–2: Ingesta y rescate seminal", "Fig. 2 · MultiCite + unarXive 2024", "Rescate Top-250 apoyado en caché local y APIs estructuró 66.314 artículos y ~145.000 pares (3.629 Gold 1:1).");
    } else if (dagStage === 1) {
      setHeader("a1", "Zoom Capa 3 — Recuperación SciBERT (512 tok) y NLI calibrado", "Fig. 3 · Consulta sin encabezados + DeBERTa-v3", "Presupuesto exacto de 512 tokens y filtrado NLI para llenar cuotas por clase cuidando calidad.");
    } else {
      setHeader("a1", "Capas 4–5 — 18.658 pares 1:1, fuga nula y ensamblado O(1)", "Tabla II · 8.336 citantes disjuntos", "Split 70/15/15 agrupado por citing_paper_id (leakage = 0) con acceso O(1) por instancia para A4/A5.");
    }
  }

  /* ---- A2 & A3 (Beats 9–14) ---- */
  function renderA2A3(step) {
    const canvas = document.getElementById("stage-a2a3-canvas");
    if (!canvas) return;

    if (step <= 1) {
      swapStage(canvas, "a2-panel", (el) => {
        controllers.a2a3Mode = "a2panel";
        controllers.a2Panel = C.createA2Panel(el, D.a2Panel);
      });
      controllers.a2Panel.update(step === 1);
      setHeader(
        "a2a3",
        step === 0 ? "Panel de 3 jueces abiertos en vLLM (A2)" : "Auditoría A2: baja concordancia y 36,4% en Gold",
        "Llama-3.1-8B · Qwen2.5-7B · Mistral-7B",
        step === 0 ? "Ejecución en A100-40GB, temperatura 0, cegados al artículo citado." : "El sesgo de sobre-atribución metodológica impidió confiar en la mayoría simple."
      );
    } else if (step === 2) {
      swapStage(canvas, "a2-strata", (el) => {
        controllers.a2a3Mode = "a2strata";
        C.renderA2Strata(el, D.a2Panel);
      });
      setHeader("a2a3", "Balance del corpus tras la regla conservadora (Ec. 3)", "18.658 instancias · Casos A–F", "El 40,7% (7.590 citas) quedó sin consenso del panel y entró con etiqueta débil a train.jsonl.");
    } else if (step === 3) {
      swapStage(canvas, "a3-wf", (el) => {
        controllers.a2a3Mode = "a3wf";
        C.renderA3Waterfall(el, D.a3Waterfall);
      });
      setHeader("a2a3", "Curaduría humana doble ciega del 100% del test (A3)", "N = 2.798 citas · κ = 0,509", "Javier y Julián revisaron de forma independiente las 2.798 citas; 68% directo + 32% desempate acotado.");
    } else {
      swapStage(canvas, "a3-morph", (el) => {
        controllers.a2a3Mode = "a3morph";
        controllers.a3Morph = C.createA3Morph(el, D.a3Distribution);
      });
      const isGold = step === 5;
      controllers.a3Morph.update(isGold);
      setHeader(
        "a2a3",
        isGold ? "Distribución Gold humano A3: giro de 54,32% en el test" : "Distribución Pre-A3: cuotas artificialmente balanceadas",
        "Tabla II · Evolución sobre N = 2.798",
        isGold ? "Background pasó de 12,4% a 42,89% y Gap bajó de 11,7% a 2,61% (línea punteada = Pre-A3)." : "Antes de A3, el conjunto de prueba reproducía las cuotas casi uniformes del minado."
      );
    }
  }

  /* ---- A4 (Beats 15–19) ---- */
  function renderA4(step) {
    const canvas = document.getElementById("stage-a4-canvas");
    if (!canvas) return;

    if (step === 0) {
      swapStage(canvas, "a4-r2", (el) => {
        controllers.a4Mode = "r2";
        C.renderA4R2(el, D.a4R2Models);
      });
      setHeader("a4", "Embudo de 92 corridas y banda plana en R2 (~0,74)", "Fig. 10b · 10 modelos 2B–4B × 3 semillas", "Ni tamaño (|ρ| < 0,2) ni arquitectura rompen el techo automático; Granite-4.2 se elige para escalar a 8B.");
    } else if (step === 1) {
      swapStage(canvas, "a4-collapse", (el) => {
        controllers.a4Mode = "collapse";
        C.renderA4Collapse(el, D.a4Definitives);
      });
      setHeader("a4", "Desplome de A4 al pasar del test automático al Gold humano", "Fig. 4 · Tabla III (N = 2.798)", "Escalar 70× (110M → 8B) no evita la caída a 0,43–0,46 Macro-F1; Config B empeora o es neutra.");
    } else {
      swapStage(canvas, "a4-fig11", (el) => {
        controllers.a4Mode = "fig11";
        controllers.fig11 = C.createFig11ThreeStep(el, D.fig11Diagnosis);
      });
      const subStep = step - 2; // 0, 1, 2
      controllers.fig11.update(subStep);
      const titles = [
        "Fig. 11 (Paso 1/3) — En citas confirmadas, el fine-tuning acierta 83–88%",
        "Fig. 11 (Paso 2/3) — En citas corregidas, A4 repite la etiqueta vieja (○)",
        "Fig. 11 (Paso 3/3) — Divergencia causal: fine-tuning cae (●), APIs aciertan"
      ];
      const foots = [
        "Confirmadas (n = 1.278): Granite-8B (87,1%) iguala o supera a Gemini (82,8%).",
        "Corregidas (n = 1.520): los modelos con fine-tuning repiten la etiqueta descartada 62,9–64,5% de las veces.",
        "Al deslizar a la etiqueta humana Gold (●), A4 cae −45,8 a −49,7 pp mientras Gemini salta +36,8 pp."
      ];
      setHeader("a4", titles[subStep], "Fig. 11 (Anexo A) · N = 2.798", foots[subStep]);
    }
  }

  /* ---- A5 / A6 / A7 (Beats 20–26) ---- */
  function renderA5A7(step) {
    const canvas = document.getElementById("stage-a5a7-canvas");
    if (!canvas) return;

    if (step === 0) {
      swapStage(canvas, "a5-parity", (el) => {
        controllers.a5a7Mode = "parity";
        C.renderA5ParityBreak(el, D.a5ParityBreak);
      });
      setHeader("a5a7", "Clímax #2 (Paso 1/2) — Empate aparente vs. ventaja real en Gold", "22.384 inferencias API · 99,991% validez JSON", "Antes de A3 empataban (p = 0,436); sobre el Gold humano Gemini 0s-B abre +0,1331 Macro-F1.");
    } else if (step === 1) {
      swapStage(canvas, "a5-byclass", (el) => {
        controllers.a5a7Mode = "byclass";
        C.renderA5ByClass(el, D.a5ByClass);
      });
      setHeader("a5a7", "Clímax #2 (Paso 2/2) — Gemini 0s-B supera a GPT en las 9 clases", "Tabla IV · F1-Score por clase ordenadas por soporte", "Mayores brechas en Basis (+0,302), Further Reading (+0,246) y Gap (+0,173).");
    } else if (step === 2) {
      swapStage(canvas, "a5-sections", (el) => {
        controllers.a5a7Mode = "sections";
        C.renderA5MacroSections(el, D.a5MacroSections);
      });
      setHeader("a5a7", "Efecto de Config B por macro-sección citante (0-Shot)", "Fig. 6 · Respuesta a PI1", "GPT gana +10,28 pp en Related Work y +4,71 pp en Intro/Prelim; Gemini reduce −19,4% sus pensamientos.");
    } else if (step === 3) {
      swapStage(canvas, "a5-chips", (el) => {
        controllers.a5a7Mode = "chips";
        C.renderA5NineChips(el, D.a5NineShotChips);
      });
      setHeader("a5a7", "Diagnóstico de 9-Shot: 4 de 9 fichas contaminadas", "seed = 42 en train.jsonl · Tabla VI", "Cuatro ejemplares provinieron de empates 1-1-1 donde el error previo en test fue 66,06% (288/436).");
    } else if (step === 4) {
      swapStage(canvas, "a6-board", (el) => {
        controllers.a5a7Mode = "board";
        C.renderA6GlobalBoard(el, D.a6GlobalBoard);
      });
      setHeader("a5a7", "Tablero global A6: las 14 variantes de Tabla III (Config A vs. B)", "A4 locales (6) · GPT (4) · Gemini (4) · Gold N = 2.798", "Verde/rojo en el conector indica si Config B mejora o cae; curvas de fiabilidad detalladas disponibles en Q&A (tecla Q).");
    } else {
      swapStage(canvas, "a7-cost", (el) => {
        controllers.a5a7Mode = "cost";
        controllers.costChart = C.createCostQualityChart(el, D.a7CostQuality);
      });
      const showOpt = step === 6;
      controllers.costChart.update(showOpt);
      setHeader(
        "a5a7",
        showOpt ? "Mejor relación precisión–costo a escala (10.000 citas)" : "Zona descartada: todas las variantes 9-Shot rinden menos y cuestan más",
        "Fig. 7 · Tabla III · Costo neto con Prompt Caching",
        showOpt ? "Dos opciones racionales: GPT 0s-B ($6,59/10k, ~1 s) o Gemini 0s-B ($17,68/10k, 0,600 F1) (frontera de Pareto)." : "Las 4 variantes 9-Shot cuestan entre $8,23 y $53,09 por 10k citas sin superar a 0-Shot."
      );
    }
  }

  /* ---- A8 (Beats 27–29: Flujo continuo por scroll) ---- */
  function renderA8(step) {
    const canvas = document.getElementById("stage-a8-canvas");
    if (!canvas) return;
    if (!controllers.a8Flow) {
      controllers.a8Flow = C.createA8ScrollFlow(canvas);
    }
    controllers.a8Flow.update(step);
    const titles = [
      "A8 (Paso 1/3) — Entrada única: cita y artículo citado hacia EvidenCite",
      "A8 (Paso 2/3) — Construcción paralela: Config A (global) vs. Config B (Top-3)",
      "A8 (Paso 3/3) — Dos clasificadores independientes y distribuciones en Δ⁸"
    ];
    const captions = [
      "Ejemplo real · inst_04783 · Normalización y resolución",
      "Ejemplo real · inst_04783 · SciBERT 768-dim + cosenos reales (0,894 / 0,889 / 0,888)",
      "Esquema ilustrativo en Δ⁸ (9 funciones) · Predicción real en la demo"
    ];
    const foots = [
      "Una misma ventana citante fija (m = 3) y un artículo citado entran al motor EvidenCite.",
      "Arriba: título y resumen global. Abajo: segmentación ≤110 pal., embeddings SciBERT y Top-3 por coseno.",
      "Dos inferencias independientes (A y B) producen sus distribuciones sobre las 9 funciones retóricas."
    ];
    const idx = Math.max(0, Math.min(2, step || 0));
    setHeader("a8", titles[idx], captions[idx], foots[idx]);
  }

  function activateScene(sceneName, step) {
    const prev = lastSteps[sceneName];
    currentDir = prev !== undefined && step < prev ? "prev" : "next";
    lastSteps[sceneName] = step;

    if (sceneName === "ctx") renderCtx(step);
    else if (sceneName === "a1") renderA1(step);
    else if (sceneName === "a2a3") renderA2A3(step);
    else if (sceneName === "a4") renderA4(step);
    else if (sceneName === "a5a7") renderA5A7(step);
    else if (sceneName === "a8") renderA8(step);
  }

  function initAllDefaults() {
    renderCtx(0);
    renderA1(0);
    renderA2A3(0);
    renderA4(0);
    renderA5A7(0);
    renderA8(0);
  }

  return { activateScene, initAllDefaults };
})();
