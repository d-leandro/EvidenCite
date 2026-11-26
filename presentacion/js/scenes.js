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
    costChart: null
  };

  function setHeader(prefix, title, caption, foot) {
    const t = document.getElementById(`stage-${prefix}-title`);
    const c = document.getElementById(`stage-${prefix}-caption`);
    const f = document.getElementById(`stage-${prefix}-foot`);
    if (t && title) t.textContent = title;
    if (c && caption) c.textContent = caption;
    if (f && foot) f.textContent = foot;
  }

  /* ---- CTX (Beats 1–3) ---- */
  function renderCtx(step) {
    const canvas = document.getElementById("stage-ctx-canvas");
    if (!canvas) return;

    if (step === 0) {
      setHeader(
        "ctx",
        "¿Por qué falla mirar solo el título y el resumen del citado?",
        "Ejemplo real · inst_04783",
        "La taxonomía de Lyu et al. (2021) distingue 9 funciones retóricas no evaluativas en Δ⁸."
      );
      canvas.innerHTML = `
        <div class="editorial-panel">
          <div class="quote-showcase">
            <div class="quote-meta">ARTÍCULO CITANTE (2212.08018) · VENTANA m = 3 ORACIONES</div>
            <p class="quote-text">
              "The procedure builds upon the recent work by Hopkins, Kamath and Majid
              <span class="cite-mark">&lt;cite&gt;Hopkins, Kamath &amp; Majid, 2022&lt;/cite&gt;</span>
              on pure DP mean estimation. First, we exploit their algorithm to precondition the unknown covariance matrix...
              uses the result of <span class="othercit-mark">[OTHERCIT]</span> to improve the conditioning of Σ."
            </p>
          </div>
          <div class="pi-grid" style="margin-bottom:0;">
            <div class="pi-item">
              <div class="pi-code">BRECHA 1 · GRANULARIDAD</div>
              <h4 class="pi-title">3–6 clases gruesas</h4>
              <p class="pi-desc">SciCite y ACL-ARC colapsan funciones metodológicas cercanas (<em>Basis</em>, <em>Application</em>, <em>Improvement</em>).</p>
            </div>
            <div class="pi-item">
              <div class="pi-code">BRECHA 2 · EVIDENCIA ESTÁTICA</div>
              <h4 class="pi-title">Solo título y resumen</h4>
              <p class="pi-desc">Config A entrega el mismo resumen global para toda mención y oculta algoritmos o pruebas internas.</p>
            </div>
            <div class="pi-item">
              <div class="pi-code">BRECHA 3 · RUIDO DE ETIQUETAS</div>
              <h4 class="pi-title">Sesgo en LLMs 7B–8B</h4>
              <p class="pi-desc">Etiquetadores abiertos sobre-atribuyen clases metodológicas y distorsionan la evaluación.</p>
            </div>
          </div>
        </div>`;
    } else if (step === 1) {
      setHeader(
        "ctx",
        "Posicionamiento frente al estado del arte",
        "Síntesis de Tabla I · 4 referencias ancla",
        "EvidenCite es el único banco que reúne 9 clases, paridad 1:1 Config A/B y 100% de test revisado manualmente."
      );
      canvas.innerHTML = `
        <div class="editorial-panel">
          <table class="sota-table">
            <thead>
              <tr>
                <th>Referencia / Sistema</th>
                <th>Clases</th>
                <th>Evidencia del citado</th>
                <th>Test Gold 9 clases</th>
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
                <td>Ninguna en CFC (F1 0,00–0,04 extractivo)</td>
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
    } else {
      setHeader(
        "ctx",
        "Preguntas de investigación (PI1–PI3) y hoja de ruta",
        "Diseño experimental en 8 actividades",
        "Al cierre de la sustentación volveremos a estas tres tarjetas con su veredicto empírico."
      );
      canvas.innerHTML = `
        <div class="editorial-panel">
          <div class="pi-grid">
            <div class="pi-item">
              <div class="pi-code">PI1 · EVIDENCIA LOCAL</div>
              <h4 class="pi-title">¿Cuándo ayuda Config B?</h4>
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
    }
  }

  /* ---- A1 (Beats 4–8) ---- */
  function ensureA1AB(canvas) {
    if (controllers.a1Mode === "ab") return;
    controllers.a1Mode = "ab";
    controllers.a1Dag = null;
    const ex = D.inst04783;
    canvas.innerHTML = `
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

        <div class="ab-evidence-wrap">
          <div class="ab-view is-visible" id="ab-view-a">
            <div class="ab-label-row">
              <span>EVIDENCIA DEL CITADO (${ex.citedId})</span>
              <span class="ab-pill cfg-a">Config A · Vista Global (Título + Resumen)</span>
            </div>
            <div class="ab-global-box">
              <h4 class="ab-global-title">${ex.citedTitle}</h4>
              <p class="ab-global-abs">"${ex.citedAbstract}"</p>
            </div>
          </div>

          <div class="ab-view is-hidden" id="ab-view-b">
            <div class="ab-label-row">
              <span>EVIDENCIA DEL CITADO (${ex.citedId}) · PARIDAD 1:1</span>
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
        </div>
      </div>`;
  }

  function renderA1(step) {
    const canvas = document.getElementById("stage-a1-canvas");
    if (!canvas) return;

    if (step <= 1) {
      ensureA1AB(canvas);
      const viewA = document.getElementById("ab-view-a");
      const viewB = document.getElementById("ab-view-b");
      if (step === 0) {
        setHeader("a1", "Paridad 1:1 — Configuración A (Global)", "Fig. 1 · inst_04783", "Misma ventana citante fija (m = 3 oraciones); varía únicamente la evidencia del artículo citado.");
        viewA.className = "ab-view is-visible";
        viewB.className = "ab-view is-hidden";
      } else {
        setHeader("a1", "Paridad 1:1 — Transformación a Configuración B (Local Top-3)", "Fig. 1 · Encabezados autorales preservados", "Encabezados exactos (Nearest Enclosing Scope) sin colapsar a IMRaD · cosenos 0,894 / 0,889 / 0,888.");
        viewA.className = "ab-view is-hidden";
        viewB.className = "ab-view is-visible";
      }
      return;
    }

    if (controllers.a1Mode !== "dag") {
      controllers.a1Mode = "dag";
      controllers.a1Dag = C.createMedallionDAG(canvas);
    }
    const dagStage = step - 2; // 0, 1, 2
    controllers.a1Dag.update(dagStage);

    if (dagStage === 0) {
      setHeader("a1", "DAG vertical Medallion — Capas 1–2: Ingesta y rescate seminal", "Fig. 2 · MultiCite + unarXive", "Rescate Top-250 apoyado en caché local y APIs elevó la cobertura Gold de 0,4% a 3.629 pares 1:1.");
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
      if (controllers.a2a3Mode !== "a2panel") {
        controllers.a2a3Mode = "a2panel";
        controllers.a2Panel = C.createA2Panel(canvas, D.a2Panel);
      }
      controllers.a2Panel.update(step === 1);
      setHeader(
        "a2a3",
        step === 0 ? "Panel de 3 jueces abiertos en vLLM (A2)" : "Auditoría A2: baja concordancia y 36,4% en Gold",
        "Llama-3.1-8B · Qwen2.5-7B · Mistral-7B",
        step === 0 ? "Ejecución en A100-40GB, temperatura 0, cegados al artículo citado." : "El sesgo de sobre-atribución metodológica impidió confiar en la mayoría simple."
      );
    } else if (step === 2) {
      controllers.a2a3Mode = "a2strata";
      C.renderA2Strata(canvas, D.a2Panel);
      setHeader("a2a3", "Balance del corpus tras la regla conservadora (Ec. 3)", "18.658 instancias · Casos A–F", "El 40,7% (7.590 citas) quedó sin consenso del panel y entró con etiqueta débil a train.jsonl.");
    } else if (step === 3) {
      controllers.a2a3Mode = "a3wf";
      C.renderA3Waterfall(canvas, D.a3Waterfall);
      setHeader("a2a3", "Curaduría humana doble ciega del 100% del test (A3)", "N = 2.798 citas · κ = 0,509", "Javier y Julián revisaron de forma independiente las 2.798 citas; 68% directo + 32% desempate acotado.");
    } else {
      if (controllers.a2a3Mode !== "a3morph") {
        controllers.a2a3Mode = "a3morph";
        controllers.a3Morph = C.createA3Morph(canvas, D.a3Distribution);
      }
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
      controllers.a4Mode = "r2";
      C.renderA4R2(canvas, D.a4R2Models);
      setHeader("a4", "Embudo de 92 corridas y banda plana en R2 (~0,74)", "Fig. 10b · 10 modelos 2B–4B × 3 semillas", "Ni tamaño (|ρ| < 0,2) ni arquitectura rompen el techo automático; Granite-4.2 se elige para escalar a 8B.");
    } else if (step === 1) {
      controllers.a4Mode = "collapse";
      C.renderA4Collapse(canvas, D.a4Definitives);
      setHeader("a4", "Desplome de A4 al pasar del test automático al Gold humano", "Fig. 4 · Tabla III (N = 2.798)", "Escalar 70× (110M → 8B) no evita la caída a 0,43–0,46 Macro-F1; Config B empeora o es neutra.");
    } else {
      if (controllers.a4Mode !== "fig11") {
        controllers.a4Mode = "fig11";
        controllers.fig11 = C.createFig11ThreeStep(canvas, D.fig11Diagnosis);
      }
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
      controllers.a5a7Mode = "parity";
      C.renderA5ParityBreak(canvas, D.a5ParityBreak);
      setHeader("a5a7", "Clímax #2 (Paso 1/2) — Empate aparente vs. ventaja real en Gold", "22.384 inferencias API · 99,991% validez JSON", "Antes de A3 empataban (p = 0,436); sobre el Gold humano Gemini 0s-B abre +0,1331 Macro-F1.");
    } else if (step === 1) {
      controllers.a5a7Mode = "byclass";
      C.renderA5ByClass(canvas, D.a5ByClass);
      setHeader("a5a7", "Clímax #2 (Paso 2/2) — Gemini 0s-B supera a GPT en las 9 clases", "Tabla IV · F1-Score por clase ordenadas por soporte", "Mayores brechas en Basis (+0,302), Further Reading (+0,246) y Gap (+0,173).");
    } else if (step === 2) {
      controllers.a5a7Mode = "sections";
      C.renderA5MacroSections(canvas, D.a5MacroSections);
      setHeader("a5a7", "Efecto de Config B por macro-sección citante (0-Shot)", "Fig. 6 · Respuesta a PI1", "GPT gana +10,28 pp en Related Work y +4,71 pp en Intro/Prelim; Gemini reduce −19,4% sus pensamientos.");
    } else if (step === 3) {
      controllers.a5a7Mode = "chips";
      C.renderA5NineChips(canvas, D.a5NineShotChips);
      setHeader("a5a7", "Diagnóstico de 9-Shot: 4 de 9 fichas contaminadas", "seed = 42 en train.jsonl · Tabla VI", "Cuatro ejemplares provinieron de empates 1-1-1 donde el error previo en test fue 66,06% (288/436).");
    } else if (step === 4) {
      controllers.a5a7Mode = "board";
      C.renderA6GlobalBoard(canvas, D.a6GlobalBoard);
      setHeader("a5a7", "Tablero global A6: las 14 variantes de Tabla III (Config A vs. B)", "A4 locales (6) · GPT (4) · Gemini (4) · Gold N = 2.798", "Verde/rojo en el conector indica si Config B mejora o cae; curvas de fiabilidad detalladas disponibles en Q&A (tecla Q).");
    } else {
      if (controllers.a5a7Mode !== "cost") {
        controllers.a5a7Mode = "cost";
        controllers.costChart = C.createCostQualityChart(canvas, D.a7CostQuality);
      }
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

  /* ---- A8 (Beat 27) ---- */
  let a8Controller = null;
  function renderA8() {
    const canvas = document.getElementById("stage-a8-canvas");
    if (canvas && !a8Controller) a8Controller = C.renderA8ConfigNarrative(canvas);
  }
  function advanceA8(dir) { return a8Controller ? a8Controller.advance(dir) : false; }
  function resetA8(step=0) { renderA8(); if(a8Controller) a8Controller.show(step); }

  function activateScene(sceneName, step) {
    if (sceneName === "ctx") renderCtx(step);
    else if (sceneName === "a1") renderA1(step);
    else if (sceneName === "a2a3") renderA2A3(step);
    else if (sceneName === "a4") renderA4(step);
    else if (sceneName === "a5a7") renderA5A7(step);
    else if (sceneName === "a8") renderA8();
  }

  function initAllDefaults() {
    renderCtx(0);
    renderA1(0);
    renderA2A3(0);
    renderA4(0);
    renderA5A7(0);
  }

  return { activateScene, initAllDefaults, advanceA8, resetA8 };
})();
