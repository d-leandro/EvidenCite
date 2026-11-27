/* ==========================================================================
   EvidenCite — Datos Auditados (Fuente Única: informe_final_ieee.tex + repo)
   100% embebido para operación offline directa (file:// y http.server)
   ========================================================================== */

window.EVIDENCITE_DATA = {
  meta: {
    title: "EvidenCite: Recomendación local de citas y clasificación de su función retórica en artículos académicos",
    authors: "Daniel R. Gallo Acevedo · Dubán L. Jiménez Puerta · Julián A. Malaver Montoya · Javier M. Torres Ortega",
    program: "Maestría en Inteligencia Artificial (MAIA) · Universidad de los Andes",
    totalMinutes: 15,
    blocks: [
      { id: "portada", phase: "CTX", label: "Portada", speaker: "Leandro", targetSec: 0, startBeat: 0, endBeat: 0 },
      { id: "b1", phase: "A1", label: "Contexto + A1", speaker: "Leandro", targetSec: 135, startBeat: 1, endBeat: 6 },
      { id: "b2", phase: "A2/A3", label: "Panel A2 + Giro A3", speaker: "Javier", targetSec: 150, startBeat: 7, endBeat: 12 },
      { id: "b3", phase: "A4", label: "Techo Supervisado A4", speaker: "Daniel", targetSec: 165, startBeat: 13, endBeat: 17 },
      { id: "b4", phase: "A5/A7", label: "LLMs API y Precisión–Costo", speaker: "Leandro", targetSec: 150, startBeat: 18, endBeat: 24 },
      { id: "b5", phase: "A8/CONC", label: "Cloud, Demo y Cierre", speaker: "Julián", targetSec: 300, startBeat: 25, endBeat: 30 }
    ]
  },

  inst04783: {
    id: "inst_04783",
    goldLabel: "Basis",
    citingId: "2212.08018",
    citingSection: "High-level overview",
    citedId: "2111.12981",
    citedTitle: "Efficient mean estimation with pure differential privacy via a sum-of-squares exponential mechanism",
    citedAbstract: "We give the first polynomial-time algorithm to estimate the mean of a d-variate probability distribution with bounded covariance from Õ(d) independent samples subject to pure differential privacy. Prior algorithms incur exponential running time or require Ω(d^1.5) samples...",
    chunks: [
      {
        rank: 1,
        score: "0,894",
        heading: "2.2.1 Private mean estimation when ||E X|| ≤ poly(d)",
        text: "A merit of our approach is that it allows powerful techniques from robust statistics to be used for private algorithm design: our algorithm draws heavily on a robust mean estimation algorithm..."
      },
      {
        rank: 2,
        score: "0,889",
        heading: "6 Fine Estimation",
        text: "These properties of the SDP relaxation, together with its computational efficiency, yield a computationally efficient algorithm for estimating the mean up to O(√(1/m)) error..."
      },
      {
        rank: 3,
        score: "0,888",
        heading: "6 Fine Estimation",
        text: "Our goal is to privately estimate the mean of this distribution up to accuracy α with probability 1−β. Moreover, suppose we have access to an initial estimate of the mean..."
      }
    ]
  },

  a2Panel: {
    judges: [
      { name: "Llama-3.1-8B-Instruct", params: "8B · vLLM", abst: "0,00%" },
      { name: "Qwen2.5-7B-Instruct", params: "7B · vLLM", abst: "0,00%" },
      { name: "Mistral-7B-Instruct-v0.3", params: "7B · vLLM", abst: "0,12% (23/18.658)" }
    ],
    kappaValid: 0.3939,
    goldAccuracy: 36.40,
    goldHits: "1.321 / 3.629",
    strata: [
      { label: "MultiCite Gold (Caso F)", count: 3629, pct: 19.45, colorVar: "--accent-gold", desc: "Etiquetas humanas originales preservadas sin sobrescritura." },
      { label: "Consenso Panel (Casos A + B)", count: 7439, pct: 39.87, colorVar: "--accent-primary", desc: "Unanimidad 3/3 (sobrescribe) o mayoría 2/3 confirmatoria." },
      { label: "Needs Review (Casos C + D)", count: 7590, pct: 40.68, colorVar: "--accent-danger", desc: "Mayoría divergente 2/3 o empate 1-1-1: retiene etiqueta débil previa y entra así a train.jsonl." }
    ]
  },

  a3Waterfall: {
    total: 2798,
    papers: 1258,
    kappa: 0.509,
    humanAgree: 1750,
    humanAgreePct: 62.5,
    disagreements: 1048,
    multiciteTie: 147,
    caseByCase: 7,
    humanDirectTotal: 1904,
    humanDirectPct: 68.0,
    llmTieBreak: 894,
    llmTieBreakPct: 32.0
  },

  // Tabla II — Evolución Pre-A3 vs. Test Gold A3 (N = 2.798)
  a3Distribution: [
    { cls: "Background",      preN: 348, prePct: 12.44, goldN: 1200, goldPct: 42.89, deltaPct: +30.45 },
    { cls: "Application",     preN: 453, prePct: 16.19, goldN: 637,  goldPct: 22.77, deltaPct: +6.58 },
    { cls: "Comparison",      preN: 493, prePct: 17.62, goldN: 298,  goldPct: 10.65, deltaPct: -6.97 },
    { cls: "Further Reading", preN: 251, prePct: 8.97,  goldN: 227,  goldPct: 8.11,  deltaPct: -0.86 },
    { cls: "Originator",      preN: 257, prePct: 9.19,  goldN: 105,  goldPct: 3.75,  deltaPct: -5.44 },
    { cls: "Evidence",        preN: 253, prePct: 9.04,  goldN: 100,  goldPct: 3.57,  deltaPct: -5.47 },
    { cls: "Basis",           preN: 199, prePct: 7.11,  goldN: 90,   goldPct: 3.22,  deltaPct: -3.89 },
    { cls: "Gap",             preN: 327, prePct: 11.69, goldN: 73,   goldPct: 2.61,  deltaPct: -9.08 },
    { cls: "Improvement",     preN: 217, prePct: 7.76,  goldN: 68,   goldPct: 2.43,  deltaPct: -5.33 }
  ],

  // A4-01 Embudo de 92 corridas (pestañas R1 · R2 · Definitivos · Generativo vs. clasificador).
  // Generado desde analisis/comparacion.csv y recursos.csv de los tres zips de analizar_resultados.ipynb
  // (comparacion_modelos, comparacion_modelos_v2, EvidenCite_A6_A7_Evaluacion_Comparativa). Memoria en GB = MB / 1024.
  a4Funnel: {
    // R1 (19 corridas, 1 semilla, iteración previa del etiquetado): Macro-F1 de test e IC95 por configuración
    r1: [
      { name: "Llama-3.2-3B", discarded: false, a: { f1: 0.8297, lo: 0.8154, hi: 0.843, min: 21.7, gb: 39.1, pct: 0.75 }, b: { f1: 0.8345, lo: 0.8204, hi: 0.8485, min: 28.2, gb: 39.1, pct: 0.75 } },
      { name: "Phi-4-mini", discarded: false, a: { f1: 0.8272, lo: 0.8126, hi: 0.8406, min: 22.8, gb: 38.1, pct: 0.6 }, b: { f1: 0.8295, lo: 0.8151, hi: 0.8433, min: 30.9, gb: 38.1, pct: 0.6 } },
      { name: "SmolLM3-3B", discarded: false, a: { f1: 0.827, lo: 0.8126, hi: 0.8412, min: 27.7, gb: 45.1, pct: 0.97 }, b: { f1: 0.8276, lo: 0.8137, hi: 0.8412, min: 30.9, gb: 45.1, pct: 0.97 } },
      { name: "Granite-4.1-3B", discarded: false, a: { f1: 0.8289, lo: 0.8146, hi: 0.8422, min: 24.5, gb: 48.7, pct: 0.91 }, b: { f1: 0.8222, lo: 0.8077, hi: 0.8355, min: 32.2, gb: 48.7, pct: 0.91 } },
      { name: "Qwen3.5-2B", discarded: false, a: { f1: 0.819, lo: 0.805, hi: 0.8331, min: 48.3, gb: 54.5, pct: 1.04 }, b: { f1: 0.825, lo: 0.8105, hi: 0.8389, min: 39.7, gb: 54.5, pct: 1.04 } },
      { name: "Qwen3-4B", discarded: false, a: { f1: 0.8269, lo: 0.8134, hi: 0.8404, min: 30.1, gb: 56.2, pct: 0.82 }, b: { f1: 0.8149, lo: 0.7997, hi: 0.8292, min: 40.2, gb: 56.2, pct: 0.82 } },
      { name: "Hunyuan-4B", discarded: true, a: { f1: 0.8106, lo: 0.796, hi: 0.8249, min: 29.2, gb: 56.3, pct: 0.76 }, b: { f1: 0.8168, lo: 0.8031, hi: 0.832, min: 40.0, gb: 56.3, pct: 0.76 } },
      { name: "Gemma-4-E4B", discarded: true, a: { f1: 0.8172, lo: 0.8033, hi: 0.8315, min: 49.6, gb: 73.1, pct: 0.63 }, b: { f1: 0.8076, lo: 0.7927, hi: 0.8222, min: 57.0, gb: 94.0, pct: 0.63 } },
      { name: "Gemma-4-E2B", discarded: true, a: { f1: 0.81, lo: 0.7949, hi: 0.8245, min: 34.6, gb: 48.6, pct: 0.74 }, b: { f1: 0.7971, lo: 0.7823, hi: 0.812, min: 37.3, gb: 62.1, pct: 0.74 } },
      { name: "Gemma-4-26B-A4B*", discarded: true, b: { f1: 0.785, lo: 0.7702, hi: 0.8001, min: 101.0, gb: 57.3, pct: 0.12 } }
    ],
    // R2 (58 corridas = 10 modelos × A/B × 3 semillas; faltan Nemotron B y Qwen3.5-2B B con la semilla 2255891522)
    r2: [
      { name: "SmolLM3-3B", candidate: false, granite: false, mean: 0.7456, min: 0.7421, max: 0.7492, runs: [{ cfg: "A", f1: 0.744 }, { cfg: "A", f1: 0.7448 }, { cfg: "A", f1: 0.7421 }, { cfg: "B", f1: 0.7472 }, { cfg: "B", f1: 0.7492 }, { cfg: "B", f1: 0.7465 }], top7: true },
      { name: "Llama-3.2-3B", candidate: true, granite: false, mean: 0.745, min: 0.7368, max: 0.7511, runs: [{ cfg: "A", f1: 0.7505 }, { cfg: "A", f1: 0.7406 }, { cfg: "A", f1: 0.7407 }, { cfg: "B", f1: 0.7511 }, { cfg: "B", f1: 0.7501 }, { cfg: "B", f1: 0.7368 }], top7: true },
      { name: "Qwen3.5-4B", candidate: true, granite: false, mean: 0.7432, min: 0.7363, max: 0.7516, runs: [{ cfg: "A", f1: 0.7516 }, { cfg: "A", f1: 0.7387 }, { cfg: "A", f1: 0.7363 }, { cfg: "B", f1: 0.7443 }, { cfg: "B", f1: 0.7496 }, { cfg: "B", f1: 0.739 }], top7: true },
      { name: "Qwen3-4B", candidate: false, granite: false, mean: 0.7429, min: 0.7347, max: 0.7504, runs: [{ cfg: "A", f1: 0.7468 }, { cfg: "A", f1: 0.7387 }, { cfg: "A", f1: 0.7389 }, { cfg: "B", f1: 0.7347 }, { cfg: "B", f1: 0.7504 }, { cfg: "B", f1: 0.7477 }], top7: true },
      { name: "Qwen3.5-2B", candidate: true, granite: false, mean: 0.7427, min: 0.7393, max: 0.7487, runs: [{ cfg: "A", f1: 0.7458 }, { cfg: "A", f1: 0.7396 }, { cfg: "A", f1: 0.7393 }, { cfg: "B", f1: 0.7487 }, { cfg: "B", f1: 0.7403 }], top7: true },
      { name: "Granite-4.1-3B", candidate: true, granite: true, mean: 0.7396, min: 0.7289, max: 0.7559, runs: [{ cfg: "A", f1: 0.7326 }, { cfg: "A", f1: 0.7334 }, { cfg: "A", f1: 0.7289 }, { cfg: "B", f1: 0.7413 }, { cfg: "B", f1: 0.7559 }, { cfg: "B", f1: 0.7456 }], top7: true },
      { name: "Granite-4.2-3B", candidate: true, granite: true, mean: 0.7384, min: 0.7303, max: 0.746, runs: [{ cfg: "A", f1: 0.7314 }, { cfg: "A", f1: 0.746 }, { cfg: "A", f1: 0.7457 }, { cfg: "B", f1: 0.7387 }, { cfg: "B", f1: 0.7383 }, { cfg: "B", f1: 0.7303 }], top7: true },
      { name: "Phi-4-mini (3,8B)", candidate: false, granite: false, mean: 0.7349, min: 0.7261, max: 0.7438, runs: [{ cfg: "A", f1: 0.7261 }, { cfg: "A", f1: 0.727 }, { cfg: "A", f1: 0.7384 }, { cfg: "B", f1: 0.7319 }, { cfg: "B", f1: 0.7438 }, { cfg: "B", f1: 0.7425 }], top7: false },
      { name: "Nemotron-Nano-4B", candidate: false, granite: false, mean: 0.7326, min: 0.7253, max: 0.7456, runs: [{ cfg: "A", f1: 0.7253 }, { cfg: "A", f1: 0.7327 }, { cfg: "A", f1: 0.729 }, { cfg: "B", f1: 0.7305 }, { cfg: "B", f1: 0.7456 }], top7: false },
      { name: "Nanbeige-4.1-3B", candidate: false, granite: false, mean: 0.7289, min: 0.7207, max: 0.7374, runs: [{ cfg: "A", f1: 0.7332 }, { cfg: "A", f1: 0.7374 }, { cfg: "A", f1: 0.7254 }, { cfg: "B", f1: 0.7207 }, { cfg: "B", f1: 0.7237 }, { cfg: "B", f1: 0.7331 }], top7: false }
    ],
    r2Band: [0.7384, 0.7456],   // medias del Top-7 (ANOVA p = 0,45)
    // Definitivos sobre el test automático (labeled-v2). TF-IDF no tiene corrida sobre ese test: solo su F1 de dev automático.
    final: [
      { name: "Granite-4.2-8B", sub: "LoRA · 8B", cfg: "A", auto: { f1: 0.7436, lo: 0.7264, hi: 0.7593, dev: 0.7391, lat: 7.14, min: 74.1, gb: 36.5 } },
      { name: "Granite-4.2-8B", sub: "LoRA · 8B", cfg: "B", auto: { f1: 0.7366, lo: 0.7197, hi: 0.7534, dev: 0.7352, lat: 11.82, min: 99.4, gb: 116.6 } },
      { name: "SciBERT", sub: "ajuste completo · 110M", cfg: "A", auto: { f1: 0.7303, lo: 0.7132, hi: 0.746, dev: 0.7145, lat: 3.12, min: 5.7, gb: 4.7 } },
      { name: "SciBERT", sub: "ajuste completo · 110M", cfg: "B", auto: { f1: 0.7224, lo: 0.7054, hi: 0.7389, dev: 0.734, lat: 6.06, min: 6.3, gb: 4.7 } },
      { name: "TF-IDF + RL", sub: "léxico", cfg: "A", devOnly: 0.5269 },
      { name: "TF-IDF + RL", sub: "léxico", cfg: "B", devOnly: 0.4661 }
    ],
    // Qwen3-0.6B, clasificador vs. generativo (4 corridas). gold: mismas predicciones recalificadas contra el Gold
    // (consolidacion_metricas_a6_a7.ipynb y celda 10c de analizar_resultados.ipynb). gc: con gradient checkpointing.
    gen: [
      { cfg: "A", mode: "clasificador", f1: 0.7459, lo: 0.7296, hi: 0.7619, dev: 0.7296, lat: 1.892, min: 8.6, gb: 6.2, gold: 0.4563 },
      { cfg: "A", mode: "generativo", f1: 0.737, lo: 0.7197, hi: 0.7533, dev: 0.7331, lat: 7.901, min: 12.3, gb: 85.1, gold: 0.434 },
      { cfg: "B", mode: "clasificador", f1: 0.737, lo: 0.7205, hi: 0.7529, dev: 0.7231, lat: 3.154, min: 11.1, gb: 31.7, gold: 0.442 },
      { cfg: "B", mode: "generativo", f1: 0.7422, lo: 0.7257, hi: 0.7585, dev: 0.7324, lat: 9.152, min: 35.9, gb: 34.2, gold: 0.436, gc: true }
    ]
  },

  // A4-02 Definitivos: Test Automático vs. Test Gold A3 (Fig. 4 y Tabla III)
  // TF-IDF no tiene corrida sobre el test automático (Fig. 4 solo lo muestra sobre el Gold): autoF1 = null.
  // devF1 es su Macro-F1 de dev (etiquetas automáticas), que antes figuraba aquí como si fuera el test automático.
  a4Definitives: [
    { name: "Granite-4.2-8B (B)", params: "8B · LoRA",  autoF1: 0.737, goldF1: 0.441, recalF1: 0.463, cfg: "B" },
    { name: "Granite-4.2-8B (A)", params: "8B · LoRA",  autoF1: 0.744, goldF1: 0.435, recalF1: 0.453, cfg: "A" },
    { name: "Qwen3-0.6B (A)",     params: "0,6B · LoRA",autoF1: 0.746, goldF1: 0.456, recalF1: 0.456, cfg: "A" },
    { name: "SciBERT 110M (A)",   params: "110M · Full",autoF1: 0.730, goldF1: 0.462, recalF1: 0.462, cfg: "A" },
    { name: "SciBERT 110M (B)",   params: "110M · Full",autoF1: 0.722, goldF1: 0.433, recalF1: 0.433, cfg: "B" },
    { name: "TF-IDF + RL (A)",    params: "Léxico",     autoF1: null,  devF1: 0.527, goldF1: 0.366, recalF1: 0.366, cfg: "A" },
    { name: "TF-IDF + RL (B)",    params: "Léxico",     autoF1: null,  devF1: 0.466, goldF1: 0.332, recalF1: 0.332, cfg: "B" }
  ],

  // A4-03 Fig. 11 — Diagnóstico por instancia (verificado en test_revisado_por_modelo.csv y cache A5)
  // Valores alineados con la Fig. 11 del informe (Qwen3-0.6B: 84,7 / 64,6 / −49,8 pp; antes 84,8 / 64,5 / −49,7).
  fig11Diagnosis: [
    { name: "Granite-4.2-8B (A4)",    type: "a4", conf: 87.1, auto: 62.9, hum: 17.1, delta: -45.8 },
    { name: "R2 LoRA 2B–4B (10 × A/B)",type: "a4", conf: 85.7, auto: 63.9, hum: 15.7, delta: -48.2 },
    { name: "Qwen3-0.6B (clasif./gen.)", type: "a4", conf: 84.7, auto: 64.6, hum: 14.8, delta: -49.8 },
    { name: "SciBERT 110M (A4)",      type: "a4", conf: 83.5, auto: 63.0, hum: 15.2, delta: -47.8 },
    { name: "GPT-4.1-mini 0s-B (A5)", type: "a5", conf: 69.8, auto: 28.2, hum: 39.0, delta: +10.8 },
    { name: "Gemini-3.8-Flash 0s-B (A5)", type: "a5", conf: 82.8, auto: 19.5, hum: 56.3, delta: +36.8 }
  ],

  // A5-01 Empate aparente (test_panel) vs. Gold A3 + F1 por 9 clases (Tabla IV)
  a5ParityBreak: {
    gptPanel: 0.4407,
    gemPanel: 0.4485,
    pPanel: "0,436 (n.s.)",
    gptGold: 0.4664,
    gemGold: 0.5996,
    deltaGold: "+0,1331",
    pGold: "< 10⁻⁴ (Holm = SÍ)"
  },

  a5ByClass: [
    { cls: "Background",      n: 1200, gpt: 0.556, gem: 0.695 },
    { cls: "Application",     n: 637,  gpt: 0.660, gem: 0.775 },
    { cls: "Comparison",      n: 298,  gpt: 0.626, gem: 0.657 },
    { cls: "Further Reading", n: 227,  gpt: 0.695, gem: 0.941 },
    { cls: "Originator",      n: 105,  gpt: 0.448, gem: 0.472 },
    { cls: "Evidence",        n: 100,  gpt: 0.308, gem: 0.378 },
    { cls: "Basis",           n: 90,   gpt: 0.276, gem: 0.578 },
    { cls: "Gap",             n: 73,   gpt: 0.273, gem: 0.446 },
    { cls: "Improvement",     n: 68,   gpt: 0.357, gem: 0.456 }
  ],

  // A5-02 Fig. 6 — Exactitud A -> B por macro-sección (0-Shot, exacto de test.jsonl)
  a5MacroSections: [
    { sec: "Related Work",   n: 214,  gptA: 41.59, gptB: 51.87, gemA: 68.22, gemB: 68.69, highlight: true },
    { sec: "Intro / Prelim", n: 551,  gptA: 49.55, gptB: 54.26, gemA: 73.68, gemB: 74.95 },
    { sec: "Method / Setup", n: 362,  gptA: 57.46, gptB: 60.77, gemA: 71.55, gemB: 73.20 },
    { sec: "Experiments",    n: 273,  gptA: 52.01, gptB: 51.28, gemA: 65.57, gemB: 64.84 },
    { sec: "Otras",          n: 1356, gptA: 49.78, gptB: 51.47, gemA: 65.04, gemB: 64.90 },
    { sec: "Conclusion",     n: 42,   gptA: 42.86, gptB: 40.48, gemA: 71.43, gemB: 76.19 }
  ],

  // A5-03 Las 9 fichas del prompt 9-Shot (seed=42 en train.jsonl)
  a5NineShotChips: [
    { cls: "Background",      contaminated: true,  origin: "Empate 1-1-1 (agr = 0,333)" },
    { cls: "Improvement",     contaminated: true,  origin: "Empate 1-1-1 (agr = 0,333)" },
    { cls: "Originator",      contaminated: true,  origin: "Empate 1-1-1 (agr = 0,333)" },
    { cls: "Further Reading", contaminated: true,  origin: "Empate 1-1-1 (agr = 0,333)" },
    { cls: "Application",     contaminated: false, origin: "MultiCite / Consenso" },
    { cls: "Comparison",      contaminated: false, origin: "MultiCite / Consenso" },
    { cls: "Basis",           contaminated: false, origin: "MultiCite / Consenso" },
    { cls: "Gap",             contaminated: false, origin: "Consenso A2" },
    { cls: "Evidence",        contaminated: false, origin: "Consenso A2" }
  ],

  // A6 Global Board — Las 14 variantes de Tabla III en 7 filas pareadas (A vs. B)
  a6GlobalBoard: [
    {
      family: "A4 · Supervisados locales (6)",
      accentVar: "--accent-warn",
      rows: [
        { name: "TF-IDF + RL",     f1A: 0.3661, f1B: 0.3317, accA: 40.21, accB: 36.95, eceA: null,   eceB: null,   bestA: false, bestB: false },
        { name: "SciBERT 110M",    f1A: 0.4623, f1B: 0.4331, accA: 50.54, accB: 45.00, eceA: null,   eceB: null,   bestA: true,  bestB: false },
        { name: "Granite-4.2-8B",  f1A: 0.4348, f1B: 0.4412, accA: 45.21, accB: 48.96, eceA: null,   eceB: null,   bestA: false, bestB: true }
      ]
    },
    {
      family: "A5 · GPT-4.1-mini vía API (4)",
      accentVar: "--accent-violet",
      rows: [
        { name: "GPT 9-Shot",      f1A: 0.4146, f1B: 0.4242, accA: 46.43, accB: 49.64, eceA: 0.3282, eceB: 0.3036, bestA: false, bestB: false },
        { name: "GPT 0-Shot",      f1A: 0.4493, f1B: 0.4664, accA: 50.21, accB: 53.07, eceA: 0.3284, eceB: 0.3360, bestA: true,  bestB: true }
      ]
    },
    {
      family: "A5 · Gemini-3.8-Flash vía API (4)",
      accentVar: "--accent-primary",
      rows: [
        { name: "Gemini 9-Shot",   f1A: 0.5974, f1B: 0.5943, accA: 65.65, accB: 65.23, eceA: 0.1139, eceB: 0.1257, bestA: true,  bestB: false },
        { name: "Gemini 0-Shot",   f1A: 0.5900, f1B: 0.5996, accA: 67.98, accB: 68.41, eceA: 0.1204, eceB: 0.1158, bestA: false, bestB: true, globalWinner: true }
      ]
    }
  ],

  // A6-01 Curvas de fiabilidad (exacto de models/llm_api_cache/*_config_b.jsonl)
  a6Calibration: {
    gem: {
      label: "Gemini-3.8-Flash 0s-B",
      ece: 0.1158,
      brier: 0.4809,
      confOk: 0.8290,
      confErr: 0.7368,
      spearman: "+0,4150",
      bins: [
        { bin: "0,4–0,5", n: 13,   conf: 0.4608, acc: 0.0000 },
        { bin: "0,5–0,6", n: 148,  conf: 0.5668, acc: 0.3446 },
        { bin: "0,6–0,7", n: 440,  conf: 0.6752, acc: 0.4318 },
        { bin: "0,7–0,8", n: 487,  conf: 0.7486, acc: 0.5565 },
        { bin: "0,8–0,9", n: 1194, conf: 0.8423, acc: 0.7747 },
        { bin: "0,9–1,0", n: 516,  conf: 0.9318, acc: 0.9244 }
      ]
    },
    gpt: {
      label: "GPT-4.1-mini 0s-B",
      ece: 0.3360,
      brier: 0.8221,
      nll: 6.0666,
      bins: [
        { bin: "0,4–0,5", n: 28,   conf: 0.4213, acc: 0.4286 },
        { bin: "0,5–0,6", n: 166,  conf: 0.5553, acc: 0.4036 },
        { bin: "0,6–0,7", n: 212,  conf: 0.6694, acc: 0.4623 },
        { bin: "0,7–0,8", n: 129,  conf: 0.7504, acc: 0.4264 },
        { bin: "0,8–0,9", n: 992,  conf: 0.8357, acc: 0.4980 },
        { bin: "0,9–1,0", n: 1270, conf: 0.9807, acc: 0.5976 }
      ]
    }
  },

  // A7-01 Relación precisión–costo (Tabla III y Fig. 7 — 8 variantes API)
  a7CostQuality: [
    { id: "gpt0a", label: "GPT 0s-A",    family: "gpt", shot: 0, cfg: "A", cost: 5.83,  f1: 0.4493, acc: 50.21, ece: 0.3284, p50: 915.9,  optimal: true },
    { id: "gpt0b", label: "GPT 0s-B",    family: "gpt", shot: 0, cfg: "B", cost: 6.59,  f1: 0.4664, acc: 53.07, ece: 0.3360, p50: 1025.2, optimal: true },
    { id: "gpt9a", label: "GPT 9s-A",    family: "gpt", shot: 9, cfg: "A", cost: 8.23,  f1: 0.4146, acc: 46.43, ece: 0.3282, p50: 986.5,  optimal: false },
    { id: "gpt9b", label: "GPT 9s-B",    family: "gpt", shot: 9, cfg: "B", cost: 11.61, f1: 0.4242, acc: 49.64, ece: 0.3036, p50: 1045.6, optimal: false },
    { id: "gem0a", label: "Gemini 0s-A", family: "gem", shot: 0, cfg: "A", cost: 16.55, f1: 0.5900, acc: 67.98, ece: 0.1204, p50: 2354.0, optimal: true },
    { id: "gem0b", label: "Gemini 0s-B", family: "gem", shot: 0, cfg: "B", cost: 17.68, f1: 0.5996, acc: 68.41, ece: 0.1158, p50: 2324.6, optimal: true, star: true },
    { id: "gem9a", label: "Gemini 9s-A", family: "gem", shot: 9, cfg: "A", cost: 39.43, f1: 0.5974, acc: 65.65, ece: 0.1139, p50: 2292.4, optimal: false },
    { id: "gem9b", label: "Gemini 9s-B", family: "gem", shot: 9, cfg: "B", cost: 53.09, f1: 0.5943, acc: 65.23, ece: 0.1257, p50: 2481.1, optimal: false }
  ],

  // Q&A Tabla V — 12 contrastes pareados (informe_final_ieee.tex L635-L648)
  qaTablaV: [
    { axis: "Config B vs. A", pair: "GPT 0-Shot (B vs. A)",      base: "0,4493", comp: "0,4664", delta: "+0,0172", pBoot: "3,68×10⁻²", holmBoot: false, pMcN: "6,75×10⁻⁶", holmMcN: true },
    { axis: "Config B vs. A", pair: "GPT 9-Shot (B vs. A)",      base: "0,4146", comp: "0,4242", delta: "+0,0096", pBoot: "2,40×10⁻¹", holmBoot: false, pMcN: "1,54×10⁻⁵", holmMcN: true },
    { axis: "Config B vs. A", pair: "Gemini 0-Shot (B vs. A)",   base: "0,5900", comp: "0,5996", delta: "+0,0096", pBoot: "1,22×10⁻¹", holmBoot: false, pMcN: "3,46×10⁻¹", holmMcN: false },
    { axis: "Config B vs. A", pair: "Gemini 9-Shot (B vs. A)",   base: "0,5974", comp: "0,5943", delta: "−0,0031", pBoot: "5,92×10⁻¹", holmBoot: false, pMcN: "4,12×10⁻¹", holmMcN: false },
    { axis: "9-Shot vs. 0-Shot", pair: "GPT Config A (9s vs. 0s)", base: "0,4493", comp: "0,4146", delta: "−0,0347", pBoot: "7,00×10⁻⁴", holmBoot: true,  pMcN: "6,74×10⁻⁶", holmMcN: true },
    { axis: "9-Shot vs. 0-Shot", pair: "GPT Config B (9s vs. 0s)", base: "0,4664", comp: "0,4242", delta: "−0,0423", pBoot: "1,00×10⁻⁴", holmBoot: true,  pMcN: "5,43×10⁻⁵", holmMcN: true },
    { axis: "9-Shot vs. 0-Shot", pair: "Gemini Config A (9s vs. 0s)", base: "0,5900", comp: "0,5974", delta: "+0,0075", pBoot: "3,64×10⁻¹", holmBoot: false, pMcN: "1,20×10⁻⁴", holmMcN: true },
    { axis: "9-Shot vs. 0-Shot", pair: "Gemini Config B (9s vs. 0s)", base: "0,5996", comp: "0,5943", delta: "−0,0053", pBoot: "5,05×10⁻¹", holmBoot: false, pMcN: "1,12×10⁻⁷", holmMcN: true },
    { axis: "Gemini vs. GPT", pair: "0-Shot Config A",           base: "0,4493", comp: "0,5900", delta: "+0,1407", pBoot: "< 10⁻⁴",    holmBoot: true,  pMcN: "4,24×10⁻⁶⁹", holmMcN: true },
    { axis: "Gemini vs. GPT", pair: "0-Shot Config B",           base: "0,4664", comp: "0,5996", delta: "+0,1331", pBoot: "< 10⁻⁴",    holmBoot: true,  pMcN: "5,17×10⁻⁵⁴", holmMcN: true },
    { axis: "Gemini vs. GPT", pair: "9-Shot Config A",           base: "0,4146", comp: "0,5974", delta: "+0,1828", pBoot: "< 10⁻⁴",    holmBoot: true,  pMcN: "9,89×10⁻⁷⁵", holmMcN: true },
    { axis: "Gemini vs. GPT", pair: "9-Shot Config B",           base: "0,4242", comp: "0,5943", delta: "+0,1701", pBoot: "< 10⁻⁴",    holmBoot: true,  pMcN: "5,32×10⁻⁵⁰", holmMcN: true }
  ],

  // Q&A Tabla VI — Exactitud por estrato previo (informe_final_ieee.tex L708-L714)
  qaTablaVI: [
    { stratum: "Human Gold (MultiCite)",       n: "563 (20,1%)",   gpt0b: "51,87%", gpt9b: "50,27%", gem0b: "60,75%", gem9b: "56,31%" },
    { stratum: "Panel Unánime (3/3)",          n: "728 (26,0%)",   gpt0b: "59,89%", gpt9b: "54,81%", gem0b: "72,25%", gem9b: "68,13%" },
    { stratum: "Panel Mayoría (2/3)",          n: "387 (13,8%)",   gpt0b: "54,78%", gpt9b: "52,97%", gem0b: "70,28%", gem9b: "68,48%" },
    { stratum: "Needs Review (Desacuerdo)",    n: "1.120 (40,0%)", gpt0b: "48,66%", gpt9b: "44,82%", gem0b: "69,11%", gem9b: "66,70%" },
    { stratum: "Panel A2 acertó vs. Humano",   n: "958 (34,2%)",   gpt0b: "80,79%", gpt9b: "73,90%", gem0b: "89,04%", gem9b: "88,20%", hi: true },
    { stratum: "Panel A2 falló vs. Humano",    n: "1.840 (65,8%)", gpt0b: "38,64%", gpt9b: "37,01%", gem0b: "57,66%", gem9b: "53,26%", hi: true }
  ],

  // Q&A Tabla VII — Matriz de Vecindad Retórica
  qaTablaVII: [
    { cls: "Background",      band: "[0,62; 0,78]", n1: "Further Reading, Gap",   n2: "Originator, Evidence" },
    { cls: "Gap",             band: "[0,60; 0,76]", n1: "Background, Comparison", n2: "Evidence, Improvement" },
    { cls: "Basis",           band: "[0,56; 0,72]", n1: "Improvement, Application", n2: "Background, Originator" },
    { cls: "Comparison",      band: "[0,58; 0,75]", n1: "Improvement, Evidence",  n2: "Gap, Background" },
    { cls: "Application",     band: "[0,60; 0,76]", n1: "Basis, Improvement",     n2: "Evidence, Comparison" },
    { cls: "Improvement",     band: "[0,56; 0,72]", n1: "Basis, Comparison",      n2: "Application, Evidence" },
    { cls: "Evidence",        band: "[0,64; 0,80]", n1: "Comparison, Background", n2: "Basis, Gap" },
    { cls: "Originator",      band: "[0,64; 0,80]", n1: "Background, Basis",      n2: "Application, Evidence" },
    { cls: "Further Reading", band: "[0,68; 0,84]", n1: "Background, Evidence",   n2: "Application, Basis" }
  ]
};
