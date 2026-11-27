/* ==========================================================================
   EvidenCite — Renderizadores SVG Nativos con Transiciones y Tooltips
   Sin dependencias externas, 100% integrados al lienzo editorial
   ========================================================================== */

window.EvidenCharts = (function () {
  const SVG_NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, parent) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        node.setAttribute(k, String(v));
      }
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function txt(content, attrs, parent) {
    const node = el("text", attrs, parent);
    node.textContent = content;
    return node;
  }

  function fmtComma(num, dec) {
    return Number(num).toFixed(dec).replace(".", ",");
  }

  // Smooth 60fps requestAnimationFrame tween (cubic ease-out) with clean cancellation
  function tween(durationMs, onFrame, delayMs) {
    const delay = delayMs || 0;
    let rafId = null;
    let timerId = null;
    let cancelled = false;

    function start() {
      if (cancelled) return;
      if (durationMs <= 0) {
        onFrame(1);
        return;
      }
      const t0 = performance.now();
      function tick(now) {
        if (cancelled) return;
        const raw = Math.min(1, (now - t0) / durationMs);
        const eased = 1 - Math.pow(1 - raw, 3);
        onFrame(eased);
        if (raw < 1) {
          rafId = requestAnimationFrame(tick);
        }
      }
      rafId = requestAnimationFrame(tick);
    }

    if (delay > 0) {
      timerId = setTimeout(start, delay);
    } else {
      rafId = requestAnimationFrame(start);
    }

    return function cancel() {
      cancelled = true;
      if (timerId !== null) clearTimeout(timerId);
      if (rafId !== null) cancelAnimationFrame(rafId);
    };
  }

  // Tooltip controller clamped inside viewport
  const Tooltip = {
    node: null,
    init() {
      this.node = document.getElementById("svg-tooltip");
    },
    show(evt, title, rows) {
      if (!this.node) this.init();
      if (!this.node) return;
      const html =
        `<div class="tt-title">${title}</div>` +
        rows.map((r) => `<div class="tt-row">${r}</div>`).join("");
      this.node.innerHTML = html;
      this.node.classList.add("is-visible");
      this.node.setAttribute("aria-hidden", "false");
      this.move(evt);
    },
    move(evt) {
      if (!this.node) return;
      const pad = 14;
      const rect = this.node.getBoundingClientRect();
      let x = evt.clientX + pad;
      let y = evt.clientY - rect.height - 10;
      if (x + rect.width > window.innerWidth - 12) {
        x = evt.clientX - rect.width - pad;
      }
      if (x < 12) x = 12;
      if (y < 12) y = evt.clientY + pad;
      if (y + rect.height > window.innerHeight - 12) {
        y = window.innerHeight - rect.height - 12;
      }
      this.node.style.left = `${x}px`;
      this.node.style.top = `${y}px`;
    },
    hide() {
      if (!this.node) return;
      this.node.classList.remove("is-visible");
      this.node.setAttribute("aria-hidden", "true");
    }
  };

  function bindTip(node, title, rows) {
    node.classList.add("svg-hoverable");
    node.addEventListener("mouseenter", (e) => Tooltip.show(e, title, rows));
    node.addEventListener("mousemove", (e) => Tooltip.move(e));
    node.addEventListener("mouseleave", () => Tooltip.hide());
  }

  /* ========================================================================
     1. A1-02 — DAG Vertical Medallion (5 Capas + Zoom 512 tokens + NLI)
     ======================================================================== */
  function createMedallionDAG(container) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 860 545", role: "img", "aria-label": "DAG vertical Medallion del pipeline A1" }, container);

    // Arrowhead marker
    const defs = el("defs", {}, svg);
    const marker = el("marker", { id: "dag-arrow", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, defs);
    el("path", { d: "M 0 1 L 10 5 L 0 9 z", fill: "var(--ink-3)" }, marker);

    // Layer 1 & 2 (Branches converging) — group g1
    const g1 = el("g", { class: "svg-anim", id: "dag-stage-1" }, svg);
    txt("CAPAS 1–2 · INGESTA MEDALLION Y RESCATE SEMINAL", { x: 24, y: 24, class: "svg-mono", "font-size": 11, "font-weight": 700, fill: "var(--accent-primary)" }, g1);

    // Branch A: MultiCite
    const bA = el("g", {}, g1);
    el("rect", { x: 24, y: 38, width: 392, height: 84, rx: 8, fill: "var(--accent-gold-soft)", stroke: "var(--accent-gold)", "stroke-width": 1.6 }, bA);
    txt("MultiCite (Human Gold · 7 clases)", { x: 40, y: 62, "font-size": 14, "font-weight": 700 }, bA);
    txt("11.251 citas → cruce inicial 0,4% (43)", { x: 40, y: 83, class: "svg-mono svg-ink2", "font-size": 11.8 }, bA);
    txt("Rescate Top-250 (caché + APIs) → 3.629 Gold 1:1", { x: 40, y: 105, class: "svg-mono", "font-size": 11.4, "font-weight": 700, fill: "var(--accent-gold)" }, bA);
    bindTip(bA, "Rama MultiCite + Rescate Seminal", [
      "Cruce inicial sin rescate: 43 / 11.251 (0,4%)",
      "Adquisición Top-250 (ar5iv + PDF Docling): 146 validados / 135 cuerpos",
      "Resultado: 3.629 citas Gold 1:1 rescatadas"
    ]);

    // Branch B: unarXive
    const bB = el("g", {}, g1);
    el("rect", { x: 436, y: 38, width: 400, height: 84, rx: 8, fill: "var(--accent-secondary-soft)", stroke: "var(--accent-secondary)", "stroke-width": 1.6 }, bB);
    txt("unarXive 2024 (CS · Texto Completo LaTeX)", { x: 452, y: 62, "font-size": 14, "font-weight": 700 }, bB);
    txt("66.314 artículos estructurados en Capa 2", { x: 452, y: 83, class: "svg-mono svg-ink2", "font-size": 11.8 }, bB);
    txt("30.722 citados con cuerpo → 144.826 candidatos", { x: 452, y: 105, class: "svg-mono", "font-size": 11.4, "font-weight": 700, fill: "var(--accent-secondary)" }, bB);
    bindTip(bB, "Rama unarXive (Capa 2)", [
      "66.314 artículos normalizados con encabezados autorales",
      "30.722 artículos citados con fragmentos válidos",
      "Apoyado en caché local SQLite y APIs académicas"
    ]);

    // Converging arrows to Layer 3
    el("path", { d: "M 220 122 L 220 144 L 418 144 L 418 164", fill: "none", stroke: "var(--ink-3)", "stroke-width": 1.5, "marker-end": "url(#dag-arrow)" }, g1);
    el("path", { d: "M 636 122 L 636 144 L 442 144 L 442 164", fill: "none", stroke: "var(--ink-3)", "stroke-width": 1.5, "marker-end": "url(#dag-arrow)" }, g1);

    // Layer 3 (Zoom inline A1-03: 512-token budget + A1-04 NLI) — group g2
    const g2 = el("g", { class: "svg-anim", id: "dag-stage-2" }, svg);
    txt("CAPA 3 · RECUPERACIÓN TOP-3 (ZOOM A1-03) Y FILTRADO NLI (A1-04)", { x: 24, y: 184, class: "svg-mono", "font-size": 11, "font-weight": 700, fill: "var(--accent-primary)" }, g2);

    el("rect", { x: 24, y: 196, width: 812, height: 158, rx: 10, fill: "var(--surface-tint)", stroke: "var(--accent-primary)", "stroke-width": 1.8 }, g2);

    // Left sub-block: 512 token budget bar
    txt("Presupuesto exacto de 512 tokens en SciBERT (Config B)", { x: 42, y: 222, "font-size": 13.5, "font-weight": 700 }, g2);
    txt("Consulta aislada SIN encabezados (evita homonimia Introduction–Introduction) · cos Top-1 = 0,8943", { x: 42, y: 242, class: "svg-ink2", "font-size": 11.8 }, g2);

    const barX = 42, barY = 256, barW = 776, barH = 34;
    const segs = [
      { label: "[CLS]+Cit. ~90t", pct: 0.18, color: "#334155", tip: "Ventana citante m=3 oraciones (media 83,94 pal. ≈ 90 tokens)" },
      { label: "Chunk R1 (≤110 pal · ~140 tok)", pct: 0.273, color: "var(--accent-primary)", tip: "Top-1 pasaje + encabezado autoral preservado" },
      { label: "Chunk R2 (~140 tok)", pct: 0.273, color: "var(--accent-secondary)", tip: "Top-2 pasaje + encabezado autoral preservado" },
      { label: "Chunk R3 (~140 tok)", pct: 0.274, color: "var(--accent-violet)", tip: "Top-3 pasaje + encabezado autoral preservado" }
    ];
    let cx = barX;
    segs.forEach((s) => {
      const w = barW * s.pct;
      const r = el("rect", { x: cx, y: barY, width: w - 3, height: barH, rx: 5, fill: s.color }, g2);
      txt(s.label, { x: cx + (w - 3) / 2, y: barY + 21, "text-anchor": "middle", "font-size": 10.8, "font-weight": 700, fill: "#FFFFFF" }, g2);
      bindTip(r, s.label, [s.tip, "Límite SciBERT: 512 tokens totales sin truncar chunks"]);
      cx += w;
    });

    // Brief NLI strip inside Layer 3 (A1-04)
    const nliBox = el("g", {}, g2);
    el("rect", { x: 42, y: 304, width: 776, height: 36, rx: 6, fill: "var(--accent-warn-soft)", stroke: "var(--accent-warn)", "stroke-width": 1.2 }, nliBox);
    txt("NLI calibrado (DeBERTa-v3): 135.035 → 63.802 aceptadas (47,25%) para llenar cuotas cuidando calidad", { x: 56, y: 326, class: "svg-mono", "font-size": 10.9, "font-weight": 600, fill: "var(--ink)" }, nliBox);
    bindTip(nliBox, "Supervisión Débil NLI (A1-04)", [
      "Evaluados: 135.035 · Aceptados: 63.802 (47,25%)",
      "Umbrales calibrados: τ=0,6 · θ_conf=0,50 · θ_margin=0,15",
      "Admisión 2,44× y captura Basis/Improvement 5,5–13×"
    ]);

    // Arrow to Layer 4-5
    el("path", { d: "M 430 354 L 430 384", fill: "none", stroke: "var(--ink-3)", "stroke-width": 1.7, "marker-end": "url(#dag-arrow)" }, g2);

    // Layer 4 & 5 — Consolidated Dataset 1:1 + Leakage=0 + O(1) — group g3
    const g3 = el("g", { class: "svg-anim", id: "dag-stage-3" }, svg);
    txt("CAPAS 4–5 · CONSOLIDACIÓN 1:1, ANTI-FUGA Y ENSAMBLADO O(1)", { x: 24, y: 402, class: "svg-mono", "font-size": 11, "font-weight": 700, fill: "var(--accent-gold)" }, g3);

    el("rect", { x: 24, y: 414, width: 812, height: 112, rx: 10, fill: "var(--accent-gold-soft)", stroke: "var(--accent-gold)", "stroke-width": 1.8 }, g3);

    const splits = [
      { x: 44,  w: 240, title: "Train (70,1%)", val: "13.077", sub: "5.835 citantes únicos", col: "var(--accent-secondary)" },
      { x: 300, w: 170, title: "Dev (14,9%)",   val: "2.783",  sub: "1.243 citantes",        col: "var(--accent-primary)" },
      { x: 486, w: 170, title: "Test (15,0%)",  val: "2.798",  sub: "1.258 citantes · A3",   col: "var(--accent-gold)" }
    ];
    splits.forEach((sp) => {
      const box = el("g", {}, g3);
      el("rect", { x: sp.x, y: 428, width: sp.w, height: 82, rx: 6, fill: "var(--surface)", stroke: sp.col, "stroke-width": 1.4 }, box);
      txt(sp.title, { x: sp.x + 12, y: 448, class: "svg-mono", "font-size": 11, "font-weight": 700, fill: sp.col }, box);
      txt(sp.val, { x: sp.x + 12, y: 478, class: "svg-mono", "font-size": 22, "font-weight": 700 }, box);
      txt(sp.sub, { x: sp.x + 12, y: 498, class: "svg-muted", "font-size": 11 }, box);
      bindTip(box, `${sp.title}: ${sp.val} pares 1:1`, [sp.sub, "Paridad estricta Config A = Config B", "0 artículos citantes compartidos entre splits"]);
    });

    // Lock + O(1) badge
    const badge = el("g", {}, g3);
    el("rect", { x: 672, y: 428, width: 148, height: 82, rx: 6, fill: "var(--ink)", opacity: 0.92 }, badge);
    txt("LEAKAGE = 0", { x: 746, y: 454, "text-anchor": "middle", class: "svg-mono", "font-size": 13, "font-weight": 700, fill: "var(--bg)" }, badge);
    txt("8.336 citantes", { x: 746, y: 474, "text-anchor": "middle", class: "svg-mono", "font-size": 11, fill: "var(--bg)" }, badge);
    txt("Acceso O(1)/inst.", { x: 746, y: 496, "text-anchor": "middle", class: "svg-mono", "font-size": 11, "font-weight": 700, fill: "#25DB9A" }, badge);
    bindTip(badge, "Garantías Estructurales A1", [
      "Leakage por citing_paper_id: 0 (Train/Dev/Test disjuntos)",
      "Total consolidado: 18.658 pares 1:1",
      "Ensamblado determinístico O(1) por instancia para A4 y A5"
    ]);

    return {
      update(stage) {
        // stage 0: highlight g1; stage 1: highlight g2; stage 2: all lit with g3 hero
        g1.style.opacity = stage === 0 ? "1" : "0.78";
        g2.style.opacity = stage === 0 ? "0.22" : stage === 1 ? "1" : "0.85";
        g3.style.opacity = stage < 2 ? "0.22" : "1";
      }
    };
  }

  /* ========================================================================
     2. A2-01 — Panel 3 Jueces (2 pasos de scroll)
     ======================================================================== */
  function createA2Panel(container, data) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 860 470", role: "img", "aria-label": "Los tres jueces de A2" }, container);
    const acento = "var(--accent-violet)";
    const nombres = [["Llama-3.1-8B-Instruct", "Meta"], ["Qwen2.5-7B-Instruct", "Alibaba"], ["Mistral-7B-Instruct-v0.3", "Mistral AI"]];

    const cardW = 250, gap = 30;
    nombres.forEach((n, i) => {
      const x = 24 + i * (cardW + gap);
      const g = el("g", {}, svg);
      el("rect", { x, y: 40, width: cardW, height: 112, rx: 10, fill: "var(--accent-violet-soft)", stroke: acento, "stroke-width": 1.6 }, g);
      txt(`Juez ${i + 1}`, { x: x + 18, y: 70, "font-size": 13, "font-weight": 600, fill: acento }, g);
      txt(n[0], { x: x + 18, y: 104, "font-size": 16, "font-weight": 700 }, g);
      txt(n[1], { x: x + 18, y: 132, class: "svg-ink2", "font-size": 14 }, g);
      bindTip(g, data.judges[i].name, [`Abstención: ${data.judges[i].abst}`]);
    });

    const verdict = el("g", { class: "svg-anim", id: "a2-verdict" }, svg);
    const filas = [
      { lbl: "Acuerdo entre los jueces (Kappa)", val: 0.3939, txtv: "0,39", col: "var(--ink-3)", y: 226 },
      { lbl: "Acierto frente a MultiCite", val: 0.364, txtv: "36 %", col: acento, y: 326 }
    ];
    filas.forEach((f) => {
      txt(f.lbl, { x: 24, y: f.y, class: "svg-ink2", "font-size": 15 }, verdict);
      el("rect", { x: 24, y: f.y + 14, width: 620, height: 30, rx: 6, fill: "var(--surface-tint)" }, verdict);
      el("rect", { x: 24, y: f.y + 14, width: 620 * f.val, height: 30, rx: 6, fill: f.col }, verdict);
      txt(f.txtv, { x: 664, y: f.y + 37, class: "svg-serif", "font-size": 24, "font-weight": 700, fill: f.col }, verdict);
    });

    return {
      update(revealed) {
        verdict.style.opacity = revealed ? "1" : "0";
        verdict.style.transform = revealed ? "translateY(0)" : "translateY(14px)";
      }
    };
  }

  function renderA2Strata(container, data) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 860 400", role: "img", "aria-label": "Cómo quedó el corpus después del panel" }, container);
    const acento = "var(--accent-violet)";
    const partes = [
      { pct: 19.4501, n: "3.629", col: "var(--line-strong)", tinta: "var(--ink)", t: "MultiCite", d: "Etiqueta humana, se deja como está." },
      { pct: 39.87, n: "7.439", col: "var(--ink-3)", tinta: "#FFFFFF", t: "Confirmadas por el panel", d: "Los jueces respaldaron la etiqueta." },
      { pct: 40.68, n: "7.590", col: acento, tinta: "#FFFFFF", t: "Sin respaldo del panel", d: "Conservaron la etiqueta anterior y así entraron a entrenamiento." }
    ];
    const barX = 24, barY = 40, barW = 812, barH = 64;
    let cx = barX;
    partes.forEach((s) => {
      const w = (s.pct / 100) * barW;
      el("rect", { x: cx, y: barY, width: w - 4, height: barH, rx: 6, fill: s.col }, svg);
      txt(`${fmtComma(s.pct, 1)} %`, { x: cx + (w - 4) / 2, y: barY + 40, "text-anchor": "middle", "font-size": 19, "font-weight": 700, fill: s.tinta }, svg);
      cx += w;
    });
    partes.forEach((s, i) => {
      const y = 160 + i * 74;
      el("rect", { x: 24, y: y - 16, width: 14, height: 14, rx: 3, fill: s.col }, svg);
      txt(`${s.t} · ${s.n}`, { x: 50, y: y - 4, "font-size": 16, "font-weight": 700 }, svg);
      txt(s.d, { x: 50, y: y + 22, class: "svg-ink2", "font-size": 14 }, svg);
    });
  }

  function renderA3Waterfall(container, wf) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 860 420", role: "img", "aria-label": "Cómo se resolvió el test en A3" }, container);
    const acento = "var(--accent-violet)";
    const bX = 24, bW = 812;
    const humano = (wf.humanDirectTotal / wf.total) * bW;

    el("rect", { x: bX, y: 40, width: humano - 4, height: 64, rx: 6, fill: acento }, svg);
    el("rect", { x: bX + humano, y: 40, width: bW - humano, height: 64, rx: 6, fill: "var(--ink-3)" }, svg);
    txt("68 %", { x: bX + (humano - 4) / 2, y: 82, "text-anchor": "middle", "font-size": 21, "font-weight": 700, fill: "#FFFFFF" }, svg);
    txt("32 %", { x: bX + humano + (bW - humano) / 2, y: 82, "text-anchor": "middle", "font-size": 21, "font-weight": 700, fill: "#FFFFFF" }, svg);
    txt("Revisión humana o MultiCite · 1.904", { x: bX, y: 136, "font-size": 15, "font-weight": 700 }, svg);
    txt("Desempate entre las dos etiquetas humanas · 894", { x: bX + bW, y: 136, "text-anchor": "end", "font-size": 15, "font-weight": 700 }, svg);

    txt("Acuerdo (Kappa)", { x: 24, y: 220, class: "svg-ink2", "font-size": 15 }, svg);
    const kappas = [
      { lbl: "Panel A2", val: 0.3939, txtv: "0,394", col: "var(--ink-3)", y: 244 },
      { lbl: "Revisores", val: 0.509, txtv: "0,509", col: acento, y: 300 }
    ];
    kappas.forEach((k) => {
      txt(k.lbl, { x: 24, y: k.y + 20, "font-size": 15, "font-weight": 600 }, svg);
      el("rect", { x: 170, y: k.y, width: 480, height: 28, rx: 6, fill: "var(--surface-tint)" }, svg);
      el("rect", { x: 170, y: k.y, width: 480 * k.val, height: 28, rx: 6, fill: k.col }, svg);
      txt(k.txtv, { x: 670, y: k.y + 22, class: "svg-serif", "font-size": 22, "font-weight": 700, fill: k.col }, svg);
    });
  }

  function createA3Morph(container, rows) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 860 480", role: "img", "aria-label": "Distribución del test antes y después de A3" }, container);
    const acento = "var(--accent-violet)";
    const marginL = 162, chartW = 560, startY = 40, rowH = 42;
    const maxPct = 46;

    [0, 10, 20, 30, 40].forEach((tick) => {
      const x = marginL + (tick / maxPct) * chartW;
      el("line", { x1: x, y1: startY - 12, x2: x, y2: startY + rows.length * rowH - 8, class: "svg-grid" }, svg);
      txt(`${tick} %`, { x, y: startY + rows.length * rowH + 14, "text-anchor": "middle", class: "svg-muted", "font-size": 12 }, svg);
    });

    const barEls = [];
    rows.forEach((r, i) => {
      const y = startY + i * rowH;
      const g = el("g", {}, svg);
      txt(r.cls, { x: marginL - 14, y: y + 18, "text-anchor": "end", "font-size": 14, "font-weight": r.cls === "Background" ? 700 : 500 }, g);
      const preW = (r.prePct / maxPct) * chartW;
      el("rect", { x: marginL, y: y + 4, width: preW, height: 22, rx: 4, fill: "none", stroke: "var(--line-strong)", "stroke-dasharray": "4 3", "stroke-width": 1.2 }, g);
      const rect = el("rect", { x: marginL, y: y + 4, width: preW, height: 22, rx: 4, fill: "var(--ink-3)" }, g);
      rect.style.transition = "fill 500ms ease";
      const lbl = txt(`${fmtComma(r.prePct, 1)} %`, { x: marginL + preW + 10, y: y + 20, "font-size": 13, "font-weight": 600 }, g);
      bindTip(g, r.cls, [`Antes: ${r.preN} citas`, `Después: ${r.goldN} citas`]);
      barEls.push({ r, rect, lbl, curPct: r.prePct });
    });

    let activeCancel = null;
    let lastGoldState = null;

    return {
      update(isGold) {
        if (lastGoldState === isGold) return;
        const firstPaint = lastGoldState === null;
        lastGoldState = isGold;
        if (activeCancel) activeCancel();

        const snapshots = barEls.map((item) => {
          const destacada = isGold && item.r.cls === "Background";
          item.rect.setAttribute("fill", destacada ? acento : "var(--ink-3)");
          return { item, fromPct: item.curPct, toPct: isGold ? item.r.goldPct : item.r.prePct };
        });

        activeCancel = tween(firstPaint ? 0 : 1300, (t) => {
          snapshots.forEach((s) => {
            const p = s.fromPct + (s.toPct - s.fromPct) * t;
            s.item.curPct = p;
            const w = (p / maxPct) * chartW;
            s.item.rect.setAttribute("width", String(w));
            s.item.lbl.setAttribute("x", String(marginL + w + 10));
            s.item.lbl.textContent = `${fmtComma(p, 1)} %`;
          });
        }, firstPaint ? 0 : 120);
      }
    };
  }

  /* ========================================================================
     6. A4-01 — Embudo de 92 corridas en 4 pestañas (R1 · R2 · Final · Gen. vs. clasif.)
     Cada pestaña dibuja su propia gráfica; la de los modelos definitivos va por defecto.
     Config A = círculo, Config B = cuadrado (forma + color, nunca solo color).
     ======================================================================== */
  function createA4Funnel(container, data, opts) {
    const options = opts || {};
    const TABS = [
      { key: "r1",    t: "R1 · 19 corridas",     s: "Descarta multimodales y MoE",          aria: "Ronda 1: Macro-F1 de test e IC95 de 10 modelos en Config A y B" },
      { key: "r2",    t: "R2 · 58 corridas",     s: "10 LLMs 2B–4B × 3 semillas",           aria: "Ronda 2: las 58 corridas y la media por modelo" },
      { key: "final", t: "Final · 11 corridas",  s: "Granite-4.2-8B · SciBERT · TF-IDF",    aria: "Modelos definitivos sobre el test automático" },
      { key: "gen",   t: "Gen. vs. clasif. · 4", s: "Qwen3-0.6B · misma base y LoRA",       aria: "Qwen3-0.6B: modo generativo frente a cabeza de clasificación" }
    ];
    const CA = "var(--a4-cfg-a)", CB = "var(--a4-cfg-b)";
    const colOf = (cfg) => (cfg === "A" ? CA : CB);
    const f3 = (v) => fmtComma(v, 3);

    container.innerHTML = `
      <div class="a4-funnel">
        <div class="a4-tabs" role="tablist" aria-label="Etapas del embudo de selección de A4">
          ${TABS.map((tb, i) => `<button type="button" role="tab" id="a4-tab-${tb.key}" data-a4-tab="${tb.key}" aria-controls="a4-tabpanel" aria-selected="false"><span class="a4-tab-step">${i + 1}</span><span class="a4-tab-t">${tb.t}</span><span class="a4-tab-s">${tb.s}</span></button>`).join("")}
        </div>
        <div class="a4-tabpanel" id="a4-tabpanel" role="tabpanel"></div>
      </div>`;
    const root = container.querySelector(".a4-funnel");
    const panel = root.querySelector(".a4-tabpanel");
    const buttons = Array.from(root.querySelectorAll("[data-a4-tab]"));

    // ---- piezas comunes ----
    function marker(parent, cfg, x, y, o) {
      const opt = o || {};
      const r = opt.r || 5;
      const fill = opt.hollow ? "var(--bg)" : colOf(cfg);
      const stroke = opt.hollow ? colOf(cfg) : "var(--bg)";
      if (opt.diamond) {
        const d = r + 1.5;
        return el("polygon", { points: `${x},${y - d} ${x + d},${y} ${x},${y + d} ${x - d},${y}`, fill, stroke, "stroke-width": 2 }, parent);
      }
      if (cfg === "A") return el("circle", { cx: x, cy: y, r, fill, stroke, "stroke-width": 2 }, parent);
      const side = r * 1.8;
      return el("rect", { x: x - side / 2, y: y - side / 2, width: side, height: side, rx: 1.5, fill, stroke, "stroke-width": 2 }, parent);
    }
    function whisker(parent, cfg, xa, xb, y) {
      el("line", { x1: xa, y1: y, x2: xb, y2: y, stroke: colOf(cfg), "stroke-width": 2, "stroke-linecap": "round", opacity: 0.5 }, parent);
    }
    function grid(svg, xs, ticks, y0, y1, dec) {
      ticks.forEach((tk) => {
        const x = xs(tk);
        el("line", { x1: x, y1: y0, x2: x, y2: y1, class: "svg-grid" }, svg);
        txt(fmtComma(tk, dec), { x, y: y1 + 17, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 11.5 }, svg);
      });
    }
    function legend(svg, items, x, y) {
      let cx = x;
      items.forEach(([kind, label]) => {
        if (kind === "A" || kind === "B") marker(svg, kind, cx + 5, y - 4, { r: 4.6 });
        else if (kind === "dev") marker(svg, "A", cx + 5, y - 4, { r: 3.6, hollow: true, diamond: true });
        else if (kind === "band") el("rect", { x: cx, y: y - 11, width: 12, height: 13, rx: 2, fill: "var(--ink-3)", opacity: 0.16 }, svg);
        else if (kind === "tick") el("line", { x1: cx + 5, y1: y - 11, x2: cx + 5, y2: y + 2, stroke: "var(--ink)", "stroke-width": 2.5 }, svg);
        else if (kind === "ci") el("line", { x1: cx, y1: y - 4, x2: cx + 14, y2: y - 4, stroke: "var(--ink-3)", "stroke-width": 2, "stroke-linecap": "round" }, svg);
        const t = txt(label, { x: cx + 18, y, class: "svg-ink2", "font-size": 12 }, svg);
        cx += 18 + (t.getComputedTextLength() || label.length * 6.6) + 22;
      });
    }
    function hitRow(g, x, y, w, h) {
      el("rect", { x, y, width: w, height: h, fill: "transparent" }, g);
    }
    function band(svg, xa, xb, y0, y1) {
      el("rect", { x: xa, y: y0, width: Math.max(2, xb - xa), height: y1 - y0, fill: "var(--ink-3)", opacity: 0.12 }, svg);
    }
    // Barra horizontal: esquina de datos redondeada, base recta
    function barPath(x0, y, w, h) {
      const r = Math.min(4, w / 2, h / 2);
      return `M${x0},${y} H${x0 + w - r} Q${x0 + w},${y} ${x0 + w},${y + r} V${y + h - r} Q${x0 + w},${y + h} ${x0 + w - r},${y + h} H${x0} Z`;
    }

    // ---- R1: 10 modelos de 7 familias, una semilla (Fig. 10a) ----
    function drawR1(svg) {
      const rows = data.r1;
      const mL = 182, cW = 600, x0 = 0.765, x1 = 0.86;
      const xs = (v) => mL + ((v - x0) / (x1 - x0)) * cW;
      const top = 70, rowH = 36;
      const yOf = (i) => top + i * rowH;
      const yEnd = yOf(rows.length - 1) + rowH / 2;
      legend(svg, [["A", "Config A (título + resumen)"], ["B", "Config B (Top-3 fragmentos)"], ["ci", "IC95"]], mL, 22);
      const d0 = rows.findIndex((r) => r.discarded);
      el("rect", { x: 14, y: yOf(d0) - rowH / 2, width: mL + cW - 6, height: yEnd - yOf(d0) + rowH / 2, rx: 6, fill: "var(--ink-3)", opacity: 0.09 }, svg);
      txt("descartados", { x: mL + cW - 8, y: yEnd - 8, "text-anchor": "end", class: "svg-mono svg-muted", "font-size": 11.5, "font-style": "italic" }, svg);
      grid(svg, xs, [0.78, 0.80, 0.82, 0.84, 0.86], top - 20, yEnd + 4, 2);
      rows.forEach((r, i) => {
        const y = yOf(i);
        const g = el("g", {}, svg);
        hitRow(g, 14, y - rowH / 2, mL + cW, rowH);
        txt(r.name, { x: mL - 14, y: y + 4.5, "text-anchor": "end", "font-size": 13, "font-weight": 500, fill: r.discarded ? "var(--ink-2)" : "var(--ink)" }, g);
        const tip = [];
        [["A", r.a, -6.5], ["B", r.b, 6.5]].forEach(([cfg, v, dy]) => {
          if (!v) return;
          whisker(g, cfg, xs(v.lo), xs(v.hi), y + dy);
          marker(g, cfg, xs(v.f1), y + dy, { r: 4.6 });
          tip.push(`Config ${cfg}: ${f3(v.f1)} [${f3(v.lo)}–${f3(v.hi)}] · ${fmtComma(v.min, 1)} min · ${fmtComma(v.gb, 0)} GB`);
        });
        if (r.name.endsWith("*")) tip.push("MoE: LoRA cubrió solo el 0,12% de los parámetros; otra receta (1.024 tokens, LR 2e-4)");
        else if (r.discarded) tip.push("Descartado: últimas posiciones de R1");
        bindTip(g, r.name.replace("*", ""), tip);
      });
      txt("Macro-F1 de test e IC95 · etiquetado previo con sección citante (no comparable con R2)", { x: mL + cW / 2, y: yEnd + 44, "text-anchor": "middle", class: "svg-muted", "font-size": 12 }, svg);
      txt("* mezcla de expertos (MoE), solo Config B y otra receta", { x: mL + cW, y: yEnd + 66, "text-anchor": "end", class: "svg-muted", "font-size": 11 }, svg);
    }

    // ---- R2: 10 LLMs 2B–4B × A/B × 3 semillas (Fig. 10b) ----
    function drawR2(svg) {
      const rows = data.r2;
      const mL = 182, cW = 492, x0 = 0.718, x1 = 0.76;
      const xs = (v) => mL + ((v - x0) / (x1 - x0)) * cW;
      const top = 78, rowH = 36;
      const yOf = (i) => top + i * rowH;
      const yEnd = yOf(rows.length - 1) + rowH / 2;
      legend(svg, [["A", "Config A"], ["B", "Config B"], ["tick", "media del modelo"], ["band", "Top-7"]], mL, 22);
      const [b0, b1] = data.r2Band;
      band(svg, xs(b0), xs(b1), top - 22, yEnd);
      txt("Top-7 sin diferencia (p = 0,45 · ±0,006)", { x: (xs(b0) + xs(b1)) / 2, y: top - 28, "text-anchor": "middle", class: "svg-mono svg-ink2", "font-size": 11.5, "font-weight": 700 }, svg);
      grid(svg, xs, [0.72, 0.73, 0.74, 0.75, 0.76], top - 18, yEnd + 4, 2);
      rows.forEach((r, i) => {
        const y = yOf(i);
        const g = el("g", {}, svg);
        hitRow(g, 14, y - rowH / 2, mL + cW + 60, rowH);
        txt(r.name, { x: mL - 14, y: y + 4.5, "text-anchor": "end", "font-size": 13, "font-weight": r.candidate ? 700 : 500, fill: r.top7 ? "var(--ink)" : "var(--ink-2)" }, g);
        r.runs.forEach((run) => marker(g, run.cfg, xs(run.f1), y + (run.cfg === "A" ? -6 : 6), { r: 4.4 }));
        el("line", { x1: xs(r.mean), y1: y - 14, x2: xs(r.mean), y2: y + 14, stroke: "var(--ink)", "stroke-width": 2.5 }, g);
        txt(f3(r.mean), { x: mL + cW + 12, y: y + 4.5, class: "svg-mono svg-ink2", "font-size": 11.5, "font-weight": r.granite ? 700 : 500 }, g);
        const porCfg = (c) => r.runs.filter((x) => x.cfg === c).map((x) => f3(x.f1)).join(" · ");
        bindTip(g, `${r.name} · ${r.runs.length} corridas`, [
          `Media: ${fmtComma(r.mean, 4)} (rango ${f3(r.min)}–${f3(r.max)})`,
          `A: ${porCfg("A")}`,
          `B: ${porCfg("B")}`,
          r.granite ? "Familia elegida: se escaló a Granite-4.2-8B (pestaña Final)" : (r.candidate ? "Familia candidata" : (r.top7 ? "Dentro del Top-7 indistinguible" : "Grupo inferior (ANOVA)"))
        ]);
      });
      // Granite: familia elegida para escalar a 8B
      const gi = rows.map((r, i) => (r.granite ? i : -1)).filter((i) => i >= 0);
      if (gi.length) {
        const ya = yOf(gi[0]) - 12, yb = yOf(gi[gi.length - 1]) + 12, bx = mL + cW + 62;
        el("path", { d: `M${bx - 5},${ya} H${bx} V${yb} H${bx - 5}`, fill: "none", stroke: "var(--ink-2)", "stroke-width": 1.4 }, svg);
        txt("familia elegida", { x: bx + 8, y: (ya + yb) / 2 - 3, "font-size": 11.5, "font-weight": 700, fill: "var(--ink)" }, svg);
        txt("→ Granite-4.2-8B", { x: bx + 8, y: (ya + yb) / 2 + 12, class: "svg-mono svg-ink2", "font-size": 11 }, svg);
      }
      txt("Macro-F1 de test automático · un punto por corrida · en negrita, las familias candidatas", { x: mL + cW / 2, y: yEnd + 44, "text-anchor": "middle", class: "svg-muted", "font-size": 12 }, svg);
    }

    // ---- Final: definitivos sobre el test automático ----
    function drawFinal(svg) {
      const rows = data.final;
      const mL = 200, cW = 560, x0 = 0.44, x1 = 0.78;
      const xs = (v) => mL + ((v - x0) / (x1 - x0)) * cW;
      const yOf = (i) => 84 + i * 32 + Math.floor(i / 2) * 34;
      const top = yOf(0) - 22, yEnd = yOf(rows.length - 1) + 18;
      legend(svg, [["A", "Config A"], ["B", "Config B"], ["dev", "TF-IDF: F1 de dev (sin test automático)"], ["band", "banda R2"]], mL, 22);
      const [b0, b1] = data.r2Band;
      band(svg, xs(b0), xs(b1), top - 4, yEnd);
      txt(`R2 (2B–4B): ${f3(b0)}–${f3(b1)}`, { x: (xs(b0) + xs(b1)) / 2, y: top - 10, "text-anchor": "middle", class: "svg-mono svg-ink2", "font-size": 11.5, "font-weight": 700 }, svg);
      grid(svg, xs, [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75], top, yEnd + 4, 2);
      rows.forEach((r, i) => {
        const y = yOf(i);
        const g = el("g", {}, svg);
        hitRow(g, 14, y - 13, mL + cW + 70, 26);
        if (i % 2 === 0) {
          const yc = (y + yOf(i + 1)) / 2;
          txt(r.name, { x: mL - 40, y: yc - 3, "text-anchor": "end", "font-size": 13.5, "font-weight": 700, fill: "var(--ink)" }, svg);
          txt(r.sub, { x: mL - 40, y: yc + 13, "text-anchor": "end", class: "svg-muted", "font-size": 11.5 }, svg);
          el("line", { x1: mL - 30, y1: y - 6, x2: mL - 30, y2: yOf(i + 1) + 6, stroke: "var(--line-strong)", "stroke-width": 1.2 }, svg);
        }
        txt(`Config ${r.cfg}`.replace("Config ", ""), { x: mL - 12, y: y + 4, "text-anchor": "end", class: "svg-mono svg-muted", "font-size": 11.5 }, g);
        if (r.auto) {
          const a = r.auto;
          whisker(g, r.cfg, xs(a.lo), xs(a.hi), y);
          marker(g, r.cfg, xs(a.f1), y, { r: 5.2 });
          txt(f3(a.f1), { x: xs(a.hi) + 9, y: y + 4.5, class: "svg-mono svg-ink2", "font-size": 12, "font-weight": 700 }, g);
          bindTip(g, `${r.name} · Config ${r.cfg}`, [
            `Test automático: ${fmtComma(a.f1, 4)} [IC95 ${f3(a.lo)}–${f3(a.hi)}]`,
            `Dev automático: ${f3(a.dev)}`,
            `Entrenamiento: ${fmtComma(a.min, 1)} min · ${fmtComma(a.gb, 0)} GB`
          ]);
        } else {
          marker(g, r.cfg, xs(r.devOnly), y, { r: 4.6, hollow: true, diamond: true });
          txt(`${f3(r.devOnly)} (dev)`, { x: xs(r.devOnly) + 11, y: y + 4.5, class: "svg-mono svg-ink2", "font-size": 12 }, g);
          bindTip(g, `${r.name} · Config ${r.cfg}`, [
            `Dev automático: ${f3(r.devOnly)}`,
            "Sin corrida sobre el test automático: se evaluó directamente sobre el Gold (A4-02)"
          ]);
        }
      });
      // 110M → 8B: ~70× parámetros, uno o dos puntos más
      const ga = rows.find((r) => r.name.startsWith("Granite") && r.cfg === "A").auto.f1;
      const gb = rows.find((r) => r.name.startsWith("Granite") && r.cfg === "B").auto.f1;
      const sa = rows.find((r) => r.name === "SciBERT" && r.cfg === "A").auto.f1;
      const sb = rows.find((r) => r.name === "SciBERT" && r.cfg === "B").auto.f1;
      const yt = (yOf(1) + yOf(2)) / 2 + 4;
      txt(`8B vs. 110M (~70× parámetros): +${f3(ga - sa)} (A) · +${f3(gb - sb)} (B)`, { x: xs(0.705), y: yt, "text-anchor": "end", "font-size": 12, "font-weight": 600, fill: "var(--ink)" }, svg);
      txt("Macro-F1 de test automático (etiquetas A1–A2) e IC95", { x: mL + cW / 2, y: yEnd + 44, "text-anchor": "middle", class: "svg-muted", "font-size": 12 }, svg);
      txt("Las 11 corridas: estas 4 en el test automático + 7 evaluadas directamente sobre el Gold (A4-02).", { x: mL + cW / 2, y: yEnd + 66, "text-anchor": "middle", class: "svg-muted", "font-size": 11.5 }, svg);
    }

    // ---- Gen. vs. clasif.: Qwen3-0.6B, 4 corridas ----
    function drawGen(svg) {
      const rows = data.gen;
      const mL = 172, pw = 148, gap = 28;
      const px = (k) => mL + k * (pw + gap);
      const yOf = (i) => 104 + i * 36 + (i >= 2 ? 40 : 0);
      const yTop = 78, yEnd = yOf(3) + 20;
      legend(svg, [["A", "Config A"], ["B", "Config B"], ["ci", "IC95"]], mL, 22);
      txt("CONFIG A", { x: 14, y: yOf(0) - 22, class: "svg-mono svg-muted", "font-size": 11, "font-weight": 700 }, svg);
      txt("CONFIG B", { x: 14, y: yOf(2) - 22, class: "svg-mono svg-muted", "font-size": 11, "font-weight": 700 }, svg);
      rows.forEach((r, i) => {
        txt(r.mode === "clasificador" ? "Clasificador" : "Generativo", { x: mL - 16, y: yOf(i) + 4.5, "text-anchor": "end", "font-size": 13, "font-weight": r.mode === "clasificador" ? 700 : 500, fill: "var(--ink)" }, svg);
      });
      const panels = [
        { title: "Macro-F1 test auto.", x0: 0.70, x1: 0.78, ticks: [0.72, 0.76], dec: 2, kind: "dot", val: (r) => r.f1, ci: true, lbl: (r) => f3(r.f1) },
        { title: "Macro-F1 Gold", x0: 0.40, x1: 0.48, ticks: [0.42, 0.46], dec: 2, kind: "dot", val: (r) => r.gold, lbl: (r) => f3(r.gold) },
        { title: "Latencia (ms/inst.)", x0: 0, x1: 10, ticks: [0, 5, 10], dec: 0, kind: "bar", val: (r) => r.lat, lbl: (r) => fmtComma(r.lat, 1) },
        { title: "Memoria pico (GB)", x0: 0, x1: 90, ticks: [0, 45, 90], dec: 0, kind: "bar", val: (r) => r.gb, lbl: (r) => fmtComma(r.gb, 1) + (r.gc ? "*" : "") }
      ];
      panels.forEach((p, k) => {
        const xa = px(k);
        const xs = (v) => xa + ((v - p.x0) / (p.x1 - p.x0)) * pw;
        txt(p.title, { x: xa, y: yTop - 18, "font-size": 12, "font-weight": 700, fill: "var(--ink)" }, svg);
        el("line", { x1: xa, y1: yTop - 6, x2: xa, y2: yEnd, stroke: "var(--line-strong)", "stroke-width": 1 }, svg);
        grid(svg, xs, p.ticks, yTop - 6, yEnd, p.dec);
        rows.forEach((r, i) => {
          const y = yOf(i);
          const g = el("g", {}, svg);
          hitRow(g, xa - 4, y - 14, pw + 8, 28);
          const v = p.val(r);
          if (p.kind === "bar") {
            const bw = Math.max(1, xs(v) - xs(0));
            el("path", { d: barPath(xs(0), y - 8, bw, 16), fill: colOf(r.cfg) }, g);
            let label = p.lbl(r);
            if (r.mode === "generativo") {
              const ref = rows.find((q) => q.cfg === r.cfg && q.mode === "clasificador");
              label += ` · ×${fmtComma(v / p.val(ref), 1)}`;
            }
            const w = label.length * 6.9;
            const fuera = xs(v) + 6 + w < xa + pw + gap - 4;
            if (fuera) {
              txt(label, { x: xs(v) + 6, y: y + 4.5, class: "svg-mono svg-ink2", "font-size": 11.5, "font-weight": r.mode === "generativo" ? 700 : 500 }, g);
            } else {
              // No cabe a la derecha: dentro del extremo de la barra, en blanco (APCA ≥ 59 sobre ambos tonos y temas)
              txt(label, { x: xs(v) - 6, y: y + 4.5, "text-anchor": "end", class: "svg-mono", "font-size": 11.5, "font-weight": 700, fill: "#FFFFFF" }, g);
            }
          } else {
            if (p.ci) whisker(g, r.cfg, xs(r.lo), xs(r.hi), y);
            marker(g, r.cfg, xs(v), y, { r: 5 });
            txt(p.lbl(r), { x: xs(v), y: y - 11, "text-anchor": "middle", class: "svg-mono svg-ink2", "font-size": 11 }, g);
          }
          bindTip(g, `Qwen3-0.6B · ${r.mode} · Config ${r.cfg}`, [
            `Macro-F1 test automático: ${f3(r.f1)} [IC95 ${f3(r.lo)}–${f3(r.hi)}]`,
            `Macro-F1 Gold (recalificado): ${f3(r.gold)}`,
            `Latencia: ${fmtComma(r.lat, 2)} ms/instancia · entrenamiento ${fmtComma(r.min, 1)} min`,
            `Memoria pico: ${fmtComma(r.gb, 1)} GB${r.gc ? " (con gradient checkpointing)" : ""}`
          ]);
        });
      });
      txt("Mismo F1 (diferencias de 0,005–0,022, del orden del IC95 con una corrida por variante); generar la etiqueta cuesta 2,9–4,2× más latencia.", { x: 14, y: yEnd + 42, class: "svg-muted", "font-size": 11.5 }, svg);
      txt("* Generativo B con gradient checkpointing: sin él no cabía en la B200 (180 GB), así que 31,7 vs. 34,2 GB no se comparan en igualdad.", { x: 14, y: yEnd + 62, class: "svg-muted", "font-size": 11.5 }, svg);
    }

    const DRAW = { r1: drawR1, r2: drawR2, final: drawFinal, gen: drawGen };
    const VIEWBOX = { r1: "0 0 860 520", r2: "0 0 860 520", final: "0 0 860 470", gen: "0 0 860 430" };
    let current = null;

    function show(key) {
      const k = DRAW[key] ? key : "final";
      current = k;
      buttons.forEach((b) => {
        const on = b.dataset.a4Tab === k;
        b.classList.toggle("is-active", on);
        b.setAttribute("aria-selected", String(on));
      });
      panel.setAttribute("aria-labelledby", `a4-tab-${k}`);
      Tooltip.hide();
      panel.innerHTML = "";
      const svg = el("svg", { viewBox: VIEWBOX[k], role: "img", "aria-label": TABS.find((tb) => tb.key === k).aria }, panel);
      DRAW[k](svg);
      if (options.onChange) options.onChange(k);
    }

    buttons.forEach((b) => b.addEventListener("click", () => show(b.dataset.a4Tab)));
    return { show, get tab() { return current; } };
  }

  /* ========================================================================
     7. A4-02 — Desplome Automático -> Gold (Fig. 4 animado)
     ======================================================================== */
  function renderA4Collapse(container, rows) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 860 475", role: "img", "aria-label": "Caída de modelos A4 del test automático al Gold humano" }, container);

    const mL = 190, cW = 610, startY = 70, rowH = 48;
    const xMin = 0.28, xMax = 0.78;
    const xScale = (v) => mL + ((v - xMin) / (xMax - xMin)) * cW;

    // Vertical reference lines for API benchmarks (GPT = violet, Gemini = primary cyan/blue)
    const xGpt = xScale(0.4664), xGem = xScale(0.5996);
    el("line", { x1: xGpt, y1: startY - 18, x2: xGpt, y2: startY + rows.length * rowH - 8, stroke: "var(--accent-violet)", "stroke-dasharray": "5 4", "stroke-width": 1.6 }, svg);
    txt("GPT 0s-B (0,466)", { x: xGpt, y: startY - 26, "text-anchor": "middle", class: "svg-mono", "font-size": 11.5, "font-weight": 700, fill: "var(--accent-violet)" }, svg);

    el("line", { x1: xGem, y1: startY - 18, x2: xGem, y2: startY + rows.length * rowH - 8, stroke: "var(--accent-primary)", "stroke-dasharray": "5 4", "stroke-width": 1.8 }, svg);
    txt("Gemini 0s-B (0,600)", { x: xGem, y: startY - 26, "text-anchor": "middle", class: "svg-mono", "font-size": 11.5, "font-weight": 700, fill: "var(--accent-primary)" }, svg);

    [0.30, 0.40, 0.50, 0.60, 0.70, 0.75].forEach((tk) => {
      const x = xScale(tk);
      el("line", { x1: x, y1: startY - 10, x2: x, y2: startY + rows.length * rowH - 8, class: "svg-grid" }, svg);
      txt(fmtComma(tk, 2), { x, y: startY + rows.length * rowH + 14, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 11.5 }, svg);
    });
    txt("Macro-F1: Test automático (○)  →  Test Gold humano A3 (●)", { x: mL + cW / 2, y: 454, "text-anchor": "middle", class: "svg-muted", "font-size": 12.5 }, svg);

    const animRefs = [];
    rows.forEach((r, i) => {
      const y = startY + i * rowH + 16;
      const g = el("g", {}, svg);
      txt(r.name, { x: mL - 14, y: y + 4, "text-anchor": "end", "font-size": 13.5, "font-weight": i < 2 ? 700 : 500 }, g);

      const xGold = xScale(r.goldF1);
      if (r.autoF1 == null) {
        // Sin corrida sobre el test automático (TF-IDF): solo el punto Gold, como en la Fig. 4 del informe
        const dotOnly = el("circle", { cx: xGold, cy: y, r: 7.2, fill: "var(--accent-danger)", stroke: "var(--bg)", "stroke-width": 1.6, opacity: 0 }, g);
        const lblOnly = txt(fmtComma(r.goldF1, 3), { x: xGold - 12, y: y + 4, "text-anchor": "end", class: "svg-mono", "font-size": 12, "font-weight": 700, fill: "var(--accent-danger)", opacity: 0 }, g);
        txt("solo evaluado sobre el Gold", { x: mL - 14, y: y + 20, "text-anchor": "end", class: "svg-muted", "font-size": 11 }, g);
        bindTip(g, `${r.name} (${r.params})`, [
          "Test automático: sin corrida (se evaluó directamente sobre el Gold)",
          `Dev automático: ${fmtComma(r.devF1, 3)}`,
          `Gold humano A3 (Tabla III): ${fmtComma(r.goldF1, 3)}`
        ]);
        animRefs.push({ fadeOnly: [dotOnly, lblOnly] });
        return;
      }
      const xAuto = xScale(r.autoF1);

      const bar = el("line", { x1: xAuto, y1: y, x2: xAuto, y2: y, stroke: "var(--accent-danger)", "stroke-width": 3, "stroke-linecap": "round" }, g);
      el("circle", { cx: xAuto, cy: y, r: 6.5, fill: "var(--bg)", stroke: "var(--ink-3)", "stroke-width": 2.2 }, g);
      txt(fmtComma(r.autoF1, 3), { x: xAuto + 12, y: y + 4, class: "svg-mono svg-muted", "font-size": 11.5 }, g);

      const dotGold = el("circle", { cx: xAuto, cy: y, r: 7.2, fill: "var(--accent-danger)", stroke: "var(--bg)", "stroke-width": 1.6, opacity: 0 }, g);
      const lblGold = txt(fmtComma(r.goldF1, 3), { x: xGold - 12, y: y + 4, "text-anchor": "end", class: "svg-mono", "font-size": 12, "font-weight": 700, fill: "var(--accent-danger)", opacity: 0 }, g);

      bindTip(g, `${r.name} (${r.params})`, [
        `Test automático: ${fmtComma(r.autoF1, 3)}`,
        `Gold humano A3 (Tabla III): ${fmtComma(r.goldF1, 3)}`,
        `Caída: ${fmtComma(r.goldF1 - r.autoF1, 3)} Macro-F1`
      ]);
      animRefs.push({ bar, dotGold, lblGold, xAuto, xGold });
    });

    // Brief 420ms hold so cross-dissolve settles and audience sees ~0,74 first, then 1,3s glide
    tween(1300, (t) => {
      const dotOp = Math.min(1, t * 4);
      const lblOp = t > 0.65 ? (t - 0.65) / 0.35 : 0;
      animRefs.forEach(({ bar, dotGold, lblGold, xAuto, xGold, fadeOnly }) => {
        if (fadeOnly) {
          // TF-IDF (solo Gold): aparece en su sitio, sin deslizarse
          fadeOnly[0].setAttribute("opacity", String(dotOp));
          fadeOnly[1].setAttribute("opacity", String(lblOp));
          return;
        }
        const curX = xAuto + (xGold - xAuto) * t;
        bar.setAttribute("x1", String(curX));
        dotGold.setAttribute("cx", String(curX));
        dotGold.setAttribute("opacity", String(dotOp));
        lblGold.setAttribute("opacity", String(lblOp));
      });
    }, 420);
  }

  /* ========================================================================
     8. A4-03 ★ Estrella #4 (Clímax #1) — Fig. 11 en 3 PASOS EXACTOS
     Paso 0: Panel A confirmadas
     Paso 1: Panel B corregidas SOLO marcadores huecos (○ automática)
     Paso 2: Deslizamiento animado hacia marcadores rellenos (● humana) + Δ
     ======================================================================== */
  function createFig11ThreeStep(container, rows) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 910 480", role: "img", "aria-label": "Diagnóstico causal por instancia (Fig. 11) en 3 pasos" }, container);

    const mL = 205;
    const p1X = mL, p1W = 196;
    const p2X = 446, p2W = 314;
    const deltaX = 786;
    const startY = 80, rowH = 54;

    const xConf = (pct) => p1X + ((pct - 50) / 50) * p1W; // 50%..100%
    const xCorr = (pct) => p2X + (pct / 80) * p2W;        // 0%..80%

    const yFor = (i) => startY + i * rowH + (i >= 4 ? 18 : 0);

    // Panel A group
    const gA = el("g", { class: "svg-anim" }, svg);
    txt("(a) Confirmadas (n = 1.278)", { x: p1X, y: 34, "font-size": 13.5, "font-weight": 700 }, gA);
    txt("% acierto del modelo", { x: p1X, y: 52, class: "svg-mono svg-muted", "font-size": 11 }, gA);
    [50, 75, 100].forEach((tk) => {
      const x = xConf(tk);
      el("line", { x1: x, y1: 62, x2: x, y2: 415, class: "svg-grid" }, gA);
      txt(`${tk}%`, { x, y: 434, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 11 }, gA);
    });

    // Panel B group
    const gB = el("g", { class: "svg-anim" }, svg);
    txt("(b) Corregidas (n = 1.520)", { x: p2X, y: 34, "font-size": 13.5, "font-weight": 700 }, gB);
    txt("% que coincide con la vieja (○) / humana (●)", { x: p2X, y: 52, class: "svg-mono svg-muted", "font-size": 11 }, gB);
    [0, 20, 40, 60, 80].forEach((tk) => {
      const x = xCorr(tk);
      el("line", { x1: x, y1: 62, x2: x, y2: 415, class: "svg-grid" }, gB);
      txt(`${tk}%`, { x, y: 434, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 11 }, gB);
    });

    const gDeltaHead = txt("Δ (hum − vieja)", { x: deltaX, y: 52, class: "svg-mono svg-muted", "font-size": 11, "font-weight": 700 }, gB);
    gDeltaHead.classList.add("svg-anim");

    // Separator A4 vs A5
    const sepY = yFor(3) + 34;
    el("line", { x1: 20, y1: sepY, x2: 860, y2: sepY, stroke: "var(--line-strong)", "stroke-dasharray": "5 4", "stroke-width": 1.2 }, svg);
    txt("Fine-tuning local (A4)", { x: 20, y: sepY - 8, class: "svg-mono", "font-size": 10.5, "font-weight": 700, fill: "var(--accent-warn)" }, svg);
    txt("LLMs API 0-Shot (A5)", { x: 20, y: sepY + 15, class: "svg-mono", "font-size": 10.5, "font-weight": 700, fill: "var(--accent-primary)" }, svg);

    const dynRows = [];
    rows.forEach((r, i) => {
      const y = yFor(i);
      const isA5 = r.type === "a5";
      const colMain = isA5 ? "var(--accent-primary)" : "var(--accent-warn)";
      const colSlide = isA5 ? "var(--accent-primary)" : "var(--accent-danger)";

      // Y label
      txt(r.name, { x: mL - 14, y: y + 5, "text-anchor": "end", "font-size": 12.8, "font-weight": isA5 || i === 0 ? 700 : 500 }, svg);

      // Panel A row
      const xC = xConf(r.conf);
      const rowA = el("g", {}, gA);
      el("line", { x1: p1X, y1: y, x2: xC, y2: y, stroke: colMain, "stroke-width": 4, opacity: 0.25, "stroke-linecap": "round" }, rowA);
      if (isA5) {
        el("polygon", { points: `${xC},${y - 7} ${xC + 7},${y} ${xC},${y + 7} ${xC - 7},${y}`, fill: colMain, stroke: "var(--bg)", "stroke-width": 1.4 }, rowA);
      } else {
        el("circle", { cx: xC, cy: y, r: 6.5, fill: colMain, stroke: "var(--bg)", "stroke-width": 1.4 }, rowA);
      }
      txt(`${fmtComma(r.conf, 1)}%`, { x: xC + 10, y: y + 4, class: "svg-mono", "font-size": 11.5, "font-weight": 700 }, rowA);
      bindTip(rowA, `${r.name} — Confirmadas`, [`Acierto: ${fmtComma(r.conf, 1)}% sobre 1.278 citas confirmadas`]);

      // Panel B row
      const xAuto = xCorr(r.auto);
      const xHum = xCorr(r.hum);
      const rowB = el("g", {}, gB);

      // Connector line (starts zero-length at xAuto, glides smoothly at 60fps to xHum in step 2)
      const line = el("line", { x1: xAuto, y1: y, x2: xAuto, y2: y, stroke: colSlide, "stroke-width": 3.2, "stroke-linecap": "round" }, rowB);

      // Hollow marker (old automatic label)
      const hollow = isA5
        ? el("polygon", { points: `${xAuto},${y - 7} ${xAuto + 7},${y} ${xAuto},${y + 7} ${xAuto - 7},${y}`, fill: "var(--bg)", stroke: colSlide, "stroke-width": 2.4 }, rowB)
        : el("circle", { cx: xAuto, cy: y, r: 6.8, fill: "var(--bg)", stroke: colSlide, "stroke-width": 2.4 }, rowB);
      const lblAuto = txt(`${fmtComma(r.auto, 1)}%`, { x: isA5 ? xAuto - 12 : xAuto + 12, y: y + 4, "text-anchor": isA5 ? "end" : "start", class: "svg-mono svg-muted", "font-size": 11.5 }, rowB);

      // Solid marker (human Gold label — starts at xAuto hidden, glides smoothly to xHum in step 2)
      const solidGroup = el("g", { transform: `translate(${xAuto}, ${y})`, opacity: 0 }, rowB);
      if (isA5) {
        el("polygon", { points: "0,-7.5 7.5,0 0,7.5 -7.5,0", fill: colSlide, stroke: "var(--bg)", "stroke-width": 1.6 }, solidGroup);
      } else {
        el("circle", { cx: 0, cy: 0, r: 7.2, fill: colSlide, stroke: "var(--bg)", "stroke-width": 1.6 }, solidGroup);
      }

      const lblHum = txt(`${fmtComma(r.hum, 1)}%`, {
        x: isA5 ? xHum + 12 : xHum - 12,
        y: y + 4,
        "text-anchor": isA5 ? "start" : "end",
        class: "svg-mono",
        "font-size": 12,
        "font-weight": 700,
        fill: colSlide,
        opacity: 0
      }, rowB);

      const lblDelta = txt(`${r.delta > 0 ? "+" : "−"}${fmtComma(Math.abs(r.delta), 1)} pp`, {
        x: deltaX,
        y: y + 4,
        class: "svg-mono",
        "font-size": 12.5,
        "font-weight": 700,
        fill: colSlide,
        opacity: 0
      }, rowB);

      bindTip(rowB, `${r.name} — Corregidas (n = 1.520)`, [
        `Repite la etiqueta automática descartada (○): ${fmtComma(r.auto, 1)}%`,
        `Coincide con la etiqueta humana Gold (●): ${fmtComma(r.hum, 1)}%`,
        `Otra clase: ${fmtComma(100 - r.auto - r.hum, 1)}%`,
        `Δ (humana − vieja): ${r.delta > 0 ? "+" : "−"}${fmtComma(Math.abs(r.delta), 1)} pp`
      ]);

      dynRows.push({ isA5, line, solidGroup, lblHum, lblDelta, lblAuto, hollow, xAuto, xHum, y });
    });

    // Bottom clean legend
    const legY = 466;
    el("circle", { cx: 220, cy: legY - 4, r: 5.5, fill: "var(--bg)", stroke: "var(--ink-2)", "stroke-width": 2 }, svg);
    txt("repite la etiqueta vieja (○)", { x: 232, y: legY, class: "svg-muted", "font-size": 11.5 }, svg);
    el("circle", { cx: 420, cy: legY - 4, r: 5.5, fill: "var(--ink-2)" }, svg);
    txt("coincide con la humana Gold (●)", { x: 432, y: legY, class: "svg-muted", "font-size": 11.5 }, svg);
    txt("el resto predice una 3.ª clase", { x: 640, y: legY, class: "svg-muted", "font-size": 11.5 }, svg);

    let cancelA4 = null;
    let cancelA5 = null;
    let lastStep = null;

    function applyRowProgress(row, t) {
      const curX = row.xAuto + (row.xHum - row.xAuto) * t;
      row.line.setAttribute("x2", String(curX));
      row.solidGroup.setAttribute("transform", `translate(${curX}, ${row.y})`);
      row.solidGroup.setAttribute("opacity", String(Math.min(1, t * 4)));
      const lblOp = t > 0.65 ? (t - 0.65) / 0.35 : 0;
      row.lblHum.setAttribute("opacity", String(lblOp));
      row.lblDelta.setAttribute("opacity", String(lblOp));
    }

    return {
      update(step) {
        // step 0: Panel A only; step 1: Panel B hollow markers; step 2: A4 glides left, +350ms later A5 launches right
        gA.style.opacity = step === 0 ? "1" : "0.78";
        gB.style.opacity = step === 0 ? "0.06" : "1";

        if (lastStep === step) return;
        lastStep = step;

        if (cancelA4) { cancelA4(); cancelA4 = null; }
        if (cancelA5) { cancelA5(); cancelA5 = null; }

        if (step < 2) {
          gDeltaHead.setAttribute("opacity", "0");
          dynRows.forEach((row) => applyRowProgress(row, 0));
          return;
        }

        gDeltaHead.setAttribute("opacity", "1");
        const a4Rows = dynRows.filter((r) => !r.isA5);
        const a5Rows = dynRows.filter((r) => r.isA5);
        a4Rows.forEach((row) => applyRowProgress(row, 0));
        a5Rows.forEach((row) => applyRowProgress(row, 0));

        // 1. First the 4 fine-tuned A4 models fall leftwards (1,25s)
        cancelA4 = tween(1250, (t) => {
          a4Rows.forEach((row) => applyRowProgress(row, t));
        }, 120);

        // 2. 350ms later, GPT and Gemini launch rightwards toward human Gold (1,25s)
        cancelA5 = tween(1250, (t) => {
          a5Rows.forEach((row) => applyRowProgress(row, t));
        }, 470);
      }
    };
  }

  /* ========================================================================
     9. A5-01 ★ Estrella #5A (Clímax #2) — Empate Aparente -> Salto Gold + 9 Clases
     ======================================================================== */
  function renderA5ParityBreak(container, p) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 860 470", role: "img", "aria-label": "Fin del empate aparente entre GPT-4.1-mini y Gemini-3.8-Flash tras A3" }, container);

    const baseY = 390, maxH = 300;
    const yScale = (v) => baseY - (v / 0.70) * maxH;

    // Gridlines
    [0, 0.2, 0.4, 0.6].forEach((tk) => {
      const y = yScale(tk);
      el("line", { x1: 72, y1: y, x2: 820, y2: y, class: "svg-grid" }, svg);
      txt(fmtComma(tk, 1), { x: 58, y: y + 4, "text-anchor": "end", class: "svg-mono svg-muted", "font-size": 12 }, svg);
    });
    txt("Macro-F1", { x: 24, y: 60, class: "svg-mono svg-muted", "font-size": 12, "font-weight": 700 }, svg);

    // Group titles
    txt("ANTES DE A3 (test automático)", { x: 235, y: 42, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 12.5, "font-weight": 700 }, svg);
    txt("DESPUÉS DE A3 (test Gold humano)", { x: 635, y: 42, "text-anchor": "middle", class: "svg-mono", "font-size": 12.5, "font-weight": 700, fill: "var(--accent-primary)" }, svg);
    el("line", { x1: 435, y1: 54, x2: 435, y2: baseY, stroke: "var(--line-strong)", "stroke-dasharray": "5 4" }, svg);

    const bars = [
      { x: 140, val: p.gptPanel, col: "var(--accent-violet)",  lbl: "GPT 0s-B", sub: "0,4407", op: 0.52 },
      { x: 245, val: p.gemPanel, col: "var(--accent-primary)", lbl: "Gemini 0s-B", sub: "0,4485", op: 0.52 },
      { x: 540, val: p.gptGold,  col: "var(--accent-violet)",  lbl: "GPT 0s-B", sub: "0,4664", op: 0.94 },
      { x: 645, val: p.gemGold,  col: "var(--accent-primary)", lbl: "Gemini 0s-B", sub: "0,5996", op: 1.0 }
    ];

    bars.forEach((b) => {
      const y = yScale(b.val);
      const h = baseY - y;
      const g = el("g", {}, svg);
      const rect = el("rect", { x: b.x, y: baseY, width: 82, height: 0, rx: 6, fill: b.col, opacity: b.op, class: "svg-anim" }, g);
      txt(b.sub, { x: b.x + 41, y: y - 10, "text-anchor": "middle", class: "svg-mono", "font-size": 15, "font-weight": 700, fill: b.col }, g);
      txt(b.lbl, { x: b.x + 41, y: baseY + 24, "text-anchor": "middle", "font-size": 13, "font-weight": 600 }, g);
      bindTip(g, b.lbl, [`Macro-F1: ${b.sub}`, b.x < 435 ? "Evaluado sobre test_panel (etiquetas automáticas)" : "Evaluado sobre test Gold humano A3 (N = 2.798)"]);
      requestAnimationFrame(() => {
        rect.setAttribute("y", String(y));
        rect.setAttribute("height", String(h));
      });
    });

    // Badges
    txt("Empate aparente: p = 0,436 (n.s.)", { x: 235, y: baseY + 56, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 12.5 }, svg);
    txt("Brecha real: +0,1331 Macro-F1 (p < 10⁻⁴ · Holm = SÍ)", { x: 635, y: baseY + 56, "text-anchor": "middle", class: "svg-mono", "font-size": 13, "font-weight": 700, fill: "var(--accent-gold)" }, svg);
  }

  function renderA5ByClass(container, rows) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 860 480", role: "img", "aria-label": "F1-Score por las 9 clases: GPT 0s-B vs Gemini 0s-B" }, container);

    const mL = 195, cW = 550, startY = 58, rowH = 42;
    const xScale = (v) => mL + (v / 1.0) * cW;

    [0.2, 0.4, 0.6, 0.8, 1.0].forEach((tk) => {
      const x = xScale(tk);
      el("line", { x1: x, y1: startY - 12, x2: x, y2: startY + rows.length * rowH - 8, class: "svg-grid" }, svg);
      txt(fmtComma(tk, 1), { x, y: startY + rows.length * rowH + 14, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 11.5 }, svg);
    });

    // Top inline legend (GPT = violet, Gemini = cyan/blue)
    el("circle", { cx: 210, cy: 24, r: 6.5, fill: "var(--accent-violet)" }, svg);
    txt("GPT-4.1-mini 0s-B", { x: 222, y: 28, "font-size": 12.5, "font-weight": 700, fill: "var(--accent-violet)" }, svg);
    el("circle", { cx: 388, cy: 24, r: 6.5, fill: "var(--accent-primary)" }, svg);
    txt("Gemini-3.8-Flash 0s-B (gana 9 / 9 clases)", { x: 400, y: 28, "font-size": 12.5, "font-weight": 700, fill: "var(--accent-primary)" }, svg);

    rows.forEach((r, i) => {
      const y = startY + i * rowH + 14;
      const g = el("g", {}, svg);
      txt(`${r.cls} (${r.n})`, { x: mL - 12, y: y + 4, "text-anchor": "end", "font-size": 12.8, "font-weight": r.cls === "Basis" || r.cls === "Gap" || r.cls === "Further Reading" ? 700 : 500 }, g);

      const xG = xScale(r.gpt), xM = xScale(r.gem);
      const diff = r.gem - r.gpt;
      el("line", { x1: xG, y1: y, x2: xM, y2: y, stroke: "var(--accent-primary)", "stroke-width": 3, "stroke-linecap": "round", opacity: 0.65 }, g);
      el("circle", { cx: xG, cy: y, r: 6.5, fill: "var(--accent-violet)", stroke: "var(--bg)", "stroke-width": 1.6 }, g);
      el("circle", { cx: xM, cy: y, r: 7.2, fill: "var(--accent-primary)", stroke: "var(--bg)", "stroke-width": 1.6 }, g);
      txt(`+${fmtComma(diff, 3)}`, { x: 765, y: y + 4, class: "svg-mono", "font-size": 12, "font-weight": 700, fill: diff > 0.17 ? "var(--accent-gold)" : "var(--ink-2)" }, g);

      bindTip(g, `${r.cls} (n = ${r.n})`, [
        `GPT-4.1-mini 0s-B: ${fmtComma(r.gpt, 3)}`,
        `Gemini-3.8-Flash 0s-B: ${fmtComma(r.gem, 3)}`,
        `Ventaja Gemini: +${fmtComma(diff, 3)} F1`
      ]);
    });
  }

  /* ========================================================================
     10. A5-02 — Fig. 6 Exactitud A -> B por Macro-Sección (0-Shot)
     ======================================================================== */
  function renderA5MacroSections(container, rows) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 860 480", role: "img", "aria-label": "Exactitud Config A a Config B por macro-sección citante" }, container);

    const mL = 64, cW = 760, topY = 62, botY = 378;
    const yMin = 30, yMax = 82;
    const yScale = (v) => botY - ((v - yMin) / (yMax - yMin)) * (botY - topY);

    [35, 45, 55, 65, 75].forEach((tk) => {
      const y = yScale(tk);
      el("line", { x1: mL, y1: y, x2: mL + cW, y2: y, class: "svg-grid" }, svg);
      txt(`${tk}%`, { x: mL - 10, y: y + 4, "text-anchor": "end", class: "svg-mono svg-muted", "font-size": 11.5 }, svg);
    });

    // Badge for Gemini latent thinking reduction
    el("rect", { x: 545, y: 14, width: 278, height: 34, rx: 17, fill: "var(--accent-primary-soft)", stroke: "var(--accent-primary)", "stroke-width": 1.2 }, svg);
    txt("Gemini Config B: −19,4% pensamientos", { x: 684, y: 36, "text-anchor": "middle", class: "svg-mono", "font-size": 12, "font-weight": 700, fill: "var(--accent-primary)" }, svg);

    // Legend top-left (GPT = violet, Gemini = cyan/blue)
    el("circle", { cx: 74, cy: 26, r: 5.5, fill: "var(--bg)", stroke: "var(--ink-2)", "stroke-width": 2 }, svg);
    txt("Config A (global)", { x: 86, y: 30, class: "svg-muted", "font-size": 11.5 }, svg);
    el("circle", { cx: 212, cy: 26, r: 6, fill: "var(--accent-violet)" }, svg);
    txt("GPT Config B", { x: 224, y: 30, "font-size": 11.5, "font-weight": 700, fill: "var(--accent-violet)" }, svg);
    el("circle", { cx: 332, cy: 26, r: 6, fill: "var(--accent-primary)" }, svg);
    txt("Gemini Config B", { x: 344, y: 30, "font-size": 11.5, "font-weight": 700, fill: "var(--accent-primary)" }, svg);

    const stepW = cW / rows.length;
    rows.forEach((r, i) => {
      const cx = mL + stepW * (i + 0.5);
      if (r.highlight) {
        el("rect", { x: cx - stepW * 0.44, y: topY - 6, width: stepW * 0.88, height: botY - topY + 12, rx: 8, fill: "var(--accent-violet-soft)" }, svg);
      }

      // GPT pair (left offset — violet)
      const xg = cx - 18;
      const gG = el("g", {}, svg);
      el("line", { x1: xg, y1: yScale(r.gptA), x2: xg, y2: yScale(r.gptB), stroke: "var(--accent-violet)", "stroke-width": 3 }, gG);
      el("circle", { cx: xg, cy: yScale(r.gptA), r: 6, fill: "var(--bg)", stroke: "var(--accent-violet)", "stroke-width": 2.2 }, gG);
      el("circle", { cx: xg, cy: yScale(r.gptB), r: 6.8, fill: "var(--accent-violet)", stroke: "var(--bg)", "stroke-width": 1.4 }, gG);
      bindTip(gG, `GPT 0-Shot · ${r.sec} (n = ${r.n})`, [
        `Config A: ${fmtComma(r.gptA, 2)}%`,
        `Config B: ${fmtComma(r.gptB, 2)}%`,
        `Cambio: ${r.gptB - r.gptA >= 0 ? "+" : ""}${fmtComma(r.gptB - r.gptA, 2)} pp`
      ]);

      // Gemini pair (right offset — primary cyan/blue)
      const xm = cx + 18;
      const gM = el("g", {}, svg);
      el("line", { x1: xm, y1: yScale(r.gemA), x2: xm, y2: yScale(r.gemB), stroke: "var(--accent-primary)", "stroke-width": 3 }, gM);
      el("circle", { cx: xm, cy: yScale(r.gemA), r: 6, fill: "var(--bg)", stroke: "var(--accent-primary)", "stroke-width": 2.2 }, gM);
      el("circle", { cx: xm, cy: yScale(r.gemB), r: 6.8, fill: "var(--accent-primary)", stroke: "var(--bg)", "stroke-width": 1.4 }, gM);
      bindTip(gM, `Gemini 0-Shot · ${r.sec} (n = ${r.n})`, [
        `Config A: ${fmtComma(r.gemA, 2)}%`,
        `Config B: ${fmtComma(r.gemB, 2)}%`,
        `Cambio: ${r.gemB - r.gemA >= 0 ? "+" : ""}${fmtComma(r.gemB - r.gemA, 2)} pp`
      ]);

      if (r.highlight) {
        txt("+10,28 pp", { x: xg, y: yScale(r.gptB) - 14, "text-anchor": "middle", class: "svg-mono", "font-size": 12.5, "font-weight": 700, fill: "var(--accent-violet)" }, svg);
      }

      txt(r.sec, { x: cx, y: botY + 24, "text-anchor": "middle", "font-size": 12.2, "font-weight": r.highlight ? 700 : 500 }, svg);
      txt(`n = ${r.n.toLocaleString("es-CO")}`, { x: cx, y: botY + 42, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 11 }, svg);
    });
  }

  /* ========================================================================
     11. A5-03 — Micro-gráfico de 9 Fichas (4 contaminadas en rojo)
     ======================================================================== */
  function renderA5NineChips(container, chips) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 860 440", role: "img", "aria-label": "Los 9 ejemplos del prompt 9-Shot: 4 provienen de empates 1-1-1" }, container);

    txt("MUESTREO DE LOS 9 EJEMPLOS EN TRAIN.JSONL (SEED = 42 · 1 POR CLASE)", { x: 24, y: 34, class: "svg-mono svg-muted", "font-size": 12, "font-weight": 700 }, svg);

    const colW = 258, rowH = 92, gapX = 19, gapY = 18;
    chips.forEach((c, i) => {
      const col = i % 3;
      const row = Math.floor(i / 3);
      const x = 24 + col * (colW + gapX);
      const y = 62 + row * (rowH + gapY);

      const g = el("g", {}, svg);
      const stroke = c.contaminated ? "var(--accent-danger)" : "var(--accent-gold)";
      const fill = c.contaminated ? "var(--accent-danger-soft)" : "var(--accent-gold-soft)";

      el("rect", { x, y, width: colW, height: rowH, rx: 8, fill, stroke, "stroke-width": c.contaminated ? 2.2 : 1.2 }, g);
      txt(`Ejemplo #${i + 1}`, { x: x + 16, y: y + 24, class: "svg-mono", "font-size": 11, "font-weight": 700, fill: stroke }, g);
      txt(c.cls, { x: x + 16, y: y + 52, "font-size": 16, "font-weight": 700 }, g);
      txt(c.origin, { x: x + 16, y: y + 74, class: "svg-mono svg-ink2", "font-size": 11.2 }, g);

      bindTip(g, `Ejemplar 9-Shot: ${c.cls}`, [
        `Origen en train.jsonl: ${c.origin}`,
        c.contaminated ? "Estrato empate 1-1-1: 66,06% de error (288/436) en test" : "Sin conflicto 1-1-1 en A2"
      ]);
    });

    txt("4 de 9 fichas (44,4%) inyectaron ruido directo al contexto → degradación significativa de 9-Shot frente a 0-Shot.", { x: 24, y: 416, class: "svg-serif", "font-size": 15, "font-weight": 700, fill: "var(--accent-danger)" }, svg);
  }

  /* ========================================================================
     12A. A6 — Tablero Global A4 vs. A5 (14 variantes Tabla III en 7 filas A/B)
     ======================================================================== */
  function renderA6GlobalBoard(container, families) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 880 512", role: "img", "aria-label": "Tablero global A6: las 14 variantes de Tabla III comparando Config A y Config B" }, container);

    const trackL = 178, trackR = 542;
    const xMin = 0.30, xMax = 0.63;
    const xScale = (v) => trackL + ((v - xMin) / (xMax - xMin)) * (trackR - trackL);

    const colF1A = 610;
    const colF1B = 682;
    const colAcc = 760;
    const colEce = 846;

    // Top scannable legend strip
    el("circle", { cx: 24, cy: 18, r: 5.5, fill: "var(--bg)", stroke: "var(--ink-2)", "stroke-width": 2.4 }, svg);
    txt("Config A (global)", { x: 36, y: 22, "font-size": 11.8, "font-weight": 600, fill: "var(--ink-2)" }, svg);

    el("circle", { cx: 172, cy: 18, r: 5.5, fill: "var(--accent-primary)" }, svg);
    txt("Config B (local Top-3)", { x: 184, y: 22, "font-size": 11.8, "font-weight": 700, fill: "var(--accent-primary)" }, svg);

    txt("★", { x: 356, y: 23, "font-size": 14, "font-weight": 700, fill: "var(--accent-gold)" }, svg);
    txt("Ganador global", { x: 374, y: 22, "font-size": 11.8, "font-weight": 700, fill: "var(--accent-gold)" }, svg);

    txt("Negrita = mejor A / mejor B por familia", { x: 846, y: 22, "text-anchor": "end", class: "svg-mono svg-muted", "font-size": 10.8 }, svg);

    // Column headers
    const headY = 52;
    txt("Macro-F1 Gold (Config A ○ vs. Config B ●)", { x: (trackL + trackR) / 2, y: headY, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 11, "font-weight": 700 }, svg);
    txt("F1 (A)", { x: colF1A, y: headY, "text-anchor": "end", class: "svg-mono svg-muted", "font-size": 11, "font-weight": 700 }, svg);
    txt("F1 (B)", { x: colF1B, y: headY, "text-anchor": "end", class: "svg-mono", "font-size": 11, "font-weight": 700, fill: "var(--accent-primary)" }, svg);
    txt("Acc (B)", { x: colAcc, y: headY, "text-anchor": "end", class: "svg-mono svg-muted", "font-size": 11, "font-weight": 700 }, svg);
    txt("ECE ↓", { x: colEce, y: headY, "text-anchor": "end", class: "svg-mono svg-muted", "font-size": 11, "font-weight": 700 }, svg);

    el("line", { x1: 18, y1: 62, x2: 856, y2: 62, class: "svg-axis" }, svg);

    // Vertical gridlines across dumbbell track
    const gridTop = 66, gridBot = 468;
    [0.35, 0.40, 0.45, 0.50, 0.55, 0.60].forEach((tk) => {
      const x = xScale(tk);
      el("line", { x1: x, y1: gridTop, x2: x, y2: gridBot, class: "svg-grid" }, svg);
      txt(fmtComma(tk, 2), { x, y: gridBot + 20, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 11.2 }, svg);
    });

    // Render 3 visually separated families
    let curY = 82;
    const rowStep = 39;
    const familyGap = 20;

    families.forEach((fam, fIdx) => {
      if (fIdx > 0) {
        el("line", { x1: 18, y1: curY - 14, x2: 856, y2: curY - 14, stroke: "var(--line-strong)", "stroke-dasharray": "4 4", "stroke-width": 1 }, svg);
      }

      // Family accent pill & label
      const famStartY = curY;
      const famH = fam.rows.length * rowStep + 14;
      el("rect", { x: 18, y: famStartY - 10, width: 3.5, height: famH - 4, rx: 2, fill: `var(${fam.accentVar})` }, svg);
      txt(fam.family.toUpperCase(), { x: 30, y: famStartY + 2, class: "svg-mono", "font-size": 10.6, "font-weight": 700, fill: `var(${fam.accentVar})` }, svg);

      curY += 24;

      fam.rows.forEach((r) => {
        const y = curY;
        const g = el("g", {}, svg);

        if (r.globalWinner) {
          el("rect", { x: 24, y: y - 18, width: 834, height: 36, rx: 6, fill: "var(--accent-primary-soft)" }, g);
        }

        // Row model label
        txt(r.globalWinner ? `★ ${r.name}` : r.name, {
          x: 162, y: y + 4,
          "text-anchor": "end",
          "font-size": 13,
          "font-weight": r.globalWinner || (r.bestA && r.bestB) ? 700 : 500,
          fill: r.globalWinner ? "var(--accent-primary)" : "var(--ink)"
        }, g);

        // Dumbbell A -> B
        const xA = xScale(r.f1A);
        const xB = xScale(r.f1B);
        const bImproved = r.f1B >= r.f1A;
        const segCol = bImproved ? "var(--accent-gold)" : "var(--accent-danger)";

        el("line", {
          x1: xA, y1: y, x2: xB, y2: y,
          stroke: segCol,
          "stroke-width": Math.abs(xB - xA) < 4 ? 2 : 3.2,
          "stroke-linecap": "round",
          opacity: bImproved ? 0.9 : 0.72
        }, g);

        // Config A hollow marker (slate/ink-2)
        el("circle", {
          cx: xA, cy: y, r: r.bestA ? 6.4 : 5.6,
          fill: "var(--bg)",
          stroke: r.bestA ? "var(--ink)" : "var(--ink-2)",
          "stroke-width": r.bestA ? 2.6 : 2.1
        }, g);

        // Config B solid marker (primary cyan/blue; halo on global winner)
        if (r.globalWinner) {
          el("circle", { cx: xB, cy: y, r: 12.5, fill: "var(--accent-gold-soft)", stroke: "var(--accent-gold)", "stroke-width": 1.2 }, g);
        }
        el("circle", {
          cx: xB, cy: y, r: r.globalWinner ? 7.4 : (r.bestB ? 6.6 : 5.8),
          fill: "var(--accent-primary)",
          stroke: "var(--bg)",
          "stroke-width": 1.6
        }, g);

        // Right-hand scannable numeric columns (bold for family winner in A / winner in B)
        txt(fmtComma(r.f1A, 3), {
          x: colF1A, y: y + 4, "text-anchor": "end",
          class: "svg-mono", "font-size": 12.2,
          "font-weight": r.bestA ? 700 : 400,
          fill: r.bestA ? "var(--ink)" : "var(--ink-3)"
        }, g);

        txt(r.globalWinner ? `★ ${fmtComma(r.f1B, 3)}` : fmtComma(r.f1B, 3), {
          x: colF1B, y: y + 4, "text-anchor": "end",
          class: "svg-mono", "font-size": r.globalWinner ? 12.8 : 12.2,
          "font-weight": r.bestB ? 700 : 400,
          fill: r.globalWinner ? "var(--accent-gold)" : (r.bestB ? "var(--accent-primary)" : "var(--ink-2)")
        }, g);

        txt(`${fmtComma(r.accB, 1)}%`, {
          x: colAcc, y: y + 4, "text-anchor": "end",
          class: "svg-mono", "font-size": 12,
          "font-weight": r.globalWinner ? 700 : 400,
          fill: r.globalWinner ? "var(--ink)" : "var(--ink-2)"
        }, g);

        const eceTxt = r.eceB === null ? "—" : fmtComma(r.eceB, 3);
        const eceCol = r.eceB === null ? "var(--ink-3)" : (r.eceB < 0.15 ? "var(--accent-gold)" : "var(--accent-danger)");
        txt(eceTxt, {
          x: colEce, y: y + 4, "text-anchor": "end",
          class: "svg-mono", "font-size": 12,
          "font-weight": r.eceB !== null && r.eceB < 0.15 ? 700 : 500,
          fill: eceCol
        }, g);

        const deltaF1 = r.f1B - r.f1A;
        bindTip(g, `${r.name} · Config A vs. Config B`, [
          `Config A: F1 = ${fmtComma(r.f1A, 4)} · Acc = ${fmtComma(r.accA, 2)}%` + (r.eceA !== null ? ` · ECE = ${fmtComma(r.eceA, 4)}` : " · ECE = —"),
          `Config B: F1 = ${fmtComma(r.f1B, 4)} · Acc = ${fmtComma(r.accB, 2)}%` + (r.eceB !== null ? ` · ECE = ${fmtComma(r.eceB, 4)}` : " · ECE = —"),
          `Diferencia B − A: ${deltaF1 >= 0 ? "+" : "−"}${fmtComma(Math.abs(deltaF1), 4)} Macro-F1`
        ]);

        curY += rowStep;
      });

      curY += familyGap;
    });
  }

  /* ========================================================================
     12B. A6-01 — Curvas de Fiabilidad y Revisión Selectiva (en Q&A)
     ======================================================================== */
  function renderA6Calibration(container, cal) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 860 470", role: "img", "aria-label": "Curvas de fiabilidad: Gemini 0s-B vs GPT 0s-B" }, container);

    const mL = 78, mR = 540, topY = 44, botY = 398;
    const xScale = (v) => mL + ((v - 0.35) / 0.67) * (mR - mL);
    const yScale = (v) => botY - (v / 1.0) * (botY - topY);

    // Gridlines
    [0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach((tk) => {
      const y = yScale(tk);
      el("line", { x1: mL, y1: y, x2: mR, y2: y, class: "svg-grid" }, svg);
      txt(`${Math.round(tk * 100)}%`, { x: mL - 10, y: y + 4, "text-anchor": "end", class: "svg-mono svg-muted", "font-size": 11.5 }, svg);
    });
    [0.4, 0.6, 0.8, 1.0].forEach((tk) => {
      const x = xScale(tk);
      el("line", { x1: x, y1: topY, x2: x, y2: botY, class: "svg-grid" }, svg);
      txt(fmtComma(tk, 1), { x, y: botY + 20, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 11.5 }, svg);
    });

    // Ideal diagonal
    el("line", { x1: xScale(0.35), y1: yScale(0.35), x2: xScale(1.0), y2: yScale(1.0), stroke: "var(--ink-3)", "stroke-dasharray": "6 5", "stroke-width": 1.6 }, svg);
    txt("Calibración ideal", { x: xScale(0.62), y: yScale(0.69), class: "svg-mono svg-muted", "font-size": 11, transform: `rotate(-37 ${xScale(0.62)} ${yScale(0.69)})` }, svg);

    // Helper to draw reliability polyline + bubbles
    function drawSeries(series, col) {
      const pts = series.bins.filter((b) => b.n >= 20);
      const d = pts.map((b, i) => `${i === 0 ? "M" : "L"} ${xScale(b.conf)} ${yScale(b.acc)}`).join(" ");
      el("path", { d, fill: "none", stroke: col, "stroke-width": 3, "stroke-linecap": "round" }, svg);
      pts.forEach((b) => {
        const r = Math.max(5.5, Math.min(13, Math.sqrt(b.n) * 0.34));
        const c = el("circle", { cx: xScale(b.conf), cy: yScale(b.acc), r, fill: col, stroke: "var(--bg)", "stroke-width": 1.8, opacity: 0.92 }, svg);
        bindTip(c, `${series.label} · Bin ${b.bin}`, [
          `Instancias: n = ${b.n.toLocaleString("es-CO")}`,
          `Confianza media: ${fmtComma(b.conf * 100, 1)}%`,
          `Exactitud real: ${fmtComma(b.acc * 100, 1)}%`
        ]);
      });
    }

    drawSeries(cal.gpt, "var(--accent-danger)");
    drawSeries(cal.gem, "var(--accent-gold)");

    txt("Confianza reportada (máx. en Δ⁸)", { x: (mL + mR) / 2, y: 448, "text-anchor": "middle", class: "svg-muted", "font-size": 12.5 }, svg);
    txt("Exactitud empírica", { x: 22, y: 225, "text-anchor": "middle", class: "svg-muted", "font-size": 12.5, transform: "rotate(-90 22 225)" }, svg);

    // Right side callout: Selective Review explanation
    const rx = 576;
    txt("USO EN PRODUCCIÓN", { x: rx, y: 66, class: "svg-mono svg-muted", "font-size": 11.5, "font-weight": 700 }, svg);

    el("rect", { x: rx, y: 84, width: 262, height: 132, rx: 8, fill: "var(--accent-gold-soft)", stroke: "var(--accent-gold)", "stroke-width": 1.6 }, svg);
    txt("Gemini 0s-B · ECE = 0,1158", { x: rx + 14, y: 108, class: "svg-mono", "font-size": 13, "font-weight": 700, fill: "var(--accent-gold)" }, svg);
    txt("Confianza en aciertos: 0,829", { x: rx + 14, y: 136, class: "svg-mono", "font-size": 12.5 }, svg);
    txt("Confianza en errores:  0,737", { x: rx + 14, y: 158, class: "svg-mono", "font-size": 12.5 }, svg);
    txt("Bin ≥0,9 (n=516): 92,4% real", { x: rx + 14, y: 184, class: "svg-mono", "font-size": 12, "font-weight": 700, fill: "var(--accent-gold)" }, svg);
    txt("→ Permite revisión selectiva", { x: rx + 14, y: 204, "font-size": 12.5, "font-weight": 700 }, svg);

    el("rect", { x: rx, y: 236, width: 262, height: 122, rx: 8, fill: "var(--accent-danger-soft)", stroke: "var(--accent-danger)", "stroke-width": 1.4 }, svg);
    txt("GPT 0s-B · ECE = 0,3360", { x: rx + 14, y: 260, class: "svg-mono", "font-size": 13, "font-weight": 700, fill: "var(--accent-danger)" }, svg);
    txt("Bin ≥0,9 (n=1.270):", { x: rx + 14, y: 288, class: "svg-mono", "font-size": 12.5 }, svg);
    txt("Reporta 98,1% · Acierta 59,8%", { x: rx + 14, y: 310, class: "svg-mono", "font-size": 12.5, "font-weight": 700, fill: "var(--accent-danger)" }, svg);
    txt("→ No discrimina para umbral", { x: rx + 14, y: 340, "font-size": 12.5, "font-weight": 600 }, svg);
  }

  /* ========================================================================
     13. A7-01 ★ Estrella #5B — Relación Precisión–Costo (Sin solapamientos)
     ======================================================================== */
  function createCostQualityChart(container, pts) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 860 475", role: "img", "aria-label": "Relación costo neto por 10.000 citas vs Macro-F1 Gold" }, container);

    const mL = 78, mR = 824, topY = 42, botY = 394;
    const xScale = (cost) => mL + (cost / 58) * (mR - mL);
    const yScale = (f1) => botY - ((f1 - 0.38) / 0.25) * (botY - topY);

    // Gridlines
    [0.40, 0.45, 0.50, 0.55, 0.60].forEach((tk) => {
      const y = yScale(tk);
      el("line", { x1: mL, y1: y, x2: mR, y2: y, class: "svg-grid" }, svg);
      txt(fmtComma(tk, 2), { x: mL - 10, y: y + 4, "text-anchor": "end", class: "svg-mono svg-muted", "font-size": 11.5 }, svg);
    });
    [0, 10, 20, 30, 40, 50].forEach((tk) => {
      const x = xScale(tk);
      el("line", { x1: x, y1: topY, x2: x, y2: botY, class: "svg-grid" }, svg);
      txt(`$${tk}`, { x, y: botY + 20, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 11.5 }, svg);
    });
    txt("Costo neto proyectado por 10.000 citas (USD)", { x: (mL + mR) / 2, y: 446, "text-anchor": "middle", class: "svg-muted", "font-size": 12.5 }, svg);
    txt("Macro-F1 Gold", { x: 22, y: 220, "text-anchor": "middle", class: "svg-muted", "font-size": 12.5, transform: "rotate(-90 22 220)" }, svg);

    // Dominated region polygon (below/right of 0-shot frontier)
    const frontierPts = pts.filter((p) => p.optimal).sort((a, b) => a.cost - b.cost);
    const polyPts = [
      `${xScale(frontierPts[0].cost)},${botY}`,
      ...frontierPts.map((p) => `${xScale(p.cost)},${yScale(p.f1)}`),
      `${mR},${yScale(frontierPts[frontierPts.length - 1].f1)}`,
      `${mR},${botY}`
    ].join(" ");
    const domZone = el("polygon", { points: polyPts, fill: "var(--accent-danger-soft)", opacity: 0.72, class: "svg-anim" }, svg);
    const domLbl = txt("ZONA DESCARTADA (más caras y menos precisas)", { x: xScale(36), y: yScale(0.455), "text-anchor": "middle", class: "svg-mono", "font-size": 12, "font-weight": 700, fill: "var(--accent-danger)" }, svg);

    // Optimal curve group (revealed in step 2)
    const optGroup = el("g", { class: "svg-anim" }, svg);
    const curveD = frontierPts.map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.cost)} ${yScale(p.f1)}`).join(" ");
    el("path", { d: curveD, fill: "none", stroke: "var(--accent-gold)", "stroke-width": 3, "stroke-dasharray": "6 4" }, optGroup);

    // Explicit non-overlapping label offsets for every single point
    const offsets = {
      gpt0a: { dx: 12,  dy: 16,  anchor: "start" },
      gpt0b: { dx: 12,  dy: -10, anchor: "start" },
      gpt9a: { dx: 10,  dy: 18,  anchor: "start" },
      gpt9b: { dx: 14,  dy: 4,   anchor: "start" },
      gem0a: { dx: -14, dy: -6,  anchor: "end" },
      gem0b: { dx: 12,  dy: -16, anchor: "start" },
      gem9a: { dx: -8,  dy: 24,  anchor: "end" },
      gem9b: { dx: 8,   dy: 24,  anchor: "end" }
    };

    // Draw dominated 9-Shot points
    pts.filter((p) => !p.optimal).forEach((p) => {
      const x = xScale(p.cost), y = yScale(p.f1);
      const off = offsets[p.id];
      const g = el("g", {}, svg);
      el("rect", { x: x - 6, y: y - 6, width: 12, height: 12, rx: 2, fill: "var(--bg)", stroke: "var(--accent-danger)", "stroke-width": 2.2 }, g);
      txt(`${p.label} ($${fmtComma(p.cost, 2)})`, { x: x + off.dx, y: y + off.dy, "text-anchor": off.anchor, class: "svg-mono svg-ink2", "font-size": 11.5 }, g);
      bindTip(g, `${p.label} (Descartada)`, [
        `Macro-F1 Gold: ${fmtComma(p.f1, 4)} · Acc: ${fmtComma(p.acc, 2)}%`,
        `Costo neto / 10k: $${fmtComma(p.cost, 2)}`,
        `ECE: ${fmtComma(p.ece, 4)} · p50: ${fmtComma(p.p50 / 1000, 2)} s`
      ]);
    });

    // Draw optimal 0-Shot points (GPT = violet, Gemini = cyan/blue)
    frontierPts.forEach((p) => {
      const x = xScale(p.cost), y = yScale(p.f1);
      const off = offsets[p.id];
      const col = p.family === "gem" ? "var(--accent-primary)" : "var(--accent-violet)";
      const g = el("g", {}, optGroup);
      if (p.star) {
        el("circle", { cx: x, cy: y, r: 15, fill: "var(--accent-primary-soft)" }, g);
      }
      el("circle", { cx: x, cy: y, r: p.star ? 8.5 : 6.8, fill: col, stroke: "var(--bg)", "stroke-width": 2 }, g);
      txt(
        p.star ? `★ ${p.label} ($${fmtComma(p.cost, 2)} · 0,600)` : `${p.label} ($${fmtComma(p.cost, 2)})`,
        { x: x + off.dx, y: y + off.dy, "text-anchor": off.anchor, class: "svg-mono", "font-size": p.star ? 13 : 11.8, "font-weight": p.star || p.id === "gpt0b" ? 700 : 600, fill: col },
        g
      );
      bindTip(g, `${p.label} (Óptimo 0-Shot)`, [
        `Macro-F1 Gold: ${fmtComma(p.f1, 4)} · Acc: ${fmtComma(p.acc, 2)}%`,
        `Costo neto / 10k: $${fmtComma(p.cost, 2)}`,
        `ECE: ${fmtComma(p.ece, 4)} · Latencia p50: ${fmtComma(p.p50 / 1000, 2)} s`
      ]);
    });

    // Clean legend placed in open bottom-right corner (zero collision with any point!)
    const lgX = 560, lgY = 330;
    el("circle", { cx: lgX, cy: lgY, r: 5.5, fill: "var(--accent-primary)" }, svg);
    txt("0-Shot óptimos (mejor relación)", { x: lgX + 12, y: lgY + 4, "font-size": 11.5, "font-weight": 600 }, svg);
    el("rect", { x: lgX - 5, y: lgY + 18, width: 10, height: 10, rx: 2, fill: "var(--bg)", stroke: "var(--accent-danger)", "stroke-width": 2 }, svg);
    txt("9-Shot descartadas", { x: lgX + 12, y: lgY + 27, class: "svg-muted", "font-size": 11.5 }, svg);

    return {
      update(showOptimal) {
        domZone.style.opacity = showOptimal ? "0.45" : "0.85";
        domLbl.style.opacity = showOptimal ? "0.75" : "1";
        optGroup.style.opacity = showOptimal ? "1" : "0.14";
      }
    };
  }

  /* ========================================================================
     14. A8-01 — Arquitectura Cloud Reproducible AWS EC2 (4 Etapas)
     ======================================================================== */
  function renderA8Architecture(container) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 880 470", role: "img", "aria-label": "Arquitectura de inferencia y despliegue cloud en AWS EC2" }, container);

    const defs = el("defs", {}, svg);
    const m = el("marker", { id: "a8-arr", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, defs);
    el("path", { d: "M 0 1 L 10 5 L 0 9 z", fill: "var(--ink-3)" }, m);

    txt("AWS EC2 (t3.large · 8 GiB RAM) · Terraform + Docker Compose · Puerto SSH cerrado", { x: 20, y: 26, class: "svg-mono", "font-size": 11.5, "font-weight": 700, fill: "var(--accent-gold)" }, svg);

    // Outer EC2 dashed boundary
    el("rect", { x: 18, y: 40, width: 844, height: 292, rx: 12, fill: "var(--surface-tint)", stroke: "var(--line-strong)", "stroke-dasharray": "6 4", "stroke-width": 1.4 }, svg);

    // Edge nodes: Client -> Caddy -> Nginx -> FastAPI
    const nodes = [
      { x: 34,  y: 96, w: 114, h: 74, title: "Usuario", sub: "Navegador HTTPS", col: "var(--ink-2)", fill: "var(--surface)" },
      { x: 170, y: 96, w: 114, h: 74, title: "Caddy", sub: "TLS automático", col: "var(--accent-gold)", fill: "var(--accent-gold-soft)" },
      { x: 306, y: 96, w: 126, h: 74, title: "Nginx Gateway", sub: "Rate limit 6 r/m", col: "var(--accent-warn)", fill: "var(--accent-warn-soft)" }
    ];
    nodes.forEach((n, idx) => {
      const g = el("g", {}, svg);
      el("rect", { x: n.x, y: n.y, width: n.w, height: n.h, rx: 8, fill: n.fill, stroke: n.col, "stroke-width": 1.6 }, g);
      txt(n.title, { x: n.x + n.w / 2, y: n.y + 32, "text-anchor": "middle", "font-size": 13, "font-weight": 700 }, g);
      txt(n.sub, { x: n.x + n.w / 2, y: n.y + 54, "text-anchor": "middle", class: "svg-mono svg-ink2", "font-size": 10.4 }, g);
      if (idx < nodes.length - 1) {
        el("line", { x1: n.x + n.w, y1: n.y + 37, x2: nodes[idx + 1].x - 2, y2: n.y + 37, stroke: "var(--ink-3)", "stroke-width": 1.6, "marker-end": "url(#a8-arr)" }, svg);
      }
    });
    el("line", { x1: 432, y1: 133, x2: 454, y2: 133, stroke: "var(--ink-3)", "stroke-width": 1.6, "marker-end": "url(#a8-arr)" }, svg);

    // FastAPI 4-stage pipeline box
    el("rect", { x: 458, y: 60, width: 386, height: 252, rx: 10, fill: "var(--surface)", stroke: "var(--accent-primary)", "stroke-width": 1.8 }, svg);
    txt("FastAPI · Pipeline de Inferencia (4 etapas)", { x: 476, y: 86, "font-size": 13.5, "font-weight": 700, fill: "var(--accent-primary)" }, svg);

    const pipe = [
      "1. Normalización (<cite>, [OTHERCIT])",
      "2. Resolución citada (SHA-256 · ar5iv · PDF)",
      "3. Chunking ≤110 pal. + SciBERT Top-3 (Ec. 1)",
      "4. Inferencia: TF-IDF · SciBERT · Flash-Lite"
    ];
    pipe.forEach((p, idx) => {
      const y = 104 + idx * 42;
      el("rect", { x: 476, y, width: 350, height: 32, rx: 5, fill: idx === 3 ? "var(--accent-primary-soft)" : "var(--surface-tint)" }, svg);
      txt(p, { x: 488, y: y + 21, class: "svg-mono", "font-size": 10.8, "font-weight": idx === 3 ? 700 : 500 }, svg);
    });

    // SQLite metrics inside EC2
    const sq = el("g", {}, svg);
    el("rect", { x: 34, y: 214, width: 398, height: 96, rx: 8, fill: "var(--surface)", stroke: "var(--accent-violet)", "stroke-width": 1.5 }, sq);
    txt("Observabilidad y Privacidad (SQLite + UI)", { x: 50, y: 240, "font-size": 13.2, "font-weight": 700, fill: "var(--accent-violet)" }, sq);
    txt("• Latencia, tokens y costos sin guardar texto", { x: 50, y: 266, class: "svg-ink2", "font-size": 11.8 }, sq);
    txt("• Separa MLflow histórico de métricas en vivo", { x: 50, y: 288, class: "svg-ink2", "font-size": 11.8 }, sq);
    el("line", { x1: 458, y1: 262, x2: 436, y2: 262, stroke: "var(--ink-3)", "stroke-width": 1.5, "marker-end": "url(#a8-arr)" }, svg);

    // Bottom external AWS services + optional Granite CPU INT8 worker
    const bottom = [
      { x: 18,  w: 252, title: "Amazon S3 Privado", sub: "Artefactos TF-IDF A/B y SciBERT A/B", col: "var(--accent-secondary)" },
      { x: 286, w: 246, title: "AWS SSM (0 SSH)", sub: "Parameter Store + Run Command", col: "var(--accent-gold)" },
      { x: 548, w: 314, title: "Worker Granite CPU INT8 (Opcional)", sub: "14 GiB RAM · hot-swap LoRA (no en t3.large)", col: "var(--accent-warn)", dashed: true }
    ];
    bottom.forEach((b) => {
      const g = el("g", {}, svg);
      el("rect", {
        x: b.x, y: 360, width: b.w, height: 82, rx: 8,
        fill: b.dashed ? "var(--accent-warn-soft)" : "var(--surface-tint)",
        stroke: b.col,
        "stroke-width": 1.6,
        "stroke-dasharray": b.dashed ? "5 4" : "none"
      }, g);
      txt(b.title, { x: b.x + 14, y: 390, "font-size": 12.8, "font-weight": 700, fill: b.col }, g);
      txt(b.sub, { x: b.x + 14, y: 416, class: "svg-mono svg-ink2", "font-size": 10.4 }, g);
      bindTip(g, b.title, [b.sub]);
    });
  }


  /* ========================================================================
     15. A8 — Flujo continuo de inferencia en tiempo real por scroll (3 pasos)
     Paso 0 (Beat 27): Entrada única -> caja EvidenCite
     Paso 1 (Beat 28): Contenedor gris se expande -> Config A arriba || Config B abajo (k chunks -> SciBERT -> Coseno -> R1/R2/R3)
     Paso 2 (Beat 29): Acordeón compacta recuperación -> 2 Clasificadores (dentro del gris) -> 2×9 barras Δ⁸ (afuera a la derecha)
     ======================================================================== */
  function createA8ScrollFlow(container) {
    container.innerHTML = "";
    const ex = window.EVIDENCITE_DATA.inst04783;
    const svg = el("svg", {
      viewBox: "0 0 1000 590",
      class: "a8-flow-svg",
      role: "img",
      "aria-label": "Flujo interactivo de inferencia en tiempo real de EvidenCite en 3 pasos"
    }, container);

    // Arrow markers
    const defs = el("defs", {}, svg);
    [
      { id: "a8-m-n", col: "var(--ink-3)" },
      { id: "a8-m-a", col: "var(--accent-secondary)" },
      { id: "a8-m-b", col: "var(--accent-primary)" },
      { id: "a8-m-v", col: "var(--accent-violet)" }
    ].forEach((m) => {
      const mk = el("marker", { id: m.id, viewBox: "0 0 10 10", refX: 8.5, refY: 5, markerWidth: 6.5, markerHeight: 6.5, orient: "auto" }, defs);
      el("path", { d: "M 0 1 L 9 5 L 0 9 z", fill: m.col }, mk);
    });

    function lerp(a, b, t) { return a + (b - a) * t; }
    function setBox(rect, b) {
      rect.setAttribute("x", String(b.x));
      rect.setAttribute("y", String(b.y));
      rect.setAttribute("width", String(b.w));
      rect.setAttribute("height", String(b.h));
    }

    // ------------------------------------------------------------------
    // LAYER 1: Pale-gray EvidenCite outer container (morphs across S0 -> S1 -> S2)
    // ------------------------------------------------------------------
    const gEvid = el("g", {}, svg);
    const evidRect = el("rect", {
      x: 492, y: 162, width: 438, height: 250, rx: 16,
      fill: "var(--surface-tint)",
      stroke: "var(--line-strong)",
      "stroke-width": 1.8
    }, gEvid);

    const evidHeader = txt("EVIDENCITE · PIPELINE DE REPRESENTACIÓN", {
      x: 224, y: 48, class: "svg-mono svg-muted", "font-size": 11.5, "font-weight": 700, opacity: 0
    }, gEvid);

    // S0 centered intake content inside evidRect
    const gS0Intake = el("g", {}, gEvid);
    txt("EvidenCite", { x: 711, y: 222, "text-anchor": "middle", "font-size": 27, "font-weight": 700 }, gS0Intake);
    txt("Normalización + resolución de referencia", { x: 711, y: 254, "text-anchor": "middle", "font-size": 15.5, "font-weight": 600, fill: "var(--ink-2)" }, gS0Intake);
    el("line", { x1: 556, y1: 278, x2: 866, y2: 278, stroke: "var(--line-strong)", "stroke-dasharray": "4 4" }, gS0Intake);
    // Two preview badges inside S0 intake box
    el("rect", { x: 528, y: 298, width: 174, height: 48, rx: 8, fill: "var(--accent-secondary-soft)", stroke: "var(--accent-secondary)", "stroke-width": 1.3 }, gS0Intake);
    txt("CONFIG A · GLOBAL", { x: 615, y: 318, "text-anchor": "middle", class: "svg-mono", "font-size": 10.5, "font-weight": 700, fill: "var(--accent-secondary)" }, gS0Intake);
    txt("Título + resumen", { x: 615, y: 336, "text-anchor": "middle", "font-size": 12, fill: "var(--ink-2)" }, gS0Intake);

    el("rect", { x: 718, y: 298, width: 174, height: 48, rx: 8, fill: "var(--accent-primary-soft)", stroke: "var(--accent-primary)", "stroke-width": 1.3 }, gS0Intake);
    txt("CONFIG B · LOCAL", { x: 805, y: 318, "text-anchor": "middle", class: "svg-mono", "font-size": 10.5, "font-weight": 700, fill: "var(--accent-primary)" }, gS0Intake);
    txt("Top-3 por SciBERT", { x: 805, y: 336, "text-anchor": "middle", "font-size": 12, fill: "var(--ink-2)" }, gS0Intake);

    txt("Misma cita, dos formas de representar el citado", { x: 711, y: 382, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 11.5 }, gS0Intake);

    // ------------------------------------------------------------------
    // LAYER 2: Config A top lane (morphs from S1 wide -> S2 compact)
    // ------------------------------------------------------------------
    const gLaneA = el("g", { opacity: 0 }, svg);
    const rectA = el("rect", {
      x: 228, y: 66, width: 732, height: 114, rx: 12,
      fill: "var(--accent-secondary-soft)",
      stroke: "var(--accent-secondary)",
      "stroke-width": 1.6
    }, gLaneA);

    const gAFull = el("g", { opacity: 0 }, gLaneA);
    txt("CONFIG A · REPRESENTACIÓN GLOBAL (SIN RECUPERACIÓN)", { x: 248, y: 92, class: "svg-mono", "font-size": 11.5, "font-weight": 700, fill: "var(--accent-secondary)" }, gAFull);
    txt("Contexto citante (fijo)  +  Título y Resumen del artículo citado", { x: 248, y: 122, "font-size": 16, "font-weight": 700 }, gAFull);
    txt("“Efficient mean estimation with pure differential privacy…” · vista global estática para toda mención", { x: 248, y: 150, "font-size": 12.5, fill: "var(--ink-2)" }, gAFull);

    const gACompact = el("g", { opacity: 0 }, gLaneA);
    txt("CONFIG A · GLOBAL", { x: 176, y: 100, class: "svg-mono", "font-size": 11, "font-weight": 700, fill: "var(--accent-secondary)" }, gACompact);
    txt("Contexto citante +", { x: 176, y: 132, "font-size": 15, "font-weight": 700 }, gACompact);
    txt("Título + Resumen", { x: 176, y: 156, "font-size": 15, "font-weight": 700 }, gACompact);
    txt("Vista global estática (sin pasajes)", { x: 176, y: 192, "font-size": 12, fill: "var(--ink-2)" }, gACompact);

    // ------------------------------------------------------------------
    // LAYER 3: Config B bottom lane (frame + detailed retrieval + compact pill + persistent Top-3)
    // ------------------------------------------------------------------
    const gLaneB = el("g", { opacity: 0 }, svg);
    const frameB = el("rect", {
      x: 228, y: 202, width: 732, height: 332, rx: 12,
      fill: "var(--surface)",
      stroke: "var(--accent-primary)",
      "stroke-width": 1.3,
      "stroke-dasharray": "5 4"
    }, gLaneB);
    const frameBKicker = txt("CONFIG B · RECUPERACIÓN DENSA", {
      x: 246, y: 224, class: "svg-mono", "font-size": 11.5, "font-weight": 700, fill: "var(--accent-primary)"
    }, gLaneB);

    // 3A. Detailed dense retrieval blocks (visible in S1, smoothly compress left & fade in S2)
    const gRetFull = el("g", { opacity: 0 }, gLaneB);

    // Column 1: k chunks (x=246, y=254, w=118, h=258)
    el("rect", { x: 246, y: 254, width: 118, height: 258, rx: 10, fill: "var(--surface-tint)", stroke: "var(--line-strong)", "stroke-width": 1.3 }, gRetFull);
    txt("k chunks", { x: 305, y: 277, "text-anchor": "middle", "font-size": 13.5, "font-weight": 700 }, gRetFull);
    txt("≤110 pal./chunk", { x: 305, y: 295, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 10 }, gRetFull);
    ["Chunk 1", "Chunk 2", "Chunk 3", "…", "Chunk k"].forEach((lbl, idx) => {
      const cy = 310 + idx * 38;
      el("rect", { x: 258, y: cy, width: 94, height: 28, rx: 5, fill: "var(--surface)", stroke: idx < 3 ? "var(--accent-primary)" : "var(--line-strong)", "stroke-width": idx < 3 ? 1.2 : 1 }, gRetFull);
      txt(lbl, { x: 305, y: cy + 18, "text-anchor": "middle", class: "svg-mono", "font-size": 10.5, fill: idx < 3 ? "var(--ink)" : "var(--ink-3)" }, gRetFull);
    });

    // Arrow k chunks -> SciBERT
    el("path", { d: "M 364 383 H 388", fill: "none", stroke: "var(--accent-primary)", "stroke-width": 1.8, "marker-end": "url(#a8-m-b)", class: "a8-flow-dash" }, gRetFull);

    // Column 2: SciBERT Embeddings (x=392, y=254, w=146, h=258)
    el("rect", { x: 392, y: 254, width: 146, height: 258, rx: 10, fill: "var(--surface)", stroke: "var(--accent-violet)", "stroke-width": 1.6 }, gRetFull);
    txt("SciBERT", { x: 465, y: 277, "text-anchor": "middle", "font-size": 14, "font-weight": 700, fill: "var(--accent-violet)" }, gRetFull);
    txt("Embeddings 768-d", { x: 465, y: 295, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 10 }, gRetFull);

    // Query vector mini strip
    el("rect", { x: 404, y: 306, width: 122, height: 34, rx: 5, fill: "var(--surface-tint)", stroke: "var(--line)", "stroke-width": 1 }, gRetFull);
    txt("v(consulta)", { x: 412, y: 320, class: "svg-mono svg-muted", "font-size": 9.5 }, gRetFull);
    for (let j = 0; j < 11; j++) {
      el("rect", { x: 412 + j * 9.5, y: 325, width: 6, height: 6 + ((j * 5 + 3) % 7), rx: 1.5, fill: "var(--ink-2)", opacity: 0.45 + (j % 3) * 0.2 }, gRetFull);
    }
    // Chunk vectors
    for (let r = 0; r < 4; r++) {
      const vy = 354 + r * 32;
      for (let j = 0; j < 11; j++) {
        el("rect", { x: 412 + j * 9.5, y: vy, width: 6, height: 8 + ((j * 7 + r * 11) % 13), rx: 1.5, fill: "var(--accent-violet)", opacity: 0.38 + ((j + r) % 3) * 0.22 }, gRetFull);
      }
    }
    txt("Normalización L2", { x: 465, y: 498, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 10 }, gRetFull);

    // Arrow SciBERT -> Coseno
    el("path", { d: "M 538 383 H 562", fill: "none", stroke: "var(--accent-primary)", "stroke-width": 1.8, "marker-end": "url(#a8-m-b)", class: "a8-flow-dash" }, gRetFull);

    // Column 3: Coseno ranking (x=566, y=254, w=120, h=258)
    el("rect", { x: 566, y: 254, width: 120, height: 258, rx: 10, fill: "var(--surface-tint)", stroke: "var(--line-strong)", "stroke-width": 1.3 }, gRetFull);
    txt("Coseno", { x: 626, y: 277, "text-anchor": "middle", "font-size": 14, "font-weight": 700 }, gRetFull);
    txt("Consulta × chunks", { x: 626, y: 295, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 9.8 }, gRetFull);
    const cosBars = [
      { w: 92, hi: true, y: 318 },
      { w: 90, hi: true, y: 348 },
      { w: 89, hi: true, y: 378 },
      { w: 50, hi: false, y: 412 },
      { w: 34, hi: false, y: 440 }
    ];
    cosBars.forEach((cb) => {
      el("rect", { x: 580, y: cb.y, width: cb.w, height: 15, rx: 4, fill: cb.hi ? "var(--accent-primary)" : "var(--ink-3)", opacity: cb.hi ? 0.88 : 0.35 }, gRetFull);
    });
    txt("Top-3 ↓", { x: 626, y: 498, "text-anchor": "middle", class: "svg-mono", "font-size": 10.5, "font-weight": 700, fill: "var(--accent-primary)" }, gRetFull);

    // 3 branching arrows Coseno -> R1, R2, R3
    [
      "M 686 325 H 699 V 292 H 712",
      "M 686 355 H 701 V 383 H 712",
      "M 686 385 H 699 V 474 H 712"
    ].forEach((d) => {
      el("path", { d, fill: "none", stroke: "var(--accent-primary)", "stroke-width": 1.7, "marker-end": "url(#a8-m-b)", class: "a8-flow-dash" }, gRetFull);
    });

    // 3B. Compact accordion pill for S2 (replaces k chunks + SciBERT + Coseno cleanly)
    const gRetCompact = el("g", { opacity: 0 }, gLaneB);
    el("rect", { x: 174, y: 296, width: 112, height: 212, rx: 10, fill: "var(--surface-tint)", stroke: "var(--accent-violet)", "stroke-width": 1.4 }, gRetCompact);
    txt("k chunks", { x: 230, y: 326, "text-anchor": "middle", class: "svg-mono", "font-size": 11, "font-weight": 600 }, gRetCompact);
    txt("↓", { x: 230, y: 348, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 13 }, gRetCompact);
    el("rect", { x: 184, y: 358, width: 92, height: 50, rx: 7, fill: "var(--surface)", stroke: "var(--accent-violet)", "stroke-width": 1.2 }, gRetCompact);
    txt("SciBERT", { x: 230, y: 380, "text-anchor": "middle", "font-size": 12.5, "font-weight": 700, fill: "var(--accent-violet)" }, gRetCompact);
    txt("768-dim", { x: 230, y: 398, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 9.8 }, gRetCompact);
    txt("↓", { x: 230, y: 428, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 13 }, gRetCompact);
    txt("Coseno", { x: 230, y: 452, "text-anchor": "middle", "font-size": 12, "font-weight": 700 }, gRetCompact);
    txt("Ordenar Top-3", { x: 230, y: 472, "text-anchor": "middle", class: "svg-mono", "font-size": 9.8, fill: "var(--accent-primary)" }, gRetCompact);
    el("path", { d: "M 286 402 H 306", fill: "none", stroke: "var(--accent-primary)", "stroke-width": 1.8, "marker-end": "url(#a8-m-b)", class: "a8-flow-dash" }, gRetCompact);

    // 3C. Persistent Top-3 blue boxes (R1, R2, R3) — glide smoothly from S1 right -> S2 center!
    const gTop3 = el("g", { opacity: 0 }, gLaneB);
    const shortHeads = ["2.2.1 Private mean…", "2.1 Preliminaries", "1 Introduction"];
    const shortQuotes = [
      "“our algorithm draws heavily…”",
      "“These properties of the SDP…”",
      "“Our goal is to privately…”"
    ];
    const top3Cards = ex.chunks.map((ch, idx) => {
      const gCard = el("g", {}, gTop3);
      const rect = el("rect", {
        x: 714, y: 254 + idx * 91, width: 226, height: 76, rx: 9,
        fill: "var(--accent-primary-soft)", stroke: "var(--accent-primary)", "stroke-width": 1.6
      }, gCard);
      const tScore = txt(`R${ch.rank} · cos ${ch.score}`, {
        x: 726, y: 275 + idx * 91, class: "svg-mono", "font-size": 12, "font-weight": 700, fill: "var(--accent-primary)"
      }, gCard);
      const tHeadFull = txt(shortHeads[idx], {
        x: 726, y: 296 + idx * 91, "font-size": 12, "font-weight": 600
      }, gCard);
      const tQuoteFull = txt(shortQuotes[idx], {
        x: 726, y: 315 + idx * 91, "font-size": 10.8, fill: "var(--ink-2)"
      }, gCard);
      const tHeadMini = txt(shortHeads[idx], {
        x: 322, y: 340 + idx * 74, "font-size": 11, fill: "var(--ink-2)", opacity: 0
      }, gCard);

      bindTip(gCard, `Top-3 · R${ch.rank} (cos ${ch.score})`, [
        `Sección: ${ch.heading}`,
        `Pasaje: “${ch.text.slice(0, 110)}…”`
      ]);
      return { rect, tScore, tHeadFull, tQuoteFull, tHeadMini };
    });

    // ------------------------------------------------------------------
    // LAYER 4: Step 2 Inference — 2 Independent Classifiers (inside gray) + 2×9-bar Δ⁸ charts (outside right)
    // ------------------------------------------------------------------
    const gInfer = el("g", { opacity: 0 }, svg);

    // Connectors from Config A / Config B to their independent Classifiers
    el("path", { d: "M 486 149 H 520", fill: "none", stroke: "var(--accent-secondary)", "stroke-width": 2, "marker-end": "url(#a8-m-a)", class: "a8-flow-dash" }, gInfer);
    el("path", { d: "M 486 389 H 520", fill: "none", stroke: "var(--accent-primary)", "stroke-width": 2, "marker-end": "url(#a8-m-b)", class: "a8-flow-dash" }, gInfer);

    // Classifier A (independent box inside pale-gray EvidenCite container)
    const gClfA = el("g", {}, gInfer);
    el("rect", { x: 524, y: 88, width: 168, height: 122, rx: 12, fill: "var(--surface)", stroke: "var(--accent-violet)", "stroke-width": 1.7 }, gClfA);
    txt("CLASIFICADOR A", { x: 608, y: 116, "text-anchor": "middle", class: "svg-mono", "font-size": 10.8, "font-weight": 700, fill: "var(--accent-violet)" }, gClfA);
    txt("Modelo elegido", { x: 608, y: 146, "text-anchor": "middle", "font-size": 15, "font-weight": 700 }, gClfA);
    txt("Llamada independiente", { x: 608, y: 170, "text-anchor": "middle", "font-size": 11.5, fill: "var(--ink-2)" }, gClfA);
    txt("Entrada: Config A", { x: 608, y: 192, "text-anchor": "middle", class: "svg-mono", "font-size": 10.5, fill: "var(--accent-secondary)" }, gClfA);
    bindTip(gClfA, "Clasificador independiente (Config A)", [
      "Recibe contexto citante + título y resumen del citado.",
      "Puede ejecutarse con TF-IDF, SciBERT o LLM API."
    ]);

    // Classifier B (independent box inside pale-gray EvidenCite container)
    const gClfB = el("g", {}, gInfer);
    el("rect", { x: 524, y: 328, width: 168, height: 122, rx: 12, fill: "var(--surface)", stroke: "var(--accent-violet)", "stroke-width": 1.7 }, gClfB);
    txt("CLASIFICADOR B", { x: 608, y: 356, "text-anchor": "middle", class: "svg-mono", "font-size": 10.8, "font-weight": 700, fill: "var(--accent-violet)" }, gClfB);
    txt("Modelo elegido", { x: 608, y: 386, "text-anchor": "middle", "font-size": 15, "font-weight": 700 }, gClfB);
    txt("Llamada independiente", { x: 608, y: 410, "text-anchor": "middle", "font-size": 11.5, fill: "var(--ink-2)" }, gClfB);
    txt("Entrada: Config B", { x: 608, y: 432, "text-anchor": "middle", class: "svg-mono", "font-size": 10.5, fill: "var(--accent-primary)" }, gClfB);
    bindTip(gClfB, "Clasificador independiente (Config B)", [
      "Recibe contexto citante + pasajes Top-3 (R1–R3) con encabezados.",
      "Inferencia separada sobre la misma familia de modelo."
    ]);

    // Arrows crossing the pale-gray border (x=714) out to the two 9-class charts
    el("path", { d: "M 692 149 H 740", fill: "none", stroke: "var(--accent-secondary)", "stroke-width": 2.2, "marker-end": "url(#a8-m-a)", class: "a8-flow-dash" }, gInfer);
    el("path", { d: "M 692 389 H 740", fill: "none", stroke: "var(--accent-primary)", "stroke-width": 2.2, "marker-end": "url(#a8-m-b)", class: "a8-flow-dash" }, gInfer);

    // Two 9-class probability bar charts (OUTSIDE right of EvidenCite box: x=744..988)
    const clsCodes = ["Bg", "Ga", "Ba", "Cm", "Ap", "Im", "Ev", "Or", "FR"];
    const clsFull = ["Background", "Gap", "Basis", "Comparison", "Application", "Improvement", "Evidence", "Originator", "Further Reading"];
    // Schematic illustrative probability profiles (varied heights, clearly labeled illustrative)
    const distA = [0.42, 0.14, 0.62, 0.25, 0.20, 0.12, 0.08, 0.07, 0.06];
    const distB = [0.16, 0.07, 0.94, 0.11, 0.24, 0.14, 0.06, 0.05, 0.05];

    const animBars = [];
    function buildDistChart(yTop, title, accent, profile, tag) {
      const gOut = el("g", {}, gInfer);
      el("rect", {
        x: 744, y: yTop, width: 244, height: 196, rx: 12,
        fill: "var(--surface)", stroke: accent, "stroke-width": 1.6
      }, gOut);
      txt(title, { x: 758, y: yTop + 24, class: "svg-mono", "font-size": 11, "font-weight": 700, fill: accent }, gOut);
      txt("Probabilidad en 9 clases (Δ⁸)", { x: 758, y: yTop + 42, "font-size": 11.5, fill: "var(--ink-2)" }, gOut);

      const baseY = yTop + 158;
      const maxH = 96;
      el("line", { x1: 758, y1: baseY, x2: 974, y2: baseY, stroke: "var(--line-strong)", "stroke-width": 1.2 }, gOut);
      el("line", { x1: 758, y1: baseY - maxH * 0.5, x2: 974, y2: baseY - maxH * 0.5, class: "svg-grid" }, gOut);

      const startX = 762;
      const pitch = 23.5;
      const barW = 16;
      const maxVal = Math.max(...profile);

      profile.forEach((v, idx) => {
        const bx = startX + idx * pitch;
        const targetH = Math.max(4, v * maxH);
        const isPeak = v === maxVal;
        const bar = el("rect", {
          x: bx, y: baseY, width: barW, height: 0, rx: 3,
          fill: accent, opacity: isPeak ? 0.96 : 0.44
        }, gOut);
        txt(clsCodes[idx], {
          x: bx + barW / 2, y: baseY + 16, "text-anchor": "middle",
          class: "svg-mono", "font-size": 9.5, "font-weight": isPeak ? 700 : 500,
          fill: isPeak ? accent : "var(--ink-3)"
        }, gOut);

        bindTip(bar, `${tag} · ${clsFull[idx]}`, [
          "Esquema ilustrativo de distribución sobre las 9 funciones (Δ⁸).",
          "La predicción numérica real se inspecciona en la demo en vivo."
        ]);
        animBars.push({ bar, baseY, targetH, idx });
      });

      txt("Esquema ilustrativo · ∑ p = 1", { x: 866, y: yTop + 188, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 9.5 }, gOut);
    }

    buildDistChart(52,  "SALIDA A · 9 FUNCIONES", "var(--accent-secondary)", distA, "Salida A");
    buildDistChart(292, "SALIDA B · 9 FUNCIONES", "var(--accent-primary)",   distB, "Salida B");

    // ------------------------------------------------------------------
    // LAYER 5: Input cards (persistent across S0 -> S1 -> S2) + Step-specific routing arrows
    // ------------------------------------------------------------------
    // S0 arrows (Inputs -> EvidenCite intake)
    const gArrS0 = el("g", {}, svg);
    el("path", { d: "M 366 177 H 428 V 256 H 488", fill: "none", stroke: "var(--ink-3)", "stroke-width": 2, "marker-end": "url(#a8-m-n)", class: "a8-flow-dash" }, gArrS0);
    el("path", { d: "M 366 397 H 428 V 318 H 488", fill: "none", stroke: "var(--ink-3)", "stroke-width": 2, "marker-end": "url(#a8-m-n)", class: "a8-flow-dash" }, gArrS0);

    // S1 arrows (Inputs -> Config A top + Config B bottom: citing query -> SciBERT, cited body -> k chunks)
    const gArrS1 = el("g", { opacity: 0 }, svg);
    // Citing -> Config A
    el("path", { d: "M 158 112 H 224", fill: "none", stroke: "var(--ink-3)", "stroke-width": 1.9, "marker-end": "url(#a8-m-n)", class: "a8-flow-dash" }, gArrS1);
    // Citing -> SciBERT query (clean top band inside frameB, zero overlap!)
    el("path", { d: "M 158 156 H 176 V 236 H 465 V 250", fill: "none", stroke: "var(--ink-3)", "stroke-width": 1.6, "marker-end": "url(#a8-m-n)", class: "a8-flow-dash" }, gArrS1);
    txt("consulta", { x: 474, y: 232, class: "svg-mono svg-muted", "font-size": 10 }, gArrS1);
    // Cited -> Config A (title + abstract)
    el("path", { d: "M 158 354 H 190 V 148 H 224", fill: "none", stroke: "var(--accent-secondary)", "stroke-width": 1.8, "marker-end": "url(#a8-m-a)", class: "a8-flow-dash" }, gArrS1);
    // Cited -> Config B (k chunks)
    el("path", { d: "M 158 396 H 242", fill: "none", stroke: "var(--accent-primary)", "stroke-width": 1.9, "marker-end": "url(#a8-m-b)", class: "a8-flow-dash" }, gArrS1);

    // S2 arrows (Mini inputs -> Config A + Config B)
    const gArrS2 = el("g", { opacity: 0 }, svg);
    el("path", { d: "M 114 134 H 156", fill: "none", stroke: "var(--ink-3)", "stroke-width": 1.7, "marker-end": "url(#a8-m-n)", class: "a8-flow-dash" }, gArrS2);
    el("path", { d: "M 114 166 H 124 V 352 H 156", fill: "none", stroke: "var(--ink-3)", "stroke-width": 1.4, "marker-end": "url(#a8-m-n)", class: "a8-flow-dash" }, gArrS2);
    el("path", { d: "M 114 378 H 132 V 176 H 156", fill: "none", stroke: "var(--accent-secondary)", "stroke-width": 1.5, "marker-end": "url(#a8-m-a)", class: "a8-flow-dash" }, gArrS2);
    el("path", { d: "M 114 408 H 156", fill: "none", stroke: "var(--accent-primary)", "stroke-width": 1.7, "marker-end": "url(#a8-m-b)", class: "a8-flow-dash" }, gArrS2);

    // Card 1: Contexto Citante
    const gCiting = el("g", {}, svg);
    const rectCiting = el("rect", {
      x: 46, y: 92, width: 320, height: 170, rx: 12,
      fill: "var(--surface-tint)", stroke: "var(--ink-3)", "stroke-width": 1.6
    }, gCiting);

    const citingFull = el("g", {}, gCiting);
    txt("CONTEXTO CITANTE · FIJO", { x: 64, y: 120, class: "svg-mono svg-muted", "font-size": 11.5, "font-weight": 700 }, citingFull);
    txt("“The procedure builds upon the", { x: 64, y: 150, "font-size": 14.5 }, citingFull);
    txt("recent work by Hopkins, Kamath", { x: 64, y: 173, "font-size": 14.5 }, citingFull);
    txt("and Majid <cite>…</cite>”", { x: 64, y: 196, "font-size": 14.5 }, citingFull);
    txt(`3 oraciones · ${ex.citingId}`, { x: 64, y: 236, class: "svg-mono", "font-size": 12.5, "font-weight": 700, fill: "var(--accent-gold)" }, citingFull);

    const citingMed = el("g", { opacity: 0 }, gCiting);
    txt("CITANTE · FIJO", { x: 26, y: 98, class: "svg-mono svg-muted", "font-size": 10.5, "font-weight": 700 }, citingMed);
    txt("3 oraciones", { x: 26, y: 126, "font-size": 13.5, "font-weight": 700 }, citingMed);
    txt(ex.citingId, { x: 26, y: 148, class: "svg-mono", "font-size": 11.5, fill: "var(--ink-2)" }, citingMed);
    txt("Cita marcada", { x: 26, y: 168, "font-size": 11, fill: "var(--ink-3)" }, citingMed);

    const citingMini = el("g", { opacity: 0 }, gCiting);
    txt("CITANTE", { x: 63, y: 130, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 10.5, "font-weight": 700 }, citingMini);
    txt("3 oraciones", { x: 63, y: 156, "text-anchor": "middle", "font-size": 12, "font-weight": 700 }, citingMini);
    txt("fijo", { x: 63, y: 176, "text-anchor": "middle", class: "svg-mono", "font-size": 10.5, fill: "var(--accent-gold)" }, citingMini);

    // Card 2: Artículo Citado
    const gCited = el("g", {}, svg);
    const rectCited = el("rect", {
      x: 46, y: 312, width: 320, height: 170, rx: 12,
      fill: "var(--surface)", stroke: "var(--line-strong)", "stroke-width": 1.6
    }, gCited);

    const citedFull = el("g", {}, gCited);
    txt("ARTÍCULO CITADO", { x: 64, y: 340, class: "svg-mono svg-muted", "font-size": 11.5, "font-weight": 700 }, citedFull);
    txt(ex.citedId, { x: 64, y: 366, class: "svg-mono", "font-size": 14, "font-weight": 700 }, citedFull);
    txt("Efficient mean estimation with pure DP", { x: 64, y: 392, "font-size": 13.5, fill: "var(--ink-2)" }, citedFull);
    [0, 1, 2, 3].forEach((j) => {
      el("rect", { x: 64, y: 410 + j * 13, width: 256 - (j % 3) * 38, height: 5, rx: 2.5, fill: "var(--ink-3)", opacity: 0.32 }, citedFull);
    });

    const citedMed = el("g", { opacity: 0 }, gCited);
    txt("CITADO", { x: 26, y: 350, class: "svg-mono svg-muted", "font-size": 10.5, "font-weight": 700 }, citedMed);
    txt(ex.citedId, { x: 26, y: 374, class: "svg-mono", "font-size": 12.5, "font-weight": 700 }, citedMed);
    txt("Texto completo", { x: 26, y: 396, "font-size": 11.5, fill: "var(--ink-2)" }, citedMed);
    [0, 1, 2].forEach((j) => {
      el("rect", { x: 26, y: 410 + j * 12, width: 110 - (j % 2) * 26, height: 4.5, rx: 2, fill: "var(--ink-3)", opacity: 0.35 }, citedMed);
    });

    const citedMini = el("g", { opacity: 0 }, gCited);
    txt("CITADO", { x: 63, y: 378, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 10.5, "font-weight": 700 }, citedMini);
    txt(ex.citedId, { x: 63, y: 404, "text-anchor": "middle", class: "svg-mono", "font-size": 11, "font-weight": 700 }, citedMini);
    txt("documento", { x: 63, y: 424, "text-anchor": "middle", "font-size": 10.5, fill: "var(--ink-3)" }, citedMini);

    // Bottom subtle note inside SVG
    const svgFoot = txt("Normalización de etiquetas (<cite>, [OTHERCIT]) y resolución jerárquica del artículo citado.", {
      x: 500, y: 576, "text-anchor": "middle", class: "svg-mono svg-muted", "font-size": 10.8
    }, svg);

    // ------------------------------------------------------------------
    // Target state keyframes (S0, S1, S2)
    // ------------------------------------------------------------------
    const STATES = [
      {
        citing: { x: 46, y: 92, w: 320, h: 170 },
        cited:  { x: 46, y: 312, w: 320, h: 170 },
        evid:   { x: 492, y: 162, w: 438, h: 250 },
        boxA:   { x: 520, y: 186, w: 380, h: 96 },
        frameB: { x: 520, y: 292, w: 380, h: 106 },
        top3:   { x: 714, y: 254, w: 226, h: 76, gap: 91 },
        s0: 1, s1: 0, s2: 0, laneOp: 0, retFullOp: 0, retFullTx: 80, retCompactOp: 0, inferOp: 0, inferTx: 36, barP: 0
      },
      {
        citing: { x: 14, y: 74, w: 144, h: 110 },
        cited:  { x: 14, y: 326, w: 144, h: 132 },
        evid:   { x: 204, y: 24, w: 780, h: 530 },
        boxA:   { x: 228, y: 66, w: 732, h: 114 },
        frameB: { x: 228, y: 202, w: 732, h: 332 },
        top3:   { x: 714, y: 254, w: 226, h: 76, gap: 91 },
        s0: 0, s1: 1, s2: 0, laneOp: 1, retFullOp: 1, retFullTx: 0, retCompactOp: 0, inferOp: 0, inferTx: 36, barP: 0
      },
      {
        citing: { x: 12, y: 102, w: 102, h: 92 },
        cited:  { x: 12, y: 350, w: 102, h: 92 },
        evid:   { x: 138, y: 28, w: 576, h: 518 },
        boxA:   { x: 160, y: 74, w: 326, h: 150 },
        frameB: { x: 160, y: 254, w: 326, h: 270 },
        top3:   { x: 310, y: 296, w: 162, h: 62, gap: 75 },
        s0: 0, s1: 0, s2: 1, laneOp: 1, retFullOp: 0, retFullTx: -70, retCompactOp: 1, inferOp: 1, inferTx: 0, barP: 1
      }
    ];

    function cloneState(s) {
      return {
        citing: { ...s.citing }, cited: { ...s.cited }, evid: { ...s.evid },
        boxA: { ...s.boxA }, frameB: { ...s.frameB }, top3: { ...s.top3 },
        s0: s.s0, s1: s.s1, s2: s.s2, laneOp: s.laneOp,
        retFullOp: s.retFullOp, retFullTx: s.retFullTx,
        retCompactOp: s.retCompactOp, inferOp: s.inferOp, inferTx: s.inferTx, barP: s.barP
      };
    }

    let cur = cloneState(STATES[0]);
    let activeStep = -1;
    let cancelTween = null;

    function applyState(st) {
      setBox(rectCiting, st.citing);
      setBox(rectCited, st.cited);
      setBox(evidRect, st.evid);
      setBox(rectA, st.boxA);
      setBox(frameB, st.frameB);

      evidHeader.setAttribute("x", String(st.evid.x + 20));
      evidHeader.setAttribute("y", String(st.evid.y + 24));
      evidHeader.setAttribute("opacity", String(st.laneOp));

      frameBKicker.setAttribute("x", String(st.frameB.x + 18));
      frameBKicker.setAttribute("y", String(st.frameB.y + 22));

      gS0Intake.setAttribute("opacity", String(st.s0));
      gArrS0.setAttribute("opacity", String(st.s0));
      citingFull.setAttribute("opacity", String(st.s0));
      citedFull.setAttribute("opacity", String(st.s0));

      gArrS1.setAttribute("opacity", String(st.s1));
      citingMed.setAttribute("opacity", String(st.s1));
      citedMed.setAttribute("opacity", String(st.s1));
      gAFull.setAttribute("opacity", String(st.s1));

      gArrS2.setAttribute("opacity", String(st.s2));
      citingMini.setAttribute("opacity", String(st.s2));
      citedMini.setAttribute("opacity", String(st.s2));
      gACompact.setAttribute("opacity", String(st.s2));

      gLaneA.setAttribute("opacity", String(st.laneOp));
      gLaneB.setAttribute("opacity", String(st.laneOp));

      gRetFull.setAttribute("opacity", String(st.retFullOp));
      gRetFull.setAttribute("transform", `translate(${st.retFullTx}, 0)`);
      gRetCompact.setAttribute("opacity", String(st.retCompactOp));

      gTop3.setAttribute("opacity", String(st.laneOp));
      top3Cards.forEach((c, idx) => {
        const cy = st.top3.y + idx * st.top3.gap;
        c.rect.setAttribute("x", String(st.top3.x));
        c.rect.setAttribute("y", String(cy));
        c.rect.setAttribute("width", String(st.top3.w));
        c.rect.setAttribute("height", String(st.top3.h));

        c.tScore.setAttribute("x", String(st.top3.x + 12));
        c.tScore.setAttribute("y", String(cy + 23));

        c.tHeadFull.setAttribute("x", String(st.top3.x + 12));
        c.tHeadFull.setAttribute("y", String(cy + 43));
        c.tHeadFull.setAttribute("opacity", String(st.s1));

        c.tQuoteFull.setAttribute("x", String(st.top3.x + 12));
        c.tQuoteFull.setAttribute("y", String(cy + 61));
        c.tQuoteFull.setAttribute("opacity", String(st.s1));

        c.tHeadMini.setAttribute("x", String(st.top3.x + 12));
        c.tHeadMini.setAttribute("y", String(cy + 45));
        c.tHeadMini.setAttribute("opacity", String(st.s2));
      });

      gInfer.setAttribute("opacity", String(st.inferOp));
      gInfer.setAttribute("transform", `translate(${st.inferTx}, 0)`);

      animBars.forEach(({ bar, baseY, targetH, idx }) => {
        // Gentle left-to-right stagger across the 9 bars as barP moves 0 -> 1
        const localP = Math.max(0, Math.min(1, (st.barP - idx * 0.03) / 0.73));
        const h = targetH * localP;
        bar.setAttribute("y", String(baseY - h));
        bar.setAttribute("height", String(h));
      });
    }

    function lerpBox(a, b, t) {
      return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t), w: lerp(a.w, b.w, t), h: lerp(a.h, b.h, t) };
    }

    applyState(cur);

    return {
      update(step) {
        const targetIdx = Math.max(0, Math.min(2, Number(step) || 0));
        if (activeStep === targetIdx) return;
        const firstPaint = activeStep === -1;
        activeStep = targetIdx;

        evidHeader.textContent = targetIdx === 2
          ? "EVIDENCITE · REPRESENTACIÓN E INFERENCIA"
          : "EVIDENCITE · PIPELINE DE REPRESENTACIÓN";
        svgFoot.textContent = targetIdx === 0
          ? "Normalización de etiquetas (<cite>, [OTHERCIT]) y resolución jerárquica del artículo citado."
          : targetIdx === 1
            ? "Recuperación densa sobre inst_04783 · cosenos reales: 0,894 / 0,889 / 0,888 · resto descartado."
            : "Esquema ilustrativo de distribuciones en Δ⁸ (9 funciones retóricas) · predicción real en la demo.";

        if (cancelTween) { cancelTween(); cancelTween = null; }
        const from = cloneState(cur);
        const to = STATES[targetIdx];

        if (firstPaint) {
          cur = cloneState(to);
          applyState(cur);
          return;
        }

        cancelTween = tween(1180, (t) => {
          // Slightly delayed phase for fading-in new layers so geometry settles cleanly first
          const tIn = Math.max(0, Math.min(1, (t - 0.22) / 0.78));
          const tOut = Math.max(0, Math.min(1, t / 0.55));
          const tBars = Math.max(0, Math.min(1, (t - 0.32) / 0.68));

          cur = {
            citing: lerpBox(from.citing, to.citing, t),
            cited:  lerpBox(from.cited, to.cited, t),
            evid:   lerpBox(from.evid, to.evid, t),
            boxA:   lerpBox(from.boxA, to.boxA, t),
            frameB: lerpBox(from.frameB, to.frameB, t),
            top3: {
              x: lerp(from.top3.x, to.top3.x, t),
              y: lerp(from.top3.y, to.top3.y, t),
              w: lerp(from.top3.w, to.top3.w, t),
              h: lerp(from.top3.h, to.top3.h, t),
              gap: lerp(from.top3.gap, to.top3.gap, t)
            },
            s0: to.s0 > from.s0 ? lerp(from.s0, to.s0, tIn) : lerp(from.s0, to.s0, tOut),
            s1: to.s1 > from.s1 ? lerp(from.s1, to.s1, tIn) : lerp(from.s1, to.s1, tOut),
            s2: to.s2 > from.s2 ? lerp(from.s2, to.s2, tIn) : lerp(from.s2, to.s2, tOut),
            laneOp: to.laneOp > from.laneOp ? lerp(from.laneOp, to.laneOp, tIn) : lerp(from.laneOp, to.laneOp, tOut),
            retFullOp: to.retFullOp > from.retFullOp ? lerp(from.retFullOp, to.retFullOp, tIn) : lerp(from.retFullOp, to.retFullOp, tOut),
            retFullTx: lerp(from.retFullTx, to.retFullTx, t),
            retCompactOp: to.retCompactOp > from.retCompactOp ? lerp(from.retCompactOp, to.retCompactOp, tIn) : lerp(from.retCompactOp, to.retCompactOp, tOut),
            inferOp: to.inferOp > from.inferOp ? lerp(from.inferOp, to.inferOp, tIn) : lerp(from.inferOp, to.inferOp, tOut),
            inferTx: lerp(from.inferTx, to.inferTx, t),
            barP: to.barP > from.barP ? lerp(from.barP, to.barP, tBars) : lerp(from.barP, to.barP, tOut)
          };
          applyState(cur);
        }, 90);
      }
    };
  }

  return {
    Tooltip,
    createMedallionDAG,
    createA2Panel,
    renderA2Strata,
    renderA3Waterfall,
    createA3Morph,
    createA4Funnel,
    renderA4Collapse,
    createFig11ThreeStep,
    renderA5ParityBreak,
    renderA5ByClass,
    renderA5MacroSections,
    renderA5NineChips,
    renderA6GlobalBoard,
    renderA6Calibration,
    createCostQualityChart,
    renderA8Architecture,
    createA8ScrollFlow
  };
})();
