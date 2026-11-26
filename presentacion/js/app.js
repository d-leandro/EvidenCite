/* ==========================================================================
   EvidenCite — Motor Híbrido de Scrollytelling, Teclado/Clicker, Tema y Reloj
   Atajos: ←/→/↑/↓/PageUp/PageDown/Espacio · T (tema) · H (reloj) · R (reset) · Q (Q&A)
   ========================================================================== */

(function () {
  const D = window.EVIDENCITE_DATA;
  const Scenes = window.EvidenScenes;
  const QA = window.EvidenQA;

  let beats = [];
  let currentBeat = 0;
  let isProgrammaticScroll = false;
  let scrollUnlockTimer = null;

  // Micro-timer state (strictly optional via H, zero notes)
  const timerState = {
    visible: false,
    started: false,
    elapsedSec: 0,
    intervalId: null
  };

  function pad2(n) {
    return String(n).padStart(2, "0");
  }

  function formatClock(sec) {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${pad2(m)}:${pad2(s)} / 15:00`;
  }

  function updateTimerHUD() {
    const hud = document.getElementById("timer-hud");
    const clock = document.getElementById("timer-clock");
    const blockLbl = document.getElementById("timer-block");
    if (!hud || !clock || !blockLbl) return;

    clock.textContent = formatClock(timerState.elapsedSec);

    const beatEl = beats[currentBeat];
    const blkIdx = Number(beatEl ? beatEl.dataset.block : 0);
    const blk = D.meta.blocks[blkIdx] || D.meta.blocks[0];
    const targetMin = blk.targetSec > 0 ? `${Math.floor(blk.targetSec / 60)}:${pad2(blk.targetSec % 60)}` : "Inicio";
    blockLbl.textContent = `${blk.speaker} · ${targetMin}`;

    hud.classList.remove("is-warn", "is-over");
    if (timerState.elapsedSec >= 900) {
      hud.classList.add("is-over");
    } else if (timerState.elapsedSec >= 780) {
      hud.classList.add("is-warn");
    }
  }

  function toggleTimer() {
    const hud = document.getElementById("timer-hud");
    if (!hud) return;
    timerState.visible = !timerState.visible;
    hud.classList.toggle("is-visible", timerState.visible);
    if (timerState.visible && !timerState.started) {
      timerState.started = true;
      timerState.intervalId = setInterval(() => {
        timerState.elapsedSec += 1;
        updateTimerHUD();
      }, 1000);
    }
    updateTimerHUD();
  }

  function resetTimer() {
    if (!timerState.visible) return;
    timerState.elapsedSec = 0;
    updateTimerHUD();
  }

  function toggleTheme() {
    const root = document.documentElement;
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
  }

  function updateTopBar(beatIdx) {
    const beatEl = beats[beatIdx];
    if (!beatEl) return;
    const phase = beatEl.dataset.phase || "CTX";
    const speaker = beatEl.dataset.speaker || "Leandro";

    const counter = document.getElementById("beat-counter");
    if (counter) counter.textContent = `${pad2(beatIdx)} / ${pad2(beats.length - 1)}`;

    const fill = document.getElementById("progress-fill");
    if (fill) {
      const pct = beats.length > 1 ? (beatIdx / (beats.length - 1)) * 100 : 0;
      fill.style.width = `${pct.toFixed(1)}%`;
    }

    const badge = document.getElementById("speaker-badge");
    const name = document.getElementById("speaker-name");
    if (badge && name) {
      badge.setAttribute("data-speaker", speaker);
      name.textContent = speaker;
    }

    const phaseOrder = ["CTX", "A1", "A2", "A3", "A4", "A5", "A6/A7", "A8", "CONC"];
    const activeOrderIdx = phaseOrder.indexOf(phase);
    document.querySelectorAll(".phase-pill").forEach((pill) => {
      const p = pill.dataset.phase;
      const pIdx = phaseOrder.indexOf(p);
      pill.classList.toggle("is-active", p === phase);
      pill.classList.toggle("is-done", pIdx >= 0 && pIdx < activeOrderIdx);
    });
  }

  function activateBeat(beatIdx, shouldScroll) {
    const clamped = Math.max(0, Math.min(beats.length - 1, beatIdx));
    const previousBeat = currentBeat;
    currentBeat = clamped;

    beats.forEach((b, i) => {
      b.classList.toggle("is-active", i === clamped);
    });

    const el = beats[clamped];
    if (el.dataset.scene) {
      Scenes.activateScene(el.dataset.scene, Number(el.dataset.step || 0));
    }

    updateTopBar(clamped);
    updateTimerHUD();

    if (shouldScroll) {
      isProgrammaticScroll = true;
      clearTimeout(scrollUnlockTimer);
      const topbarH = 52;
      const rect = el.getBoundingClientRect();
      const isFull = el.classList.contains("full-beat");
      const targetY = isFull
        ? window.scrollY + rect.top - topbarH
        : window.scrollY + rect.top - (window.innerHeight - rect.height) / 2;

      window.scrollTo({ top: Math.max(0, targetY), behavior: "smooth" });
      scrollUnlockTimer = setTimeout(() => {
        isProgrammaticScroll = false;
      }, 520);
    }
  }

  function setupScrollSync() {
    const observer = new IntersectionObserver(
      (entries) => {
        if (isProgrammaticScroll) return;
        let best = null;
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            if (!best || entry.intersectionRatio > best.intersectionRatio) {
              best = entry;
            }
          }
        });
        if (best) {
          const idx = Number(best.target.dataset.beat);
          if (!Number.isNaN(idx) && idx !== currentBeat) {
            activateBeat(idx, false);
          }
        }
      },
      {
        rootMargin: "-26% 0px -26% 0px",
        threshold: [0.25, 0.5, 0.75]
      }
    );

    beats.forEach((b) => observer.observe(b));
  }

  function setupKeyboard() {
    document.addEventListener("keydown", (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const key = e.key;

      if (key === "Escape") {
        if (QA.isOpen()) {
          e.preventDefault();
          QA.close();
        }
        return;
      }

      if (key === "q" || key === "Q") {
        e.preventDefault();
        QA.toggle();
        return;
      }

      if (key === "t" || key === "T") {
        e.preventDefault();
        toggleTheme();
        return;
      }

      if (key === "h" || key === "H") {
        e.preventDefault();
        toggleTimer();
        return;
      }

      if (key === "r" || key === "R") {
        if (timerState.visible) {
          e.preventDefault();
          resetTimer();
        }
        return;
      }

      // While Q&A overlay is open, do not advance background beats
      if (QA.isOpen()) return;

      if (key === "ArrowRight" || key === "ArrowDown" || key === "PageDown" || key === " ") {
        e.preventDefault();
        activateBeat(currentBeat + 1, true);
      } else if (key === "ArrowLeft" || key === "ArrowUp" || key === "PageUp") {
        e.preventDefault();
        activateBeat(currentBeat - 1, true);
      } else if (key === "Home") {
        e.preventDefault();
        activateBeat(0, true);
      } else if (key === "End") {
        e.preventDefault();
        activateBeat(beats.length - 1, true);
      }
    });
  }

  function setupTopNavClicks() {
    const home = document.getElementById("brand-home");
    if (home) {
      home.addEventListener("click", () => activateBeat(0, true));
    }

    document.querySelectorAll(".phase-pill[data-goto-beat]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = Number(btn.dataset.gotoBeat);
        if (!Number.isNaN(target)) activateBeat(target, true);
      });
    });

    // Clicking a narrative beat step focuses it cleanly
    beats.forEach((b, idx) => {
      if (b.classList.contains("beat-step")) {
        b.style.cursor = "pointer";
        b.addEventListener("click", () => activateBeat(idx, true));
      }
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    beats = Array.from(document.querySelectorAll("[data-beat]")).sort(
      (a, b) => Number(a.dataset.beat) - Number(b.dataset.beat)
    );

    window.EvidenCharts.Tooltip.init();
    Scenes.initAllDefaults();
    QA.init();
    setupScrollSync();
    setupKeyboard();
    setupTopNavClicks();

    activateBeat(0, false);
  });
})();
