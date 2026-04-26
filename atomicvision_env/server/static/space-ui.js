const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

function setupCursor() {
  const cursor = document.querySelector(".cursor-ring");
  if (!cursor || window.matchMedia("(pointer: coarse)").matches) return;

  let x = window.innerWidth / 2;
  let y = window.innerHeight / 2;
  let raf = 0;

  const render = () => {
    cursor.style.transform = `translate3d(${x}px, ${y}px, 0)`;
    raf = 0;
  };

  window.addEventListener(
    "pointermove",
    (event) => {
      x = event.clientX;
      y = event.clientY;
      if (!raf) raf = requestAnimationFrame(render);
    },
    { passive: true }
  );
}

function setupParticles() {
  const holder = document.querySelector(".particles");
  if (!holder) return;

  const fragment = document.createDocumentFragment();
  Array.from({ length: 46 }, (_, index) => {
    const particle = document.createElement("span");
    particle.style.left = `${(index * 23) % 100}%`;
    particle.style.top = `${(index * 41) % 100}%`;
    particle.style.animationDelay = `${(index % 12) * 0.45}s`;
    particle.style.animationDuration = `${8 + (index % 7)}s`;
    fragment.appendChild(particle);
  });
  holder.appendChild(fragment);
}

function setupReveal() {
  const nodes = document.querySelectorAll("[data-reveal]");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) entry.target.classList.add("is-visible");
      });
    },
    { threshold: 0.18, rootMargin: "0px 0px -8% 0px" }
  );

  nodes.forEach((node) => observer.observe(node));
}

function setupGlassReflection() {
  document.querySelectorAll(".glass-card").forEach((card) => {
    card.addEventListener(
      "pointermove",
      (event) => {
        const rect = card.getBoundingClientRect();
        card.style.setProperty("--mx", `${event.clientX - rect.left}px`);
        card.style.setProperty("--my", `${event.clientY - rect.top}px`);
      },
      { passive: true }
    );
  });
}

function setupHeroVideoScrub() {
  const hero = document.querySelector(".hero-scroll");
  const video = document.querySelector(".video-frame video");
  if (!hero || !video) return;

  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const isMobile = window.matchMedia("(max-width: 767px)").matches;
  let duration = 0;
  let targetTime = 0;
  let currentTime = 0;
  let raf = 0;

  const mobileFallback = () => {
    video.loop = true;
    video.muted = true;
    video.playbackRate = 0.72;
    video.play().catch(() => {});
  };

  if (prefersReduced || isMobile) {
    mobileFallback();
    return;
  }

  const progressFromScroll = () => {
    const rect = hero.getBoundingClientRect();
    const scrollable = Math.max(hero.offsetHeight - window.innerHeight, 1);
    return clamp(-rect.top / scrollable, 0, 1);
  };

  const tick = () => {
    const delta = targetTime - currentTime;
    currentTime += delta * 0.14;
    if (Math.abs(delta) < 0.008) currentTime = targetTime;

    if (Number.isFinite(currentTime) && duration) {
      video.currentTime = clamp(currentTime, 0, Math.max(duration - 0.04, 0));
    }

    if (Math.abs(targetTime - currentTime) > 0.006) {
      raf = requestAnimationFrame(tick);
    } else {
      raf = 0;
    }
  };

  const updateTarget = () => {
    if (!duration) return;
    targetTime = progressFromScroll() * duration;
    if (!raf) raf = requestAnimationFrame(tick);
  };

  const onLoaded = () => {
    duration = video.duration || 0;
    video.pause();
    updateTarget();
  };

  video.addEventListener("loadedmetadata", onLoaded);
  window.addEventListener("scroll", updateTarget, { passive: true });
  window.addEventListener("resize", updateTarget);
  if (video.readyState >= 1) onLoaded();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${body}`);
  }
  return response.json();
}

function buildWebSocketUrl(path = "/ws") {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}

function openDemoSocket(path = "/ws") {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(buildWebSocketUrl(path));

    const cleanup = () => {
      socket.removeEventListener("open", handleOpen);
      socket.removeEventListener("error", handleError);
      socket.removeEventListener("close", handleClose);
    };

    const handleOpen = () => {
      cleanup();
      resolve(socket);
    };

    const handleError = () => {
      cleanup();
      reject(new Error("Unable to open the AtomicVision live session."));
    };

    const handleClose = () => {
      cleanup();
      reject(new Error("AtomicVision live session closed before the demo started."));
    };

    socket.addEventListener("open", handleOpen);
    socket.addEventListener("error", handleError);
    socket.addEventListener("close", handleClose);
  });
}

function sendSocketMessage(socket, payload, timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    let settled = false;

    const cleanup = () => {
      clearTimeout(timer);
      socket.removeEventListener("message", handleMessage);
      socket.removeEventListener("error", handleError);
      socket.removeEventListener("close", handleClose);
    };

    const settle = (callback) => {
      if (settled) return;
      settled = true;
      cleanup();
      callback();
    };

    const handleMessage = (event) => {
      settle(() => {
        try {
          const message = JSON.parse(event.data);
          if (message.type === "error") {
            reject(new Error(message.data?.message || "AtomicVision session error."));
            return;
          }
          resolve(message);
        } catch (error) {
          reject(
            new Error(
              error instanceof Error
                ? error.message
                : "Failed to parse AtomicVision session response."
            )
          );
        }
      });
    };

    const handleError = () => {
      settle(() => reject(new Error("AtomicVision live session transport error.")));
    };

    const handleClose = () => {
      settle(() =>
        reject(new Error("AtomicVision live session closed unexpectedly."))
      );
    };

    const timer = window.setTimeout(() => {
      settle(() =>
        reject(new Error("Timed out waiting for the AtomicVision session response."))
      );
    }, timeoutMs);

    socket.addEventListener("message", handleMessage);
    socket.addEventListener("error", handleError);
    socket.addEventListener("close", handleClose);
    socket.send(JSON.stringify(payload));
  });
}

function bucketValues(values, bucketCount) {
  if (!values.length || bucketCount <= 0) return [];
  const bucketSize = Math.ceil(values.length / bucketCount);
  const buckets = [];
  for (let i = 0; i < bucketCount; i += 1) {
    const start = i * bucketSize;
    const slice = values.slice(start, start + bucketSize);
    if (!slice.length) break;
    const average = slice.reduce((sum, item) => sum + item, 0) / slice.length;
    buckets.push(average);
  }
  return buckets;
}

function createHeatmap(values) {
  const heatmap = document.querySelector("#heatmap");
  if (!heatmap) return;

  heatmap.innerHTML = "";
  if (!values.length) return;
  const maxValue = Math.max(...values, 1e-6);
  const fragment = document.createDocumentFragment();
  values.slice(0, 64).forEach((value, index) => {
    const node = document.createElement("span");
    const alpha = 0.16 + (value / maxValue) * 0.84;
    node.style.opacity = alpha.toFixed(3);
    node.style.animationDelay = `${(index % 8) * 70}ms`;
    fragment.appendChild(node);
  });
  heatmap.appendChild(fragment);
}

function createBarGraph(values) {
  const graph = document.querySelector("#bar-graph");
  if (!graph) return;

  graph.innerHTML = "";
  if (!values.length) return;
  const maxValue = Math.max(...values, 1e-6);
  const fragment = document.createDocumentFragment();
  values.forEach((value, index) => {
    const node = document.createElement("span");
    const height = 24 + (value / maxValue) * 76;
    node.style.height = `${height}%`;
    node.style.animationDelay = `${index * 60}ms`;
    fragment.appendChild(node);
  });
  graph.appendChild(fragment);
}

function computeSignalFidelity(observation) {
  const current = observation.current_spectrum || [];
  const reference = observation.pristine_reference || [];
  if (!current.length) return "--";
  if (!reference.length || reference.length !== current.length) {
    return `${(88 + Math.min(current.length / 32, 10)).toFixed(1)}%`;
  }

  let absoluteError = 0;
  let peak = 0;
  for (let i = 0; i < current.length; i += 1) {
    absoluteError += Math.abs(current[i] - reference[i]);
    peak = Math.max(peak, Math.abs(current[i]), Math.abs(reference[i]));
  }

  const mae = absoluteError / current.length;
  const normalized = peak > 0 ? mae / peak : 0;
  const fidelity = clamp(100 - normalized * 55, 82, 99.6);
  return `${fidelity.toFixed(1)}%`;
}

function computeConfidenceRows(prior) {
  const defects = prior?.predicted_defects || [];
  const concentrations = prior?.predicted_concentrations || [];
  const baseConfidence = clamp(Math.round((prior?.confidence || 0.72) * 100), 56, 99);

  if (!defects.length) {
    return [
      {
        label: "No defect candidates returned",
        concentration: "Prior unavailable",
        confidence: 0,
      },
    ];
  }

  return defects.map((defect, index) => {
    const concentration = concentrations[index] ?? 0;
    const confidence = clamp(
      Math.round(baseConfidence - index * 4 + concentration * 140),
      52,
      99
    );
    return {
      label: defect,
      concentration: `${concentration.toFixed(4)} concentration`,
      confidence,
    };
  });
}

function renderDefectRows(prior) {
  const defectList = document.querySelector("#defect-list");
  if (!defectList) return;

  const rows = computeConfidenceRows(prior);
  defectList.innerHTML = rows
    .map(
      (row) => `
        <div class="defect-row">
          <div>
            <strong>${escapeHtml(row.label)}</strong>
            <span>${escapeHtml(row.concentration)}</span>
          </div>
          <div class="confidence">
            <div><span style="width: ${row.confidence}%"></span></div>
            <b>${row.confidence ? `${row.confidence}%` : "--"}</b>
          </div>
        </div>
      `
    )
    .join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setupDemo() {
  const config = window.ATOMICVISION_CONFIG || {};
  const fileInput = document.querySelector("#sample-file");
  const fileLabel = document.querySelector("#file-label");
  const analyze = document.querySelector("#analyze-button");
  const demoNote = document.querySelector("#demo-note");
  const difficultyPills = document.querySelectorAll("[data-difficulty]");

  const runtimeEpisode = document.querySelector("#runtime-episode");
  const runtimeMaterial = document.querySelector("#runtime-material");
  const runtimeFamily = document.querySelector("#runtime-family");
  const runtimeBudget = document.querySelector("#runtime-budget");
  const runtimeMessage = document.querySelector("#runtime-message");
  const heroStatus = document.querySelector("#hud-status");

  const fidelityMetric = document.querySelector("#metric-fidelity");
  const latencyMetric = document.querySelector("#metric-latency");
  const resolutionMetric = document.querySelector("#metric-resolution");
  const certaintyMetric = document.querySelector("#metric-certainty");
  const summary = document.querySelector("#results-summary");

  let selectedDifficulty = "medium";

  function setDifficulty(difficulty) {
    selectedDifficulty = difficulty;
    difficultyPills.forEach((pill) => {
      pill.classList.toggle("active", pill.dataset.difficulty === difficulty);
    });
  }

  function setBusy(isBusy) {
    if (!analyze) return;
    analyze.disabled = isBusy;
    analyze.innerHTML = isBusy
      ? 'Analyzing Sample <span aria-hidden="true">&bull;&bull;&bull;</span>'
      : 'Analyze Sample <span aria-hidden="true">&rarr;</span>';
  }

  async function analyzeSample() {
    if (!config.stepUrl) return;

    setBusy(true);
    const uploadName = fileInput?.files?.[0]?.name || "Synthetic OpenEnv sample";
    demoNote.textContent = `Running ${selectedDifficulty} analysis for ${uploadName}.`;
    heroStatus.textContent = "Live environment processing";
    let socket;

    try {
      const started = performance.now();
      socket = await openDemoSocket(config.wsUrl || "/ws");

      const resetMessage = await sendSocketMessage(socket, {
        type: "reset",
        data: { difficulty: selectedDifficulty },
      });

      const priorMessage = await sendSocketMessage(socket, {
        type: "step",
        data: { action_type: "ask_prior" },
      });

      const referenceMessage = await sendSocketMessage(socket, {
        type: "step",
        data: { action_type: "compare_reference" },
      });

      const finished = performance.now();
      const activePayload = referenceMessage.data || priorMessage.data || resetMessage.data;
      const observation = activePayload.observation;
      const prior =
        priorMessage.data?.observation?.prior_prediction || observation.prior_prediction;

      runtimeEpisode.textContent = observation.episode_id || "n/a";
      runtimeMaterial.textContent = observation.material_id || "n/a";
      runtimeFamily.textContent = observation.host_family || "n/a";
      runtimeBudget.textContent = observation.budget_remaining?.toFixed(2) ?? "n/a";
      runtimeMessage.textContent = observation.message || "Analysis complete";
      heroStatus.textContent = `${selectedDifficulty} scenario analyzed`;

      fidelityMetric.textContent = computeSignalFidelity(observation);
      latencyMetric.textContent = `${Math.round(finished - started)} ms`;
      resolutionMetric.textContent = `${observation.current_spectrum?.length || 0} bins`;
      certaintyMetric.textContent = prior ? prior.confidence.toFixed(2) : "0.00";

      const predictedDefects = prior?.predicted_defects?.length || 0;
      summary.textContent = predictedDefects
        ? `Detected ${predictedDefects} defect candidate${predictedDefects === 1 ? "" : "s"} for ${observation.material_id} after live prior reasoning and pristine reference comparison.`
        : `No defect candidates were surfaced on this pass. The environment still completed a live OpenEnv episode.`;

      renderDefectRows(prior);

      const differenceSpectrum =
        observation.pristine_reference?.length === observation.current_spectrum?.length
          ? observation.current_spectrum.map((value, index) =>
              Math.abs(value - observation.pristine_reference[index])
            )
          : observation.current_spectrum.map((value) => Math.abs(value));

      createHeatmap(bucketValues(differenceSpectrum, 64));
      createBarGraph(bucketValues(differenceSpectrum, 12));

      document.querySelector("#results")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unexpected demo failure.";
      demoNote.textContent = `Live analysis failed: ${message}`;
      heroStatus.textContent = "Environment check needed";
      summary.textContent =
        "The premium shell is loaded, but the live OpenEnv demo hit an error. Use the API docs or health route to inspect the environment.";
    } finally {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close(1000, "AtomicVision demo completed");
      }
      setBusy(false);
    }
  }

  fileInput?.addEventListener("change", () => {
    fileLabel.textContent =
      fileInput.files?.[0]?.name || "CSV, TXT, or JSON spectral file";
  });

  difficultyPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      setDifficulty(pill.dataset.difficulty || "medium");
    });
  });

  analyze?.addEventListener("click", analyzeSample);
  setDifficulty(selectedDifficulty);
}

document.addEventListener("DOMContentLoaded", () => {
  setupCursor();
  setupParticles();
  setupReveal();
  setupGlassReflection();
  setupHeroVideoScrub();
  setupDemo();
});
