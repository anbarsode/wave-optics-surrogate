"use strict";

const SOLVER_DEFINITIONS = {
  PM: [
    ["exact", "Analytic point lens", "Closed radial point-mass solution, with mass-sheet scaling."],
    ["bgo", "Beyond geometric optics", "Stationary-phase expansion through the chosen BGO order."],
    ["glow", "GLoW reference", "Resolution-checked GLoW calculation; shear can make this slower."],
    ["geometric", "Geometric optics", "High-frequency sum over stationary images."],
  ],
  SIS: [
    ["exact", "Semianalytic SIS", "GLoW radial SIS solver, with mass-sheet scaling."],
    ["bgo", "Beyond geometric optics", "Stationary-phase images plus the SIS centre term."],
    ["glow", "GLoW reference", "Resolution-checked GLoW calculation; shear can make this slower."],
    ["geometric", "Geometric optics", "High-frequency sum over stationary images."],
  ],
};

const SOLVER_CONTROLS = [
  {id: "frequency_samples", label: "FREQUENCY POINTS", min: 48, max: 320, step: 8, value: 144, unit: "", tip: "Number of logarithmically spaced frequencies. More points give smoother curves but take longer."},
  {id: "sky_resolution", label: "SKY GRID", min: 48, max: 180, step: 6, value: 90, unit: "²", tip: "Number of lens-plane samples along each direction used for maps and contours."},
  {id: "bgo_order", label: "BGO ORDER", min: 0, max: 10, step: 1, value: 2, unit: "", tip: "Highest inverse-frequency correction kept beyond geometric optics."},
  {id: "time_samples", label: "TIME POINTS", min: 64, max: 280, step: 8, value: 144, unit: "", tip: "Number of samples used for the displayed inverse transform F(t)."},
  {id: "glow_tolerance", label: "GLOW ERROR", min: -3, max: -1, step: 0.1, value: -1.7, unit: "", tip: "Target relative L2 agreement between consecutive GLoW resolutions." , transform: v => 10 ** v, format: v => (10 ** v).toPrecision(2)},
  {id: "glow_budget", label: "GLOW BUDGET", min: 0.5, max: 8, step: 0.5, value: 2, unit: " s", tip: "Maximum preferred time for one complete GLoW frequency-array evaluation."},
];

const PARAMETER_CONTROLS = [
  {id: "impact", label: "IMPACT y", min: 0.05, max: 2.5, step: 0.01, value: 0.5, unit: "", tip: "Horizontal source position in units of the lens scale. Smaller values mean closer alignment."},
  {id: "strength", label: "PROFILE b", min: 0.2, max: 2, step: 0.02, value: 1, unit: "", tip: "Dimensionless strength of the circular point-mass or SIS profile."},
  {id: "convergence", label: "CONVERGENCE κ", min: -0.2, max: 0.6, step: 0.01, value: 0, unit: "", tip: "Uniform focusing from the surrounding lens environment."},
  {id: "shear1", label: "SHEAR γ₁", min: -0.35, max: 0.35, step: 0.01, value: 0, unit: "", tip: "First component of external tidal stretching, aligned with the horizontal and vertical axes."},
  {id: "shear2", label: "SHEAR γ₂", min: -0.35, max: 0.35, step: 0.01, value: 0, unit: "", tip: "Second component of external tidal stretching, oriented at 45 degrees."},
  {id: "z_lens", label: "LENS REDSHIFT", min: 0.01, max: 3, step: 0.01, value: 0.5, unit: "", tip: "Cosmological redshift of the lens. Planck18 is used for angular-diameter distances."},
  {id: "z_source", label: "SOURCE REDSHIFT", min: 0.05, max: 8, step: 0.05, value: 1.5, unit: "", tip: "Cosmological redshift of the source; it should be larger than the lens redshift."},
  {id: "mass", label: "LENS MASS M_Lz", min: 0, max: 7, step: 0.05, value: 2, unit: "", tip: "Total lens mass multiplied by (1 + lens redshift), in solar masses. It converts f to dimensionless w.", transform: v => 10 ** v, format: v => `${formatNumber(10 ** v)} M☉`},
  {id: "f_min", label: "GW f MIN", min: -1, max: 3, step: 0.05, value: 1, unit: "", tip: "Lowest gravitational-wave frequency shown. The mass converts this value to w.", transform: v => 10 ** v, format: v => formatFrequency(10 ** v)},
  {id: "f_max", label: "GW f MAX", min: 1, max: 5, step: 0.05, value: 3, unit: "", tip: "Highest gravitational-wave frequency shown. The mass converts this value to w.", transform: v => 10 ** v, format: v => formatFrequency(10 ** v)},
];

const PANEL_TYPES = [
  ["sky", "Sky view"],
  ["frequency", "Frequency domain"],
  ["time", "Time domain"],
  ["complex", "Complex frequency plane"],
];

const SKY_LAYERS = [
  ["critical", "Critical curves & caustics"],
  ["source", "Source & geometric images"],
  ["delayColors", "Time-delay colours"],
  ["delayContours", "Time-delay contours"],
  ["magnification", "Geometric magnification"],
  ["negative", "Negative-magnification region"],
  ["mapped", "Mapped source-plane cells"],
  ["residual", "|y − x + α(x)|"],
];

const STYLE = [
  {color: "#39d3d6", dash: []},
  {color: "#ff7d6e", dash: [8, 4]},
  {color: "#f2c14e", dash: [2, 4]},
  {color: "#46a8ff", dash: [11, 3, 2, 3]},
];

const state = {
  model: "PM",
  solvers: new Set(["exact", "bgo"]),
  values: {},
  panels: {sky: 1, frequency: 1, time: 1, complex: 1},
  skyLayers: new Set(["critical", "source", "delayColors", "delayContours"]),
  data: null,
  plots: [],
  autoTimer: null,
  inFlight: false,
  queued: false,
  gridRows: 0,
  rowSizes: [],
};

function formatNumber(value) {
  if (Math.abs(value) >= 1e4 || (Math.abs(value) > 0 && Math.abs(value) < 0.01)) return value.toExponential(1);
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function formatFrequency(value) {
  if (value >= 1e3) return `${formatNumber(value / 1e3)} kHz`;
  return `${formatNumber(value)} Hz`;
}

function createSlider(definition, order) {
  const fragment = document.getElementById("slider-template").content.cloneNode(true);
  const label = fragment.querySelector("label");
  const input = fragment.querySelector("input");
  const output = fragment.querySelector("output");
  label.dataset.controlId = definition.id;
  label.dataset.order = order;
  label.dataset.tip = definition.tip;
  fragment.querySelector(".label-text").textContent = definition.label;
  input.id = `control-${definition.id}`;
  input.min = definition.min;
  input.max = definition.max;
  input.step = definition.step;
  input.value = definition.value;
  input.setAttribute("aria-label", definition.label);
  input.title = definition.tip;
  state.values[definition.id] = definition.value;
  const show = () => {
    const raw = Number(input.value);
    output.value = definition.format ? definition.format(raw) : `${formatNumber(raw)}${definition.unit || ""}`;
  };
  show();
  input.addEventListener("input", () => {
    state.values[definition.id] = Number(input.value);
    show();
    scheduleRefresh();
  });
  return fragment;
}

function buildControls() {
  const solverContainer = document.getElementById("solver-controls");
  SOLVER_CONTROLS.forEach((definition, index) => solverContainer.append(createSlider(definition, index)));
  const parameterContainer = document.getElementById("parameter-controls");
  PARAMETER_CONTROLS.forEach((definition, index) => parameterContainer.append(createSlider(definition, index)));
  requestAnimationFrame(layoutAllControls);
}

function buildSolverOptions() {
  const container = document.getElementById("solver-options");
  container.replaceChildren();
  for (const [id, label, note] of SOLVER_DEFINITIONS[state.model]) {
    const row = document.createElement("label");
    row.className = "tip";
    row.dataset.tip = note;
    const name = document.createElement("span");
    name.textContent = label;
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.solvers.has(id);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.solvers.add(id);
      else if (state.solvers.size > 1) state.solvers.delete(id);
      else checkbox.checked = true;
      updateSolverSummary();
      scheduleRefresh();
    });
    row.append(name, checkbox);
    container.append(row);
  }
  updateSolverSummary();
}

function updateSolverSummary() {
  document.getElementById("solver-summary").textContent = `${state.solvers.size} selected`;
}

function buildPanelOptions() {
  const container = document.getElementById("panel-options");
  container.replaceChildren();
  for (const [id, label] of PANEL_TYPES) {
    const row = document.createElement("label");
    row.className = "panel-count-row";
    const name = document.createElement("span");
    name.textContent = label;
    const input = document.createElement("input");
    input.type = "number";
    input.min = 0;
    input.max = 2;
    input.step = 1;
    input.value = state.panels[id];
    input.setAttribute("aria-label", `${label} panel count`);
    input.addEventListener("change", () => {
      state.panels[id] = Math.max(0, Math.min(2, Number(input.value) || 0));
      if (Object.values(state.panels).reduce((a, b) => a + b, 0) === 0) {
        state.panels[id] = 1;
        input.value = 1;
      }
      renderPanels();
    });
    row.append(name, input);
    container.append(row);
  }
}

function buildRequest() {
  const parameters = {};
  for (const definition of PARAMETER_CONTROLS) {
    const raw = state.values[definition.id];
    parameters[definition.id] = definition.transform ? definition.transform(raw) : raw;
  }
  const solverSettings = {};
  for (const definition of SOLVER_CONTROLS) {
    const raw = state.values[definition.id];
    solverSettings[definition.id] = definition.transform ? definition.transform(raw) : raw;
  }
  return {model: state.model, solvers: [...state.solvers], parameters, solver_settings: solverSettings};
}

function layoutControlGroup(containerId) {
  const container = document.getElementById(containerId);
  const more = document.getElementById(container.dataset.overflowMenu);
  const popover = more.querySelector(".overflow-popover");
  const controls = [...container.querySelectorAll(".mini-control"), ...popover.querySelectorAll(".mini-control")]
    .sort((a, b) => Number(a.dataset.order) - Number(b.dataset.order));
  controls.forEach(control => container.append(control));
  more.classList.add("hidden");
  const halfDefaultWidth = 85;
  if (controls.length * halfDefaultWidth <= container.clientWidth) return;
  more.classList.remove("hidden");
  const available = Math.max(halfDefaultWidth, container.clientWidth - 4);
  const visible = Math.max(1, Math.floor(available / 170));
  controls.slice(visible).forEach(control => popover.append(control));
}

function layoutAllControls() {
  layoutControlGroup("solver-controls");
  layoutControlGroup("parameter-controls");
}

function scheduleRefresh() {
  const delay = Number(document.getElementById("refresh-delay").value);
  clearTimeout(state.autoTimer);
  if (delay > 0) state.autoTimer = setTimeout(refresh, delay);
}

function setStatus(kind, text) {
  const status = document.getElementById("status");
  status.className = `status ${kind}`;
  status.lastElementChild.textContent = text;
}

async function refresh() {
  clearTimeout(state.autoTimer);
  if (state.inFlight) {
    state.queued = true;
    return;
  }
  state.inFlight = true;
  const start = performance.now();
  setStatus("working", "Calculating…");
  try {
    const response = await fetch("/api/calculate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(buildRequest()),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Calculation failed");
    state.data = result;
    state.plots.forEach(plot => plot.setData(result));
    updateGeometry(result);
    const elapsed = (performance.now() - start) / 1000;
    const message = result.messages.length ? result.messages[0] : `Updated in ${elapsed.toFixed(2)} s`;
    setStatus(result.messages.length ? "error" : "ready", message);
  } catch (error) {
    setStatus("error", String(error.message || error));
  } finally {
    state.inFlight = false;
    if (state.queued) {
      state.queued = false;
      refresh();
    }
  }
}

function updateGeometry(data) {
  document.getElementById("image-count").textContent = `${data.images.length} image${data.images.length === 1 ? "" : "s"}`;
  const list = document.getElementById("image-list");
  list.replaceChildren();
  data.images.forEach(image => {
    const chip = document.createElement("div");
    chip.className = "image-chip";
    chip.style.borderLeftColor = plasma((image.log_magnification + 1) / 4);
    chip.innerHTML = `<b>${image.order} · ${image.type}</b> <span>μ=${formatNumber(image.magnification)} · Δt=${formatDelay(image.delay_seconds)}</span>`;
    list.append(chip);
  });
  const c = data.cosmology;
  document.getElementById("cosmology-note").innerHTML = `Planck18 · D<sub>L</sub> ${formatNumber(c.d_l_mpc)} Mpc · θ<sub>E</sub> ${formatNumber(c.theta_e_mas)} mas`;
}

function formatDelay(seconds) {
  if (Math.abs(seconds) < 1e-3) return `${formatNumber(seconds * 1e6)} μs`;
  if (Math.abs(seconds) < 1) return `${formatNumber(seconds * 1e3)} ms`;
  return `${formatNumber(seconds)} s`;
}

function renderPanels() {
  state.plots.forEach(plot => plot.destroy());
  state.plots = [];
  const grid = document.getElementById("panel-grid");
  grid.replaceChildren();
  const specifications = [];
  for (const [type] of PANEL_TYPES) {
    for (let i = 0; i < state.panels[type]; i++) specifications.push({type, index: i});
  }
  const rows = Math.ceil(specifications.length / 2);
  if (state.gridRows !== rows) {
    state.gridRows = rows;
    state.rowSizes = Array.from({length: rows}, () => 1 / rows);
  }
  grid.style.gridTemplateRows = state.rowSizes.map(size => `minmax(0, ${size}fr)`).join(" ");
  grid.style.gridTemplateColumns = specifications.length === 1
    ? "minmax(0, 1fr)"
    : "minmax(0, var(--column-split)) minmax(0, calc(100% - var(--column-split)))";
  document.getElementById("vertical-splitter").classList.toggle("hidden", specifications.length < 2);
  renderRowSplitters(rows);
  for (const specification of specifications) {
    const panel = document.createElement("article");
    panel.className = "plot-panel";
    panel.innerHTML = `<div class="panel-title">${panelTitle(specification.type, specification.index)}</div><canvas></canvas><div class="panel-tools"><button class="reset-view" type="button" title="Reset this panel view">⌂</button></div>`;
    if (specification.type === "sky") panel.querySelector(".panel-tools").prepend(createLayerMenu());
    grid.append(panel);
    const plot = new CanvasPlot(panel, specification.type);
    state.plots.push(plot);
    if (state.data) plot.setData(state.data);
  }
  document.getElementById("panel-summary").textContent = `${specifications.length} panel${specifications.length === 1 ? "" : "s"}`;
}

function renderRowSplitters(rows) {
  const visualization = document.getElementById("visualization");
  visualization.querySelectorAll(".dynamic-row-splitter").forEach(element => element.remove());
  for (let boundary = 0; boundary < rows - 1; boundary++) {
    const splitter = document.createElement("div");
    splitter.className = "grid-splitter horizontal dynamic-row-splitter";
    splitter.title = "Drag to resize rows";
    splitter.dataset.boundary = boundary;
    positionRowSplitter(splitter);
    splitter.addEventListener("pointerdown", event => startRowResize(event, splitter));
    visualization.append(splitter);
  }
}

function positionRowSplitter(splitter) {
  const boundary = Number(splitter.dataset.boundary);
  const position = state.rowSizes.slice(0, boundary + 1).reduce((sum, value) => sum + value, 0);
  splitter.style.top = `${position * 100}%`;
}

function startRowResize(event, splitter) {
  splitter.setPointerCapture(event.pointerId);
  const boundary = Number(splitter.dataset.boundary);
  const before = state.rowSizes.slice(0, boundary).reduce((sum, value) => sum + value, 0);
  const pairTotal = state.rowSizes[boundary] + state.rowSizes[boundary + 1];
  const move = moved => {
    const rect = document.getElementById("visualization").getBoundingClientRect();
    const ratio = (moved.clientY - rect.top) / rect.height;
    const upper = Math.max(.1, Math.min(pairTotal - .1, ratio - before));
    state.rowSizes[boundary] = upper;
    state.rowSizes[boundary + 1] = pairTotal - upper;
    document.getElementById("panel-grid").style.gridTemplateRows = state.rowSizes.map(size => `minmax(0, ${size}fr)`).join(" ");
    document.querySelectorAll(".dynamic-row-splitter").forEach(positionRowSplitter);
  };
  const finish = () => {
    splitter.removeEventListener("pointermove", move);
    splitter.removeEventListener("pointerup", finish);
    state.plots.forEach(plot => plot.resize());
  };
  splitter.addEventListener("pointermove", move);
  splitter.addEventListener("pointerup", finish);
}

function panelTitle(type, index) {
  const names = {sky: "SKY VIEW", frequency: "FREQUENCY DOMAIN", time: "TIME DOMAIN", complex: "COMPLEX F(w)"};
  return `${names[type]}${index ? ` · ${index + 1}` : ""}`;
}

function createLayerMenu() {
  const menu = document.createElement("div");
  menu.className = "hover-menu layer-menu";
  menu.innerHTML = `<button type="button" title="Choose sky layers">LAYERS</button><div class="menu-popover checklist"></div>`;
  const popover = menu.querySelector(".menu-popover");
  for (const [id, label] of SKY_LAYERS) {
    const row = document.createElement("label");
    const name = document.createElement("span");
    name.textContent = label;
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.skyLayers.has(id);
    checkbox.addEventListener("change", () => {
      checkbox.checked ? state.skyLayers.add(id) : state.skyLayers.delete(id);
      state.plots.filter(plot => plot.type === "sky").forEach(plot => plot.draw());
    });
    row.append(name, checkbox);
    popover.append(row);
  }
  return menu;
}

class CanvasPlot {
  constructor(panel, type) {
    this.panel = panel;
    this.canvas = panel.querySelector("canvas");
    this.context = this.canvas.getContext("2d");
    this.type = type;
    this.data = null;
    this.view = null;
    this.drag = null;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(panel);
    panel.querySelector(".reset-view").addEventListener("click", () => this.reset());
    this.canvas.addEventListener("wheel", event => this.zoom(event), {passive: false});
    this.canvas.addEventListener("pointerdown", event => this.startPan(event));
    this.canvas.addEventListener("pointermove", event => this.pan(event));
    this.canvas.addEventListener("pointerup", () => this.endPan());
    this.canvas.addEventListener("pointercancel", () => this.endPan());
  }

  destroy() { this.resizeObserver.disconnect(); }

  resize() {
    const rect = this.panel.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.max(1, Math.round(rect.width * ratio));
    this.canvas.height = Math.max(1, Math.round(rect.height * ratio));
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.width = rect.width;
    this.height = rect.height;
    this.draw();
  }

  setData(data) {
    this.data = data;
    if (!this.view) this.view = this.defaultView();
    this.draw();
  }

  defaultView() {
    if (!this.data) return {x0: -1, x1: 1, y0: -1, y1: 1};
    if (this.type === "sky") {
      const [x0, x1, y0, y1] = this.data.field.extent;
      return {x0, x1, y0, y1};
    }
    if (this.type === "frequency") return {x0: Math.log10(this.data.frequency_hz[0]), x1: Math.log10(this.data.frequency_hz.at(-1)), y0: 0, y1: 1};
    if (this.type === "complex") return rangeForComplex(this.data);
    const curves = Object.values(this.data.curves);
    const time = curves[0]?.time || [0, 1];
    return {x0: time[0], x1: time.at(-1), y0: 0, y1: 1};
  }

  reset() { this.view = this.defaultView(); this.draw(); }

  zoom(event) {
    if (!this.view) return;
    event.preventDefault();
    const rect = this.canvas.getBoundingClientRect();
    const px = (event.clientX - rect.left) / rect.width;
    const py = 1 - (event.clientY - rect.top) / rect.height;
    const factor = Math.exp(event.deltaY * 0.0012);
    const x = this.view.x0 + px * (this.view.x1 - this.view.x0);
    const y = this.view.y0 + py * (this.view.y1 - this.view.y0);
    this.view.x0 = x + (this.view.x0 - x) * factor;
    this.view.x1 = x + (this.view.x1 - x) * factor;
    if (this.type === "sky" || this.type === "complex") {
      this.view.y0 = y + (this.view.y0 - y) * factor;
      this.view.y1 = y + (this.view.y1 - y) * factor;
    }
    this.draw();
  }

  startPan(event) {
    this.canvas.setPointerCapture(event.pointerId);
    this.drag = {x: event.clientX, y: event.clientY, view: {...this.view}};
  }

  pan(event) {
    if (!this.drag) return;
    const dx = (event.clientX - this.drag.x) / this.width * (this.drag.view.x1 - this.drag.view.x0);
    const dy = (event.clientY - this.drag.y) / this.height * (this.drag.view.y1 - this.drag.view.y0);
    this.view.x0 = this.drag.view.x0 - dx;
    this.view.x1 = this.drag.view.x1 - dx;
    if (this.type === "sky" || this.type === "complex") {
      this.view.y0 = this.drag.view.y0 + dy;
      this.view.y1 = this.drag.view.y1 + dy;
    }
    this.draw();
  }

  endPan() { this.drag = null; }

  draw() {
    if (!this.context || !this.width || !this.height) return;
    this.context.clearRect(0, 0, this.width, this.height);
    this.context.fillStyle = "#0a141c";
    this.context.fillRect(0, 0, this.width, this.height);
    if (!this.data) return drawEmpty(this.context, this.width, this.height);
    if (this.type === "sky") this.drawSky();
    else if (this.type === "frequency") this.drawFrequency();
    else if (this.type === "time") this.drawTime();
    else this.drawComplex();
  }

  drawSky() {
    const box = {left: 43, right: this.width - 14, top: 28, bottom: this.height - 27};
    const f = this.data.field;
    if (state.skyLayers.has("residual")) drawRaster(this.context, box, this.view, f, f.residual, value => magma((value + 5) / 6));
    else if (state.skyLayers.has("magnification")) drawRaster(this.context, box, this.view, f, f.log_magnification, value => plasma((value + 1) / 4));
    else if (state.skyLayers.has("delayColors")) {
      const values = f.delay;
      const sorted = [...values].sort((a, b) => a - b);
      const lo = sorted[Math.floor(sorted.length * .03)];
      const hi = sorted[Math.floor(sorted.length * .92)];
      drawRaster(this.context, box, this.view, f, values, value => viridis((value - lo) / Math.max(hi - lo, 1e-9)));
    }
    drawAxes(this.context, box, this.view, "x₁ / θ₀", "x₂ / θ₀");
    if (state.skyLayers.has("negative")) drawMask(this.context, box, this.view, f, f.negative, "rgba(70,168,255,.28)");
    if (state.skyLayers.has("delayContours")) drawSegments(this.context, box, this.view, f.delay_contours, "rgba(238,245,248,.42)", 0.7);
    if (state.skyLayers.has("mapped")) drawPolygons(this.context, box, this.view, f.mapped_cells, "rgba(102,209,143,.34)");
    if (state.skyLayers.has("critical")) {
      drawSegments(this.context, box, this.view, f.critical, "#39d3d6", 1.6);
      drawSegments(this.context, box, this.view, f.caustic, "#f2c14e", 1.5);
    }
    if (state.skyLayers.has("source")) drawImages(this.context, box, this.view, this.data);
    drawSkyLegend(this.context, box, state.skyLayers);
    if (state.skyLayers.has("source")) drawMagnificationColorbar(this.context, box);
  }

  drawFrequency() {
    const curves = Object.entries(this.data.curves);
    if (!curves.length) return drawEmpty(this.context, this.width, this.height, "No selected solver returned a curve");
    const rows = curves.length > 1 ? 4 : 2;
    const plotTop = 27;
    const plotBottom = this.height - 20;
    const rowHeight = (plotBottom - plotTop) / rows;
    const frequencyX = this.data.frequency_hz.map(Math.log10);
    const reference = curves[Math.min(1, curves.length - 1)][1];
    const labels = rows === 4 ? ["|F|", "arg F", "Δ|F|", "Δ arg"] : ["|F|", "arg F"];
    labels.forEach((label, row) => {
      const box = {left: 47, right: this.width - 12, top: plotTop + row * rowHeight + 3, bottom: plotTop + (row + 1) * rowHeight - 4};
      const series = curves.map(([id, curve]) => {
        if (row === 0) return curve.amplitude;
        if (row === 1) return curve.phase;
        if (row === 2) return curve.amplitude.map((value, i) => Math.abs(value - reference.amplitude[i]) / Math.max(reference.amplitude[i], 1e-9));
        return curve.phase.map((value, i) => Math.abs(value - reference.phase[i]));
      });
      const range = paddedRange(series.flat());
      drawSubplot(this.context, box, this.view.x0, this.view.x1, range[0], range[1], frequencyX, series, curves, label, row === rows - 1 ? "f [Hz]" : "", true);
    });
    drawCurveLegend(this.context, curves);
  }

  drawTime() {
    const curves = Object.entries(this.data.curves);
    if (!curves.length) return drawEmpty(this.context, this.width, this.height, "No selected solver returned a curve");
    const rows = curves.length > 1 ? 2 : 1;
    const reference = curves[Math.min(1, curves.length - 1)][1];
    for (let row = 0; row < rows; row++) {
      const box = {left: 47, right: this.width - 12, top: 28 + row * (this.height - 50) / rows, bottom: 24 + (row + 1) * (this.height - 50) / rows};
      const series = curves.map(([, curve]) => row === 0 ? curve.time_response : curve.time_response.map((value, i) => value - reference.time_response[i]));
      const range = paddedRange(series.flat());
      drawSubplot(this.context, box, this.view.x0, this.view.x1, range[0], range[1], curves[0][1].time, series, curves, row ? "ΔF(t)" : "F(t)", row === rows - 1 ? "dimensionless delay t" : "");
    }
    drawCurveLegend(this.context, curves);
  }

  drawComplex() {
    const curves = Object.entries(this.data.curves);
    if (!curves.length) return drawEmpty(this.context, this.width, this.height, "No selected solver returned a curve");
    const columns = curves.length > 1 ? 2 : 1;
    const reference = curves[Math.min(1, curves.length - 1)][1];
    for (let column = 0; column < columns; column++) {
      const width = (this.width - 50) / columns;
      const box = {left: 42 + column * width, right: 34 + (column + 1) * width, top: 30, bottom: this.height - 28};
      let columnView = this.view;
      if (column) {
        const residualValues = curves.flatMap(([, curve]) => curve.real.map((value, index) => value - reference.real[index]).concat(curve.imag.map((value, index) => value - reference.imag[index])));
        const [lo, hi] = paddedRange(residualValues);
        columnView = {x0: lo, x1: hi, y0: lo, y1: hi};
      }
      drawAxes(this.context, box, columnView, column ? "Re ΔF" : "Re F", column ? "Im ΔF" : "Im F");
      curves.forEach(([id, curve], index) => {
        const xs = column ? curve.real.map((value, i) => value - reference.real[i]) : curve.real;
        const ys = column ? curve.imag.map((value, i) => value - reference.imag[i]) : curve.imag;
        drawLine(this.context, box, columnView, xs, ys, STYLE[index % STYLE.length]);
      });
    }
    drawCurveLegend(this.context, curves);
  }
}

function drawEmpty(context, width, height, message = "Calculating the first view…") {
  context.fillStyle = "#8599a7";
  context.font = "11px system-ui";
  context.textAlign = "center";
  context.fillText(message, width / 2, height / 2);
}

function paddedRange(values) {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return [-1, 1];
  let lo = Math.min(...finite), hi = Math.max(...finite);
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = .08 * (hi - lo);
  return [lo - pad, hi + pad];
}

function rangeForComplex(data) {
  const curves = Object.values(data.curves);
  const values = curves.flatMap(curve => [...curve.real, ...curve.imag]);
  const [lo, hi] = paddedRange(values);
  return {x0: lo, x1: hi, y0: lo, y1: hi};
}

function drawAxes(context, box, view, xLabel, yLabel, xFormatter = formatNumber, yFormatter = formatNumber) {
  context.save();
  context.strokeStyle = "#20323f";
  context.fillStyle = "#78909e";
  context.lineWidth = 1;
  context.font = "9px system-ui";
  context.textAlign = "center";
  for (let i = 0; i <= 4; i++) {
    const x = box.left + i / 4 * (box.right - box.left);
    const y = box.bottom - i / 4 * (box.bottom - box.top);
    context.beginPath(); context.moveTo(x, box.top); context.lineTo(x, box.bottom); context.stroke();
    context.beginPath(); context.moveTo(box.left, y); context.lineTo(box.right, y); context.stroke();
    const xv = view.x0 + i / 4 * (view.x1 - view.x0);
    const yv = view.y0 + i / 4 * (view.y1 - view.y0);
    context.fillText(xFormatter(xv), x, box.bottom + 12);
    context.textAlign = "right"; context.fillText(yFormatter(yv), box.left - 5, y + 3); context.textAlign = "center";
  }
  context.fillStyle = "#9eb0bb";
  context.fillText(xLabel, (box.left + box.right) / 2, box.bottom + 24);
  context.save(); context.translate(10, (box.top + box.bottom) / 2); context.rotate(-Math.PI / 2); context.fillText(yLabel, 0, 0); context.restore();
  context.restore();
}

function dataToPixel(box, view, x, y) {
  return [
    box.left + (x - view.x0) / (view.x1 - view.x0) * (box.right - box.left),
    box.bottom - (y - view.y0) / (view.y1 - view.y0) * (box.bottom - box.top),
  ];
}

function drawLine(context, box, view, xs, ys, style) {
  context.save();
  context.beginPath();
  context.rect(box.left, box.top, box.right - box.left, box.bottom - box.top);
  context.clip();
  context.beginPath();
  let started = false;
  for (let i = 0; i < xs.length; i++) {
    if (!Number.isFinite(xs[i]) || !Number.isFinite(ys[i])) { started = false; continue; }
    const [x, y] = dataToPixel(box, view, xs[i], ys[i]);
    if (!started) { context.moveTo(x, y); started = true; } else context.lineTo(x, y);
  }
  context.strokeStyle = style.color;
  context.lineWidth = 1.6;
  context.setLineDash(style.dash || []);
  context.stroke();
  context.restore();
}

function drawSubplot(context, box, x0, x1, y0, y1, xs, series, curves, yLabel, xLabel, logLabels = false) {
  const view = {x0, x1, y0, y1};
  drawAxes(context, box, view, xLabel, yLabel, logLabels ? value => formatNumber(10 ** value) : formatNumber);
  series.forEach((values, index) => drawLine(context, box, view, xs, values, STYLE[index % STYLE.length]));
}

function drawCurveLegend(context, curves) {
  context.save();
  context.font = "9px system-ui";
  let x = 150;
  curves.forEach(([, curve], index) => {
    const style = STYLE[index % STYLE.length];
    context.strokeStyle = style.color; context.setLineDash(style.dash); context.beginPath(); context.moveTo(x, 15); context.lineTo(x + 18, 15); context.stroke();
    context.fillStyle = "#aebfc8"; context.textAlign = "left"; context.fillText(curve.label, x + 23, 18);
    x += Math.min(170, 31 + context.measureText(curve.label).width);
  });
  context.restore();
}

function drawRaster(context, box, view, field, values, colorFunction) {
  const n = field.resolution;
  const image = context.createImageData(n, n);
  for (let i = 0; i < values.length; i++) {
    const color = hexToRgb(colorFunction(values[i]));
    const target = (n - 1 - Math.floor(i / n)) * n + (i % n);
    image.data[4 * target] = color[0]; image.data[4 * target + 1] = color[1]; image.data[4 * target + 2] = color[2]; image.data[4 * target + 3] = 235;
  }
  const offscreen = document.createElement("canvas");
  offscreen.width = n; offscreen.height = n;
  offscreen.getContext("2d").putImageData(image, 0, 0);
  const [fx0, fx1, fy0, fy1] = field.extent;
  const sx = (view.x0 - fx0) / (fx1 - fx0) * n;
  const sy = (fy1 - view.y1) / (fy1 - fy0) * n;
  const sw = (view.x1 - view.x0) / (fx1 - fx0) * n;
  const sh = (view.y1 - view.y0) / (fy1 - fy0) * n;
  context.save(); context.imageSmoothingEnabled = true;
  context.drawImage(offscreen, sx, sy, sw, sh, box.left, box.top, box.right - box.left, box.bottom - box.top);
  context.restore();
}

function drawMask(context, box, view, field, values, color) {
  const n = field.resolution;
  const [x0, x1, y0, y1] = field.extent;
  const dx = (x1 - x0) / (n - 1), dy = (y1 - y0) / (n - 1);
  context.save(); context.fillStyle = color;
  for (let row = 0; row < n; row += 2) for (let col = 0; col < n; col += 2) {
    if (!values[row * n + col]) continue;
    const [px, py] = dataToPixel(box, view, x0 + col * dx, y0 + row * dy);
    const [px2, py2] = dataToPixel(box, view, x0 + (col + 2) * dx, y0 + (row + 2) * dy);
    context.fillRect(px, py2, px2 - px + 1, py - py2 + 1);
  }
  context.restore();
}

function drawSegments(context, box, view, segments, color, width) {
  context.save(); context.beginPath(); context.rect(box.left, box.top, box.right - box.left, box.bottom - box.top); context.clip();
  context.strokeStyle = color; context.lineWidth = width; context.setLineDash([]);
  for (const segment of segments) {
    context.beginPath();
    segment.points.forEach(([x, y], index) => { const point = dataToPixel(box, view, x, y); index ? context.lineTo(...point) : context.moveTo(...point); });
    context.stroke();
  }
  context.restore();
}

function drawPolygons(context, box, view, polygons, color) {
  context.save(); context.strokeStyle = color; context.lineWidth = .55;
  for (const polygon of polygons) {
    context.beginPath(); polygon.forEach(([x, y], index) => { const p = dataToPixel(box, view, x, y); index ? context.lineTo(...p) : context.moveTo(...p); }); context.closePath(); context.stroke();
  }
  context.restore();
}

function drawImages(context, box, view, data) {
  const source = dataToPixel(box, view, data.source[0], data.source[1]);
  context.save(); context.fillStyle = "#ff7d6e"; context.beginPath(); context.arc(...source, 5, 0, 2 * Math.PI); context.fill();
  data.images.forEach(image => {
    const point = dataToPixel(box, view, image.x[0], image.x[1]);
    const major = Math.sqrt(Math.max(image.shape.major, .35));
    const minor = Math.sqrt(Math.max(image.shape.minor, .35));
    context.save(); context.translate(...point); context.rotate(image.shape.angle);
    context.fillStyle = plasma((image.log_magnification + 1) / 4); context.strokeStyle = "#fff"; context.lineWidth = .8;
    context.beginPath(); context.ellipse(0, 0, 4.5 * major, 4.5 * minor, 0, 0, 2 * Math.PI); context.fill(); context.stroke(); context.restore();
    context.fillStyle = "#fff"; context.font = "bold 9px system-ui"; context.fillText(String(image.order), point[0] + 7, point[1] - 7);
  });
  context.restore();
}

function drawMagnificationColorbar(context, box) {
  const x = box.right - 13, y = box.top + 24, height = Math.min(84, box.bottom - box.top - 45);
  if (height < 30) return;
  const gradient = context.createLinearGradient(0, y + height, 0, y);
  for (let i = 0; i <= 8; i++) gradient.addColorStop(i / 8, plasma(i / 8));
  context.save();
  context.fillStyle = "rgba(8,16,23,.72)"; context.fillRect(x - 4, y - 14, 34, height + 23);
  context.fillStyle = gradient; context.fillRect(x, y, 7, height);
  context.strokeStyle = "rgba(255,255,255,.55)"; context.strokeRect(x, y, 7, height);
  context.fillStyle = "#d5e0e5"; context.font = "7px system-ui"; context.textAlign = "left";
  context.fillText("log|μ|", x - 3, y - 4);
  context.fillText("3", x + 10, y + 4); context.fillText("−1", x + 10, y + height);
  context.restore();
}

function drawSkyLegend(context, box, layers) {
  context.save(); context.font = "8px system-ui"; context.textAlign = "left";
  let x = box.left + 7, y = box.top + 13;
  if (layers.has("critical")) {
    [["#39d3d6", "critical"], ["#f2c14e", "caustic"]].forEach(([color, label]) => { context.fillStyle = color; context.fillRect(x, y - 6, 12, 2); context.fillStyle = "#d5e0e5"; context.fillText(label, x + 16, y); x += 63; });
  }
  if (layers.has("source")) { context.fillStyle = "#ff7d6e"; context.beginPath(); context.arc(x + 4, y - 3, 3, 0, 2 * Math.PI); context.fill(); context.fillStyle = "#d5e0e5"; context.fillText("source", x + 11, y); }
  context.restore();
}

function interpolateStops(t, stops) {
  t = Math.max(0, Math.min(1, t));
  const scaled = t * (stops.length - 1), index = Math.min(stops.length - 2, Math.floor(scaled)), amount = scaled - index;
  const a = hexToRgb(stops[index]), b = hexToRgb(stops[index + 1]);
  return rgbToHex(a.map((value, i) => Math.round(value + amount * (b[i] - value))));
}
function plasma(t) { return interpolateStops(t, ["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"]); }
function viridis(t) { return interpolateStops(t, ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]); }
function magma(t) { return interpolateStops(t, ["#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"]); }
function hexToRgb(hex) { const value = Number.parseInt(hex.slice(1), 16); return [(value >> 16) & 255, (value >> 8) & 255, value & 255]; }
function rgbToHex(rgb) { return `#${rgb.map(value => value.toString(16).padStart(2, "0")).join("")}`; }

function makeSplitter(id, direction) {
  const splitter = document.getElementById(id);
  splitter.addEventListener("pointerdown", event => {
    splitter.setPointerCapture(event.pointerId);
    const move = moved => {
      const rect = document.getElementById("visualization").getBoundingClientRect();
      const value = direction === "column" ? (moved.clientX - rect.left) / rect.width : (moved.clientY - rect.top) / rect.height;
      document.documentElement.style.setProperty(direction === "column" ? "--column-split" : "--row-split", `${Math.max(.2, Math.min(.8, value)) * 100}%`);
    };
    const finish = () => { splitter.removeEventListener("pointermove", move); splitter.removeEventListener("pointerup", finish); state.plots.forEach(plot => plot.resize()); };
    splitter.addEventListener("pointermove", move);
    splitter.addEventListener("pointerup", finish);
  });
}

document.getElementById("model-select").addEventListener("change", event => {
  state.model = event.target.value;
  buildSolverOptions();
  scheduleRefresh();
});
document.getElementById("refresh-button").addEventListener("click", refresh);
document.getElementById("refresh-delay").addEventListener("change", scheduleRefresh);
window.addEventListener("resize", () => { layoutAllControls(); state.plots.forEach(plot => plot.resize()); });

buildControls();
buildSolverOptions();
buildPanelOptions();
renderPanels();
makeSplitter("vertical-splitter", "column");
refresh();
