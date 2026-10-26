"use strict";

const GLOW_MODELS = new Set(["PM", "SIS", "SIE", "CIS", "NFW"]);
const BGO_MODELS = new Set(["PM", "SIS"]);
function availableSolvers() {
  const answer = [["bgo", BGO_MODELS.has(state.model) ? "BGO series" : "Geometric optics (BGO K=0)", BGO_MODELS.has(state.model) ? "The BGO expansion. Order zero is the geometric-optics result." : "The order-zero stationary-image sum for this profile."]];
  if (GLOW_MODELS.has(state.model)) answer.unshift(["glow", "GLoW reference", "GLoW's numerical time-domain response and its own regularised FFT."]);
  if ((state.model === "PM" && !(state.values.shear || 0)) || state.model === "SIS") answer.push(["analytic", "Analytic / semi-analytic", "Point-mass mass-sheet solution, or the SIS radial integral with angular quadrature from ab_initio."]);
  if (state.model === "PM" && Math.abs(state.values.shear || 0) > 0 && 1 - (state.values.convergence || 0) > Math.abs(state.values.shear || 0)) {
    answer.push(["series", "PM field series", "The source-derivative series from ab_initio for a point mass in a positive-parity external field."]);
  }
  if (state.model === "PM") {
    answer.push(["uddeepta_raw", "Uddeepta contour", "Ulmer–Goodman's direct point-mass F(t): the contour histogram with its steps and logarithmic peaks, without saddle-peak repair."]);
    answer.push(["uddeepta_repaired", "Uddeepta peak repair", "The same direct Ulmer–Goodman F(t), with peak repair used only for its F(w) reconstruction."]);
  }
  return answer;
}

function externalFieldActive() { return (state.values.convergence || 0) > 0 || Math.abs(state.values.shear || 0) > 0; }

const MODEL_OPTIONS = [
  ["PM", "Point mass"], ["SIS", "Singular isothermal sphere"], ["SIE", "Singular isothermal ellipsoid"],
  ["CIS", "Cored isothermal sphere"], ["NFW", "Navarro–Frenk–White"], ["eNFW", "Elliptical NFW"],
  ["EPL", "Elliptical power law"], ["Hernquist", "Hernquist"],
];

const SOLVER_CONTROLS = [
  {id: "bgo_order", label: "BGO ORDER", min: 0, max: 10, step: 1, value: 0, unit: "", tip: "Order zero is geometric optics. Higher orders add series corrections whenever BGO supports them."},
  {id: "glow_accuracy", label: "GLOW ERROR", min: -3, max: -1, step: .1, value: -2.2, unit: "", tip: "Target agreement between two GLoW resolutions. It also chooses the matching time budget: tighter error means more time.", format: v => `${(10 ** v).toPrecision(2)} · ${formatNumber(Math.min(8, Math.max(.5, .25 / 10 ** v)))} s`},
];

const PARAMETER_CONTROLS = [
  {id: "impact_x", label: "IMPACT y₁", min: -2, max: 2, step: .01, value: .5, unit: "", tip: "Horizontal dimensionless source position. It is independent of y₂; both signs are physically equivalent after reflection."},
  {id: "impact_y", label: "IMPACT y₂", min: -2, max: 2, step: .01, value: 0, unit: "", tip: "Vertical dimensionless source position. It is independent of y₁; both signs are physically equivalent after reflection."},
  {id: "convergence", label: "CONVERGENCE κ", min: -1, max: .95, step: .01, value: 0, unit: "", tip: "External convergence relative to a reference line of sight. Negative values describe an underdensity; positive values describe extra projected mass. Together with shear the app enforces 1 − κ > |γ|."},
  {id: "shear", label: "AXIS SHEAR γ", min: -.95, max: .95, step: .01, value: 0, unit: "", tip: "Tidal stretch aligned with the horizontal and vertical axes. Its sign swaps the stretched axis. The app enforces κ + |γ| < 1, so the far field remains physical."},
  {id: "z_lens", label: "LENS REDSHIFT", min: .01, max: 3, step: .01, value: .5, unit: "", tip: "Cosmological redshift of the lens. Planck18 supplies the distances."},
  {id: "z_source", label: "SOURCE REDSHIFT", min: .05, max: 8, step: .05, value: 1.5, unit: "", tip: "Cosmological redshift of the source; it must exceed the lens redshift."},
  {id: "mass", label: "LENS MASS M_Lz", min: 0, max: 7, step: .05, value: 2, unit: "", tip: "Redshifted total lens mass in solar masses. It fixes the profile's angular scale and converts f to w.", transform: v => 10 ** v, format: v => `${formatNumber(10 ** v)} M☉`},
  {id: "f_min", label: "GW f MIN", min: -1, max: 3, step: .05, value: 1, unit: "", tip: "Lowest gravitational-wave frequency shown.", transform: v => 10 ** v, format: v => formatFrequency(10 ** v)},
  {id: "f_max", label: "GW f MAX", min: 1, max: 5, step: .05, value: 3, unit: "", tip: "Highest gravitational-wave frequency shown.", transform: v => 10 ** v, format: v => formatFrequency(10 ** v)},
  {id: "axis_ratio", label: "AXIS RATIO q", min: .2, max: 1, step: .01, value: .8, unit: "", models: ["SIE", "eNFW", "EPL"], tip: "Minor-to-major axis ratio for the horizontal/vertical elliptical profile. One is circular."},
  {id: "core_radius", label: "CORE RADIUS", min: .005, max: 1, step: .005, value: .08, unit: "", models: ["CIS"], tip: "Finite-density core radius for the cored isothermal sphere."},
  {id: "scale_radius", label: "SCALE RADIUS", min: .02, max: 4, step: .02, value: .3, unit: "", models: ["NFW", "eNFW", "Hernquist"], tip: "Profile transition radius in the natural lens-angle units."},
  {id: "concentration", label: "CONCENTRATION", min: 2, max: 30, step: .5, value: 8, unit: "", models: ["NFW", "eNFW"], tip: "NFW concentration, setting central density relative to the scale radius."},
  {id: "slope", label: "DENSITY SLOPE", min: 1.2, max: 2.8, step: .02, value: 2, unit: "", models: ["EPL"], tip: "Three-dimensional power-law density slope."},
];

const PANEL_CONTROLS = {
  sky: {id: "sky_resolution", label: "GRID", min: 0, max: 15, step: 1, value: 6, tip: "Lens-plane samples per side, with two geometrically spaced choices between each power of two from 32 to 1024.", transform: fineGridSize, format: v => `${fineGridSize(v)}`},
  frequency: {id: "frequency_samples", label: "POINTS", min: 0, max: 15, step: 1, value: 6, tip: "Frequency samples, with two geometrically spaced choices between each power of two from 32 to 1024.", transform: fineGridSize, format: v => `${fineGridSize(v)}`},
  time: {id: "time_samples", label: "POINTS", min: 0, max: 15, step: 1, value: 12, tip: "Shown points from the solver's direct delay histogram or contour response, with two geometrically spaced choices between each power of two.", transform: fineGridSize, format: v => `${fineGridSize(v)}`},
};

function fineGridSize(step) { return Math.round(32 * 2 ** (Number(step) / 3)); }

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

const SUBPANELS = {
  frequency: [["amplitude", "|F(w)|"], ["phase", "Unwrapped phase"], ["amplitudeError", "Relative amplitude error"], ["phaseError", "Phase error"]],
  time: [["response", "F(t) delay density"], ["error", "Delay-density error"]],
  complex: [["value", "F(w)"], ["error", "Complex error"]],
};

const STYLE = [
  {color: "#39d3d6", dash: []},
  {color: "#ff7d6e", dash: [8, 4]},
  {color: "#f2c14e", dash: [2, 4]},
  {color: "#46a8ff", dash: [11, 3, 2, 3]},
  {color: "#66d18f", dash: [4, 3]},
  {color: "#e879f9", dash: [10, 3, 2, 3, 2, 3]},
  {color: "#f59e0b", dash: [1, 3]},
  {color: "#a5b4fc", dash: [12, 4]},
];

const state = {
  model: "PM",
  solvers: new Set(["glow", "bgo"]),
  values: {},
  panels: {sky: 1, frequency: 1, time: 1, complex: 1},
  skyLayers: new Set(["critical", "source", "delayColors", "delayContours"]),
  subpanels: {frequency: new Set(["amplitude", "phase"]), time: new Set(["response"]), complex: new Set(["value"])},
  timeAxis: "linear",
  data: null,
  plots: [],
  autoTimer: null,
  inFlight: false,
  queued: false,
  gridRows: 0,
  rowSizes: [],
};

function formatNumber(value) {
  value = Number(value);
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1e4 || (Math.abs(value) > 0 && Math.abs(value) < 0.01)) return value.toExponential(1);
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function formatFrequency(value) {
  if (value >= 1e3) return `${formatNumber(value / 1e3)} kHz`;
  return `${formatNumber(value)} Hz`;
}

function sliderLimits(definition) {
  let min = definition.min;
  let max = definition.max;
  const convergence = state.values.convergence || 0;
  if (definition.id === "shear") {
    max = Math.max(0, .99 - convergence);
    min = -max;
  } else if (definition.id === "z_lens") {
    max = Math.min(max, (state.values.z_source || 1.5) - .01);
  } else if (definition.id === "z_source") {
    min = Math.max(min, (state.values.z_lens || .5) + .01);
  } else if (definition.id === "f_min") {
    max = Math.min(max, (state.values.f_max || 3) - .05);
  } else if (definition.id === "f_max") {
    min = Math.max(min, (state.values.f_min || 1) + .05);
  }
  return {min, max};
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
  const limits = sliderLimits(definition);
  input.min = limits.min;
  input.max = limits.max;
  input.step = definition.step;
  if (!(definition.id in state.values)) state.values[definition.id] = definition.value;
  state.values[definition.id] = Math.max(Number(input.min), Math.min(Number(input.max), state.values[definition.id]));
  input.value = state.values[definition.id];
  input.setAttribute("aria-label", definition.label);
  input.title = definition.tip;
  const show = () => {
    const raw = Number(input.value);
    output.value = definition.format ? definition.format(raw) : `${formatNumber(raw)}${definition.unit || ""}`;
  };
  show();
  input.addEventListener("input", () => {
    state.values[definition.id] = Number(input.value);
    if (definition.id === "convergence") {
      const shearLimit = Math.max(0, .99 - state.values.convergence);
      state.values.shear = Math.max(-shearLimit, Math.min(shearLimit, state.values.shear || 0));
    }
    show();
    scheduleRefresh();
  });
  // Rebuilding controls during a range-input event removes the slider currently
  // being dragged.  That made kappa/gamma changes appear to stop after one tick.
  // Apply every numerical value immediately, then refresh dependent limits only
  // when the drag is released.
  input.addEventListener("change", () => {
    if (["convergence", "shear", "z_lens", "z_source", "f_min", "f_max"].includes(definition.id)) buildControls();
    if (definition.id === "convergence" || definition.id === "shear") buildSolverOptions();
  });
  return fragment;
}

function buildControls() {
  const solverContainer = document.getElementById("solver-controls");
  solverContainer.replaceChildren();
  SOLVER_CONTROLS.filter(definition => (definition.id !== "bgo_order" || BGO_MODELS.has(state.model)) &&
    (definition.id !== "glow_accuracy" || GLOW_MODELS.has(state.model)))
    .forEach((definition, index) => solverContainer.append(createSlider(definition, index)));
  const parameterContainer = document.getElementById("parameter-controls");
  parameterContainer.replaceChildren();
  PARAMETER_CONTROLS.filter(definition => !definition.models || definition.models.includes(state.model))
    .forEach((definition, index) => parameterContainer.append(createSlider(definition, index)));
  requestAnimationFrame(layoutAllControls);
}

function buildModelOptions() {
  const container = document.getElementById("model-options");
  container.replaceChildren();
  for (const [id, label] of MODEL_OPTIONS) {
    const row = document.createElement("label");
    const name = document.createElement("span");
    name.textContent = label;
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "lens-model";
    radio.value = id;
    radio.checked = state.model === id;
    radio.addEventListener("change", () => {
      if (!radio.checked) return;
      state.model = id;
      document.getElementById("model-summary").textContent = label;
      buildControls();
      buildSolverOptions();
      scheduleRefresh();
    });
    row.append(name, radio);
    container.append(row);
  }
  const current = MODEL_OPTIONS.find(([id]) => id === state.model);
  document.getElementById("model-summary").textContent = current[1];
}

function buildSolverOptions() {
  const container = document.getElementById("solver-options");
  container.replaceChildren();
  const available = availableSolvers();
  const allowed = new Set(available.map(([id]) => id));
  state.solvers = new Set([...state.solvers].filter(id => allowed.has(id)));
  if (!state.solvers.size) state.solvers.add("bgo");
  for (const [id, label, note] of available) {
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
  for (const definition of Object.values(PANEL_CONTROLS)) {
    const raw = state.values[definition.id];
    solverSettings[definition.id] = definition.transform ? definition.transform(raw) : raw;
  }
  return {model: state.model, solvers: [...state.solvers], parameters, solver_settings: solverSettings};
}

function savedState() {
  return {
    format: "gravitational-lensing-state-v1",
    model: state.model,
    solvers: [...state.solvers],
    values: {...state.values},
    panels: {...state.panels},
    sky_layers: [...state.skyLayers],
    subpanels: Object.fromEntries(Object.entries(state.subpanels).map(([name, values]) => [name, [...values]])),
    time_axis: state.timeAxis,
    auto_refresh: {
      mode: document.getElementById("refresh-delay").value,
      seconds: Number(document.getElementById("timed-delay").value),
    },
    grid: {
      column_split: getComputedStyle(document.documentElement).getPropertyValue("--column-split").trim(),
      row_sizes: [...state.rowSizes],
    },
  };
}

async function exportState() {
  try {
    const response = await fetch("/api/states", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(savedState()),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not export this state");
    setStatus("ready", `Saved viz/saved_states/${result.filename}`);
  } catch (error) {
    setStatus("error", String(error.message || error));
  }
}

function applySavedState(saved) {
  if (!saved || saved.format !== "gravitational-lensing-state-v1") {
    throw new Error("This is not a Gravitational Lensing state file");
  }
  if (MODEL_OPTIONS.some(([id]) => id === saved.model)) state.model = saved.model;
  if (Array.isArray(saved.solvers) && saved.solvers.length) state.solvers = new Set(saved.solvers);
  if (saved.values && typeof saved.values === "object") state.values = {...saved.values};
  if (saved.panels && typeof saved.panels === "object") {
    for (const [name] of PANEL_TYPES) {
      if (Number.isFinite(saved.panels[name])) state.panels[name] = Math.max(0, Math.min(2, saved.panels[name]));
    }
  }
  if (Array.isArray(saved.sky_layers)) state.skyLayers = new Set(saved.sky_layers);
  if (saved.subpanels && typeof saved.subpanels === "object") {
    for (const [name] of Object.entries(SUBPANELS)) {
      if (Array.isArray(saved.subpanels[name])) state.subpanels[name] = new Set(saved.subpanels[name]);
    }
  }
  if (saved.time_axis === "linear" || saved.time_axis === "log") state.timeAxis = saved.time_axis;
  const refreshSettings = saved.auto_refresh || {};
  document.getElementById("refresh-delay").value = ["off", "instant", "timed"].includes(refreshSettings.mode) ? refreshSettings.mode : "instant";
  document.getElementById("timed-delay").value = Math.max(.1, Math.min(3, Number(refreshSettings.seconds) || .4));
  document.getElementById("timed-delay-wrap").classList.toggle("hidden", document.getElementById("refresh-delay").value !== "timed");
  if (saved.grid?.column_split) document.documentElement.style.setProperty("--column-split", saved.grid.column_split);
  state.gridRows = 0;
  buildControls();
  buildModelOptions();
  buildSolverOptions();
  buildPanelOptions();
  renderPanels();
  if (Array.isArray(saved.grid?.row_sizes) && saved.grid.row_sizes.length === state.gridRows) {
    state.rowSizes = saved.grid.row_sizes;
    document.getElementById("panel-grid").style.gridTemplateRows = state.rowSizes.map(size => `minmax(0, ${size}fr)`).join(" ");
    document.querySelectorAll(".dynamic-row-splitter").forEach(positionRowSplitter);
  }
  refresh();
}

function importState(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      applySavedState(JSON.parse(reader.result));
      setStatus("ready", `Loaded ${file.name}`);
    } catch (error) {
      setStatus("error", String(error.message || error));
    }
  };
  reader.readAsText(file);
}

function layoutControlGroup(containerId) {
  const container = document.getElementById(containerId);
  const more = document.getElementById(container.dataset.overflowMenu);
  const popover = more.querySelector(".overflow-popover");
  const controls = [...container.querySelectorAll(".mini-control"), ...popover.querySelectorAll(".mini-control")]
    .sort((a, b) => Number(a.dataset.order) - Number(b.dataset.order));
  controls.forEach(control => container.append(control));
  more.classList.add("hidden");
  // Keep labels and values legible before controls reach their strict 85 px
  // minimum.  The remaining settings move to the hover menu on compact screens.
  const minimumUsefulWidth = 120;
  if (controls.length * minimumUsefulWidth <= container.clientWidth) return;
  more.classList.remove("hidden");
  const available = Math.max(1, container.clientWidth - 4);
  const visible = Math.max(1, Math.floor(available / 170));
  controls.slice(visible).forEach(control => popover.append(control));
}

function layoutAllControls() {
  layoutControlGroup("solver-controls");
  layoutControlGroup("parameter-controls");
}

function scheduleRefresh() {
  clearTimeout(state.autoTimer);
  const mode = document.getElementById("refresh-delay").value;
  if (mode === "off") return;
  if (mode === "instant") { state.autoTimer = setTimeout(refresh, 0); return; }
  const seconds = Math.max(.1, Math.min(3, Number(document.getElementById("timed-delay").value) || .4));
  state.autoTimer = setTimeout(refresh, 1000 * seconds);
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
    state.plots.forEach(plot => {
      try { plot.setData(result); }
      catch (error) { throw new Error(`${plot.type} panel: ${error.message || error}`); }
    });
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
  const radius = c.theta_mi_infinite ? "∞" : `${formatNumber(c.theta_mi_mas)} mas`;
  document.getElementById("cosmology-note").innerHTML = `Planck18 · θ<sub>E</sub><sup>MI</sup> ${radius}`;
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
    panel.innerHTML = `<div class="panel-title">${panelTitle(specification.type, specification.index)}</div><canvas></canvas><div class="panel-tools"><button class="reset-view" type="button" title="Reset this panel view">${homeIcon()}</button></div>`;
    const tools = panel.querySelector(".panel-tools");
    if (PANEL_CONTROLS[specification.type]) tools.prepend(createPanelControl(PANEL_CONTROLS[specification.type]));
    if (specification.type === "sky") tools.prepend(createLayerMenu());
    else tools.prepend(createSubpanelMenu(specification.type));
    if (specification.type === "time") tools.prepend(createTimeAxisMenu());
    grid.append(panel);
    const plot = new CanvasPlot(panel, specification.type);
    state.plots.push(plot);
    if (state.data) plot.setData(state.data);
  }
  document.getElementById("panel-summary").textContent = `${specifications.length} panel${specifications.length === 1 ? "" : "s"}`;
}

function homeIcon() {
  return '<svg class="home-icon" aria-hidden="true" viewBox="0 0 576 512"><path d="M575.5 240.4 318.8 15.7c-17.5-15.5-44-15.5-61.5 0L.5 240.4C-13 252.2-1.6 274 16.4 274H64v174c0 35.3 28.7 64 64 64h96V368c0-17.7 14.3-32 32-32h64c17.7 0 32 14.3 32 32v144h96c35.3 0 64-28.7 64-64V274h47.6c18 0 29.4-21.8 16-33.6z"/></svg>';
}

function chevronIcon() {
  return '<svg class="chevron-icon" aria-hidden="true" viewBox="0 0 448 512"><path d="M201.4 137.4c12.5-12.5 32.8-12.5 45.3 0l160 160c12.5 12.5 12.5 32.8 0 45.3s-32.8 12.5-45.3 0L224 205.3 86.6 342.6c-12.5 12.5-32.8 12.5-45.3 0s-12.5-32.8 0-45.3l160-160z"/></svg>';
}

function createPanelControl(definition) {
  const label = document.createElement("label");
  label.className = "panel-control tip";
  label.dataset.tip = definition.tip;
  label.innerHTML = `<span>${definition.label}</span><output></output>`;
  const input = document.createElement("input");
  input.type = "range"; input.min = definition.min; input.max = definition.max; input.step = definition.step;
  if (!(definition.id in state.values)) state.values[definition.id] = definition.value;
  input.value = state.values[definition.id];
  const output = label.querySelector("output");
  const update = () => { output.value = definition.format(Number(input.value)); };
  update();
  input.addEventListener("input", () => {
    state.values[definition.id] = Number(input.value);
    document.querySelectorAll(`.panel-control input`).forEach(other => { if (other !== input && other.parentElement.dataset.tip === definition.tip) other.value = input.value; });
    document.querySelectorAll(`.panel-control output`).forEach(other => { if (other.parentElement.dataset.tip === definition.tip) other.value = definition.format(Number(input.value)); });
    scheduleRefresh();
  });
  label.append(input);
  return label;
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

function createSubpanelMenu(type) {
  const menu = document.createElement("div");
  menu.className = "hover-menu layer-menu";
  menu.innerHTML = `<button type="button" title="Choose visible sub-panels">PANELS ${chevronIcon()}</button><div class="menu-popover checklist"></div>`;
  const popover = menu.querySelector(".menu-popover");
  for (const [id, label] of SUBPANELS[type]) {
    const row = document.createElement("label");
    const name = document.createElement("span"); name.textContent = label;
    const checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.checked = state.subpanels[type].has(id);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.subpanels[type].add(id);
      else if (state.subpanels[type].size > 1) state.subpanels[type].delete(id);
      else checkbox.checked = true;
      state.plots.filter(plot => plot.type === type).forEach(plot => plot.draw());
    });
    row.append(name, checkbox); popover.append(row);
  }
  return menu;
}

function createTimeAxisMenu() {
  const menu = document.createElement("div");
  menu.className = "hover-menu layer-menu time-axis-menu";
  menu.innerHTML = `<button type="button" title="Choose linear or logarithmic time axis">T: ${state.timeAxis === "linear" ? "LIN" : "LOG"} ${chevronIcon()}</button><div class="menu-popover checklist"></div>`;
  const popover = menu.querySelector(".menu-popover");
  for (const [id, label] of [["linear", "Linear delay t"], ["log", "Logarithmic delay t"]]) {
    const row = document.createElement("label");
    const name = document.createElement("span");
    name.textContent = label;
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "time-axis";
    radio.checked = state.timeAxis === id;
    radio.addEventListener("change", () => {
      if (!radio.checked) return;
      state.timeAxis = id;
      state.plots.filter(plot => plot.type === "time").forEach(plot => plot.reset());
      document.querySelectorAll(".time-axis-menu button").forEach(button => {
        button.innerHTML = `T: ${id === "linear" ? "LIN" : "LOG"} ${chevronIcon()}`;
      });
    });
    row.append(name, radio);
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
    const previousDelay = this.type === "time" ? directDelayDomain(this.data) : null;
    this.data = data;
    const nextDelay = this.type === "time" ? directDelayDomain(data) : null;
    const changedDelayDomain = previousDelay && nextDelay &&
      (nextDelay[1] / Math.max(previousDelay[1], 1e-12) > 2 || nextDelay[1] / Math.max(previousDelay[1], 1e-12) < .5);
    if (!this.view || changedDelayDomain) this.view = this.defaultView();
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
    const delay = directDelayDomain(this.data)?.[2] || [0, 1];
    // The interesting steps and saddle logarithms occur near the geometric-image
    // delays.  Showing the full long-time tail by default made that structure a
    // nearly invisible sliver at the left edge of the panel.
    const largestImageDelay = Math.max(.25, ...this.data.images.map(image => image.delay_dimensionless));
    const focusedEnd = Math.min(delay.at(-1), Math.max(4, 4*largestImageDelay));
    if (state.timeAxis === "log") {
      const positive = delay.filter(value => value > 0);
      return {x0: Math.log10(positive[0] || 1e-6), x1: Math.log10(Math.max(focusedEnd, positive[0] || 1e-6)), y0: 0, y1: 1};
    }
    return {x0: delay[0], x1: focusedEnd, y0: 0, y1: 1};
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
    const box = {left: 43, right: this.width - 14, top: 38, bottom: this.height - 27};
    const f = this.data.field;
    let raster = null;
    if (state.skyLayers.has("residual")) {
      raster = {values: f.residual, low: -3, high: 0, label: "log₁₀|y − x + α(x)|"};
    } else if (state.skyLayers.has("magnification")) {
      raster = {values: f.log_magnification, low: -1, high: 3, label: "log₁₀|μ|"};
    }
    else if (state.skyLayers.has("delayColors")) {
      const values = f.delay;
      const sorted = [...values].sort((a, b) => a - b);
      const lo = sorted[Math.floor(sorted.length * .03)];
      const hi = sorted[Math.floor(sorted.length * .92)];
      raster = {values, low: lo, high: Math.max(hi, lo + 1e-9), label: "Δτ"};
    }
    if (raster) drawRaster(this.context, box, this.view, f, raster.values,
      value => viridis((value - raster.low) / Math.max(raster.high - raster.low, 1e-12)));
    drawAxes(this.context, box, this.view, "x₁ / θ₀", "x₂ / θ₀");
    if (state.skyLayers.has("negative")) drawMask(this.context, box, this.view, f, f.negative, "rgba(30,144,255,.30)");
    if (state.skyLayers.has("delayContours")) drawSegments(this.context, box, this.view, f.delay_contours, "rgba(238,245,248,.42)", 0.7);
    if (state.skyLayers.has("mapped")) drawPolygons(this.context, box, this.view, f.mapped_cells, "rgba(102,209,143,.34)");
    if (state.skyLayers.has("critical")) {
      drawSegments(this.context, box, this.view, f.critical, "#39d3d6", 1.6);
      drawSegments(this.context, box, this.view, f.caustic, "#f2c14e", 1.5);
    }
    if (state.skyLayers.has("source")) drawImages(this.context, box, this.view, this.data);
    drawSkyLegend(this.context, box, state.skyLayers);
    if (raster) drawRasterColorbar(this.context, box, raster);
  }

  drawFrequency() {
    const curves = Object.entries(this.data.curves);
    if (!curves.length) return drawEmpty(this.context, this.width, this.height, "No selected solver returned a curve");
    const chosen = [...state.subpanels.frequency].filter(id => curves.length > 1 || !id.endsWith("Error"));
    if (!chosen.length) return drawEmpty(this.context, this.width, this.height, "Choose a frequency sub-panel");
    const rows = chosen.length;
    const plotTop = 38;
    const plotBottom = this.height - 20;
    const rowHeight = (plotBottom - plotTop) / rows;
    const frequencyX = this.data.frequency_hz.map(Math.log10);
    const reference = curves[Math.min(1, curves.length - 1)][1];
    chosen.forEach((kind, row) => {
      const box = {left: 47, right: this.width - 12, top: plotTop + row * rowHeight + 3, bottom: plotTop + (row + 1) * rowHeight - 4};
      const series = curves.map(([id, curve]) => {
        if (kind === "amplitude") return curve.amplitude;
        if (kind === "phase") return curve.phase;
        if (kind === "amplitudeError") return curve.amplitude.map((value, i) => Math.abs(value - reference.amplitude[i]) / Math.max(reference.amplitude[i], 1e-9));
        // Compare phases on the unit circle.  Subtracting separately unwrapped
        // curves turns an otherwise zero phase difference into a false 2π jump.
        return curve.real.map((real, i) => Math.atan2(
          real * reference.imag[i] - curve.imag[i] * reference.real[i],
          real * reference.real[i] + curve.imag[i] * reference.imag[i]
        ));
      });
      const range = paddedRange(series.flat());
      const label = {amplitude: "|F|", phase: "arg F", amplitudeError: "Δ|F|", phaseError: "wrapped Δ arg"}[kind];
      drawSubplot(this.context, box, this.view.x0, this.view.x1, range[0], range[1], frequencyX, series, curves, label, row === rows - 1 ? "f [Hz]" : "", true);
    });
    drawCurveLegend(this.context, curves, this.width);
  }

  drawTime() {
    const curves = Object.entries(this.data.curves).filter(([, curve]) => Array.isArray(curve.delay_density));
    if (!curves.length) return drawEmpty(this.context, this.width, this.height, "Choose a solver with a direct contour or histogram F(t)");
    const chosen = [...state.subpanels.time].filter(id => curves.length > 1 || id !== "error");
    if (!chosen.length) return drawEmpty(this.context, this.width, this.height, "Choose a time sub-panel");
    const rows = chosen.length;
    const reference = curves[Math.min(1, curves.length - 1)][1];
    // Keep the richest direct response as the plotting grid.  In particular, this
    // preserves the Ulmer--Goodman contour steps and logarithmic peaks when it is
    // plotted beside GLoW's deliberately compact response sample.
    const gridCurve = curves.reduce((best, current) => current[1].delay.length > best[1].delay.length ? current : best);
    const rawDelayGrid = gridCurve[1].delay;
    const keep = state.timeAxis === "log" ? rawDelayGrid.map(value => value > 0) : rawDelayGrid.map(() => true);
    const delayGrid = rawDelayGrid.filter((_, index) => keep[index]);
    const displayDelay = state.timeAxis === "log" ? delayGrid.map(Math.log10) : delayGrid;
    chosen.forEach((kind, row) => {
      const plotTop = 38;
      const plotBottom = this.height - 24;
      const box = {left: 47, right: this.width - 12, top: plotTop + row * (plotBottom - plotTop) / rows + 2, bottom: plotTop + (row + 1) * (plotBottom - plotTop) / rows - 4};
      const referenceDensity = interpolateSeries(reference.delay, reference.delay_density, delayGrid);
      const series = curves.map(([, curve]) => {
        const density = interpolateSeries(curve.delay, curve.delay_density, delayGrid);
        return kind === "response" ? density : density.map((value, i) => value - referenceDensity[i]);
      });
      const range = paddedRange(series.flat());
      drawSubplot(this.context, box, this.view.x0, this.view.x1, range[0], range[1], displayDelay, series, curves,
        kind === "error" ? "ΔF(t)" : "F(t)", row === rows - 1 ? (state.timeAxis === "log" ? "log₁₀ t" : "dimensionless delay t") : "");
    });
    drawCurveLegend(this.context, curves, this.width);
  }

  drawComplex() {
    const curves = Object.entries(this.data.curves);
    if (!curves.length) return drawEmpty(this.context, this.width, this.height, "No selected solver returned a curve");
    const chosen = [...state.subpanels.complex].filter(id => curves.length > 1 || id !== "error");
    if (!chosen.length) return drawEmpty(this.context, this.width, this.height, "Choose a complex sub-panel");
    const columns = chosen.length;
    const reference = curves[Math.min(1, curves.length - 1)][1];
    chosen.forEach((kind, column) => {
      const width = (this.width - 50) / columns;
      const box = {left: 42 + column * width, right: 34 + (column + 1) * width, top: 38, bottom: this.height - 28};
      let columnView = this.view;
      if (kind === "error") {
        const residualValues = curves.flatMap(([, curve]) => curve.real.map((value, index) => value - reference.real[index]).concat(curve.imag.map((value, index) => value - reference.imag[index])));
        const [lo, hi] = paddedRange(residualValues);
        columnView = {x0: lo, x1: hi, y0: lo, y1: hi};
      }
      drawAxes(this.context, box, columnView, kind === "error" ? "Re ΔF" : "Re F", kind === "error" ? "Im ΔF" : "Im F");
      curves.forEach(([id, curve], index) => {
        const xs = kind === "error" ? curve.real.map((value, i) => value - reference.real[i]) : curve.real;
        const ys = kind === "error" ? curve.imag.map((value, i) => value - reference.imag[i]) : curve.imag;
        drawLine(this.context, box, columnView, xs, ys, STYLE[index % STYLE.length]);
      });
    });
    drawCurveLegend(this.context, curves, this.width);
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

function interpolateSeries(xs, ys, targetXs) {
  let index = 0;
  return targetXs.map(target => {
    while (index < xs.length - 2 && xs[index + 1] < target) index++;
    const left = xs[index], right = xs[index + 1];
    if (target <= left) return ys[index];
    if (target >= right && index === xs.length - 2) return ys.at(-1);
    const fraction = (target - left) / Math.max(right - left, 1e-12);
    return ys[index] + fraction * (ys[index + 1] - ys[index]);
  });
}

function directDelayDomain(data) {
  if (!data) return null;
  const curves = Object.values(data.curves).filter(curve => Array.isArray(curve.delay_density) && curve.delay.length > 1);
  if (!curves.length) return null;
  const curve = curves.reduce((best, current) => current.delay.length > best.delay.length ? current : best);
  return [curve.delay[0], curve.delay.at(-1), curve.delay];
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

function drawCurveLegend(context, curves, width) {
  // Narrow panels devote their small header to controls.  The colours and line
  // styles still identify curves without printing a legend through those controls.
  if (width < 720) return;
  context.save();
  context.font = "9px system-ui";
  let x = 150;
  curves.forEach(([, curve], index) => {
    const style = STYLE[index % STYLE.length];
    context.strokeStyle = style.color; context.setLineDash(style.dash); context.beginPath(); context.moveTo(x, 29); context.lineTo(x + 18, 29); context.stroke();
    context.fillStyle = "#aebfc8"; context.textAlign = "left"; context.fillText(curve.label, x + 23, 32);
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
  context.save(); context.fillStyle = "#228b22"; context.strokeStyle = "#00ff7f"; context.lineWidth = 2; context.beginPath(); context.arc(...source, 5, 0, 2 * Math.PI); context.fill(); context.stroke();
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

function drawRasterColorbar(context, box, raster) {
  const x = box.right - 13, y = box.top + 24, height = Math.min(84, box.bottom - box.top - 45);
  if (height < 30) return;
  const gradient = context.createLinearGradient(0, y + height, 0, y);
  for (let i = 0; i <= 8; i++) gradient.addColorStop(i / 8, viridis(i / 8));
  context.save();
  context.font = "7px system-ui";
  const labelWidth = Math.max(38, context.measureText(raster.label).width + 8);
  context.fillStyle = "rgba(8,16,23,.72)"; context.fillRect(x - labelWidth - 4, y - 14, labelWidth + 34, height + 23);
  context.fillStyle = gradient; context.fillRect(x, y, 7, height);
  context.strokeStyle = "rgba(255,255,255,.55)"; context.strokeRect(x, y, 7, height);
  context.fillStyle = "#d5e0e5"; context.textAlign = "left";
  context.fillText(raster.label, x - labelWidth, y - 4);
  context.fillText(formatNumber(raster.high), x + 10, y + 4);
  context.fillText(formatNumber(raster.low), x + 10, y + height);
  context.restore();
}

function drawSkyLegend(context, box, layers) {
  context.save(); context.font = "8px system-ui"; context.textAlign = "left";
  let x = box.left + 7, y = box.top + 13;
  if (layers.has("critical")) {
    [["#39d3d6", "critical"], ["#f2c14e", "caustic"]].forEach(([color, label]) => { context.fillStyle = color; context.fillRect(x, y - 6, 12, 2); context.fillStyle = "#d5e0e5"; context.fillText(label, x + 16, y); x += 63; });
  }
  if (layers.has("source")) { context.fillStyle = "#228b22"; context.strokeStyle = "#00ff7f"; context.beginPath(); context.arc(x + 4, y - 3, 3, 0, 2 * Math.PI); context.fill(); context.stroke(); context.fillStyle = "#d5e0e5"; context.fillText("source", x + 11, y); }
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

document.getElementById("refresh-button").addEventListener("click", refresh);
document.getElementById("export-state").addEventListener("click", exportState);
document.getElementById("import-state").addEventListener("click", () => document.getElementById("import-state-file").click());
document.getElementById("import-state-file").addEventListener("change", event => {
  importState(event.target.files[0]);
  event.target.value = "";
});
document.getElementById("refresh-delay").addEventListener("change", event => {
  document.getElementById("timed-delay-wrap").classList.toggle("hidden", event.target.value !== "timed");
  scheduleRefresh();
});
document.getElementById("timed-delay").addEventListener("input", scheduleRefresh);
window.addEventListener("resize", () => { layoutAllControls(); state.plots.forEach(plot => plot.resize()); });

buildControls();
buildModelOptions();
buildSolverOptions();
buildPanelOptions();
renderPanels();
makeSplitter("vertical-splitter", "column");
refresh();
