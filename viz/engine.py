"""Scientific calculations for the Gravitational Lensing visualizer.

The browser only draws the returned arrays.  All lens calculations live here so
that additional Python solvers can be added without changing the plotting code.

Dimensionless lensing follows the usual thin-lens equations

    y = x - alpha(x),
    tau(x, y) = |x-y|^2 / 2 - psi(x),
    F(w) = w/(2 pi i) integral exp[i w tau(x, y)] d^2x.

The point-mass potential is ``psi = b^2 log(|x|)`` and the singular isothermal
sphere (SIS) potential is ``psi = b |x|``.  A quadratic external field adds
convergence and two shear components.  The BGO implementation follows
Takahashi (2004), Appendix B: https://arxiv.org/abs/astro-ph/0402165
"""

from __future__ import annotations

import os

# Scientific libraries may otherwise start several worker threads.  The
# visualizer is deliberately single-threaded by default, as requested.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
from astropy.constants import G, M_sun, c
from astropy.cosmology import Planck18
from contourpy import contour_generator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bgo.expansion import _image_data, amplification as bgo_amplification


MODEL_NAMES = {"PM": "Point mass", "SIS": "Singular isothermal sphere"}


@dataclass(frozen=True)
class LensSettings:
    """A single lens and source configuration from the interface."""

    model: str
    y: float
    strength: float
    convergence: float
    shear1: float
    shear2: float
    lens_redshift: float
    source_redshift: float
    redshifted_mass: float

    @classmethod
    def from_request(cls, request: dict[str, Any]) -> "LensSettings":
        parameters = request.get("parameters", {})
        model = request.get("model", "PM")
        if model not in MODEL_NAMES:
            model = "PM"
        lens_redshift = float(parameters.get("z_lens", 0.5))
        source_redshift = max(float(parameters.get("z_source", 1.5)), lens_redshift + 0.01)
        return cls(
            model=model,
            y=float(parameters.get("impact", 0.5)),
            strength=float(parameters.get("strength", 1.0)),
            convergence=float(parameters.get("convergence", 0.0)),
            shear1=float(parameters.get("shear1", 0.0)),
            shear2=float(parameters.get("shear2", 0.0)),
            lens_redshift=lens_redshift,
            source_redshift=source_redshift,
            redshifted_mass=float(parameters.get("mass", 100.0)),
        )

    @property
    def source(self) -> np.ndarray:
        """Source position; circular models use a horizontal impact parameter."""
        return np.array([self.y, 0.0])

    @property
    def shear(self) -> tuple[float, float]:
        return self.shear1, self.shear2

    @property
    def lens_time_seconds(self) -> float:
        """Return t_M = 4 G M_Lz / c^3 in seconds."""
        return float(4 * G.value * self.redshifted_mass * M_sun.value / c.value**3)


def _safe_float_array(values: np.ndarray) -> list[float]:
    """Convert an array to finite JSON numbers."""
    return np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12).astype(float).tolist()


def _complex_curve(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=complex)
    return {
        "real": _safe_float_array(values.real),
        "imag": _safe_float_array(values.imag),
        "amplitude": _safe_float_array(np.abs(values)),
        "phase": _safe_float_array(np.unwrap(np.angle(values))),
    }


def _geometric_amplification(w: np.ndarray, images: list[dict[str, Any]]) -> np.ndarray:
    """Stationary-phase geometric-optics sum, referenced to the first image."""
    reference = images[0]["tau"]
    answer = np.zeros_like(w, dtype=complex)
    for image in images:
        answer += (
            np.exp(1j * w * (image["tau"] - reference) - 0.5j * np.pi * image["morse"])
            / np.sqrt(abs(image["determinant"]))
        )
    return answer


def _exact_or_semianalytic(settings: LensSettings, w: np.ndarray) -> np.ndarray:
    """Use GLoW's analytic PM or semianalytic radial SIS calculation.

    Non-zero shear requires GLoW's substantially slower contour calculation.
    The visualizer exposes that separately as the ``glow`` solver; this quick
    radial solver therefore applies the mass-sheet transformation only.
    """
    from glow import freq_domain_c

    scale = max(1.0 - settings.convergence, 1e-5)
    if settings.model == "PM":
        # x = sqrt(m/a) u, with a=1-kappa, gives the unit point-lens
        # problem at w'=m w and y'=y/sqrt(m a).  The integration measure
        # contributes the remaining factor 1/a.
        transformed_y = settings.y / np.sqrt(settings.strength * scale)
        solver = freq_domain_c.Fw_AnalyticPointLens_C(transformed_y, {"parallel": False})
        values = solver(w * settings.strength)
    else:
        transformed_y = settings.y / np.sqrt(scale)
        solver = freq_domain_c.Fw_SemiAnalyticSIS_C(
            transformed_y,
            {"parallel": False},
            psi0=settings.strength / np.sqrt(scale),
        )
        values = solver(w)
    return np.asarray(values, dtype=complex) / scale


def _glow_reference(
    settings: LensSettings,
    w: np.ndarray,
    tolerance: float,
    time_budget: float,
) -> np.ndarray:
    """Evaluate the repository's resolution-checked GLoW wrapper."""
    from bgo.adaptive_reference import adaptive_glow_reference

    values, _, _ = adaptive_glow_reference(
        settings.model,
        settings.source,
        settings.strength,
        settings.shear,
        settings.convergence,
        w,
        relative_l2_limit=tolerance,
        seconds_per_array=time_budget,
    )
    return np.asarray(values, dtype=complex)


def _time_response(w: np.ndarray, values: np.ndarray, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a real positive-frequency inverse transform of ``F(w)-1``.

    The sampled response is

        F_t(t) = Re[ integral (F(w)-1) exp(-i w t) dw / pi ].

    This is intended for solver comparison.  The removed constant corresponds
    to the unlensed delta function at t=0.  A direct trapezoidal transform is
    used because the frequency grid is logarithmic.
    """
    samples = int(np.clip(samples, 64, 320))
    delay_scale = 2 * np.pi / max(float(np.min(w)), 1e-6)
    t = np.linspace(0.0, min(delay_scale, 80.0), samples)
    phase = np.exp(-1j * np.outer(t, w))
    response = np.trapz(phase * (values - 1.0), w, axis=1).real / np.pi
    return t, response


def _image_type(morse: int) -> str:
    return ("minimum", "saddle", "maximum")[int(np.clip(morse, 0, 2))]


def _image_shape(settings: LensSettings, position: np.ndarray) -> dict[str, float]:
    """Return the local image ellipse produced from a small circular source.

    For a source-plane displacement ``dy``, the image displacement is
    ``dx = A^-1 dy``, where ``A = d y / d x`` is the lens-mapping Jacobian.
    Its orthogonal eigenvectors therefore give the ellipse orientation and
    its inverse eigenvalues give the two stretch factors.
    """
    x1, x2 = np.asarray(position, dtype=float)
    radius = max(float(np.hypot(x1, x2)), 1e-8)
    if settings.model == "PM":
        p11 = settings.strength * (radius**2 - 2 * x1**2) / radius**4
        p22 = settings.strength * (radius**2 - 2 * x2**2) / radius**4
        p12 = -2 * settings.strength * x1 * x2 / radius**4
    else:
        p11 = settings.strength * x2**2 / radius**3
        p22 = settings.strength * x1**2 / radius**3
        p12 = -settings.strength * x1 * x2 / radius**3
    jacobian = np.array([
        [1 - p11 - settings.convergence - settings.shear1, -p12 - settings.shear2],
        [-p12 - settings.shear2, 1 - p22 - settings.convergence + settings.shear1],
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(jacobian)
    stretches = 1 / np.maximum(np.abs(eigenvalues), 1e-4)
    major_index = int(np.argmax(stretches))
    vector = eigenvectors[:, major_index]
    return {
        "angle": float(np.arctan2(vector[1], vector[0])),
        "major": float(np.clip(stretches[major_index], 0.35, 7.0)),
        "minor": float(np.clip(stretches[1 - major_index], 0.35, 7.0)),
    }


def _geometry(settings: LensSettings, order: int) -> list[dict[str, Any]]:
    """Return the ordinary images found by the repository BGO lens solver."""
    return _image_data(
        settings.model,
        settings.source,
        settings.strength,
        settings.shear,
        max(0, order),
        settings.convergence,
    )


def _field(settings: LensSettings, resolution: int) -> dict[str, Any]:
    """Sample lens-plane maps, contours, critical curves, and caustics."""
    resolution = int(np.clip(resolution, 48, 180))
    span = max(2.6, 1.8 * (settings.strength + abs(settings.y) + 0.5))
    axis = np.linspace(-span, span, resolution)
    x1, x2 = np.meshgrid(axis, axis)
    radius2 = x1**2 + x2**2
    radius = np.sqrt(radius2)
    radius_safe = np.maximum(radius, 2 * span / resolution)

    if settings.model == "PM":
        lens_potential = settings.strength * np.log(radius_safe)
        radial_factor = settings.strength / radius_safe**2
        # Hessian of psi = b^2 log(r).
        p11 = settings.strength * (radius_safe**2 - 2 * x1**2) / radius_safe**4
        p22 = settings.strength * (radius_safe**2 - 2 * x2**2) / radius_safe**4
        p12 = -2 * settings.strength * x1 * x2 / radius_safe**4
    else:
        lens_potential = settings.strength * radius_safe
        radial_factor = settings.strength / radius_safe
        p11 = settings.strength * x2**2 / radius_safe**3
        p22 = settings.strength * x1**2 / radius_safe**3
        p12 = -settings.strength * x1 * x2 / radius_safe**3

    external_potential = (
        0.5 * settings.convergence * radius2
        + 0.5 * settings.shear1 * (x1**2 - x2**2)
        + settings.shear2 * x1 * x2
    )
    alpha1 = radial_factor * x1 + (settings.convergence + settings.shear1) * x1 + settings.shear2 * x2
    alpha2 = radial_factor * x2 + (settings.convergence - settings.shear1) * x2 + settings.shear2 * x1
    source1 = x1 - alpha1
    source2 = x2 - alpha2

    a11 = 1.0 - p11 - settings.convergence - settings.shear1
    a22 = 1.0 - p22 - settings.convergence + settings.shear1
    a12 = -p12 - settings.shear2
    determinant = a11 * a22 - a12**2
    magnification = np.divide(1.0, determinant, out=np.full_like(determinant, 1e12), where=np.abs(determinant) > 1e-12)

    total_potential = lens_potential + external_potential
    delay = 0.5 * ((x1 - settings.y) ** 2 + x2**2) - total_potential
    residual = np.hypot(settings.y - x1 + alpha1, -x2 + alpha2)

    critical_segments = _contour_segments(axis, determinant, [0.0])
    caustic_segments = []
    for segment in critical_segments:
        points = np.asarray(segment["points"])
        px, py = points[:, 0], points[:, 1]
        pr = np.maximum(np.hypot(px, py), 2 * span / resolution)
        factor = settings.strength / (pr**2 if settings.model == "PM" else pr)
        sa1 = factor * px + (settings.convergence + settings.shear1) * px + settings.shear2 * py
        sa2 = factor * py + (settings.convergence - settings.shear1) * py + settings.shear2 * px
        caustic_segments.append({"level": 0.0, "points": np.column_stack((px - sa1, py - sa2)).tolist()})

    stride = max(3, resolution // 20)
    mapped_cells: list[list[list[float]]] = []
    for row in range(0, resolution - stride, stride):
        for column in range(0, resolution - stride, stride):
            corners = ((row, column), (row, column + stride), (row + stride, column + stride), (row + stride, column))
            if any(determinant[i, j] < 0 for i, j in corners):
                mapped_cells.append([[float(source1[i, j]), float(source2[i, j])] for i, j in corners])

    return {
        "extent": [-span, span, -span, span],
        "resolution": resolution,
        "delay": _safe_float_array(delay.ravel()),
        "log_magnification": _safe_float_array(np.clip(np.log10(np.abs(magnification) + 1e-10), -1, 3).ravel()),
        "negative": (determinant < 0).astype(np.uint8).ravel().tolist(),
        "residual": _safe_float_array(np.clip(np.log10(residual + 1e-5), -5, 1).ravel()),
        "critical": critical_segments,
        "caustic": caustic_segments,
        "mapped_cells": mapped_cells,
    }


def _contour_segments(axis: np.ndarray, values: np.ndarray, levels: list[float]) -> list[dict[str, Any]]:
    """Convert contourpy lines to compact JSON-friendly point lists."""
    generator = contour_generator(x=axis, y=axis, z=values)
    answer: list[dict[str, Any]] = []
    for level in levels:
        for line in generator.lines(level):
            if len(line) >= 2:
                answer.append({"level": float(level), "points": np.asarray(line)[::2].tolist()})
    return answer


def _cosmology(settings: LensSettings) -> dict[str, float]:
    """Planck18 distances and an equivalent point-mass Einstein angle."""
    d_l = Planck18.angular_diameter_distance(settings.lens_redshift).value
    d_s = Planck18.angular_diameter_distance(settings.source_redshift).value
    d_ls = Planck18.angular_diameter_distance_z1z2(settings.lens_redshift, settings.source_redshift).value
    physical_mass = settings.redshifted_mass / (1 + settings.lens_redshift) * M_sun.value
    mpc = 3.085677581491367e22
    theta = np.sqrt(4 * G.value * physical_mass / c.value**2 * (d_ls * mpc) / ((d_l * mpc) * (d_s * mpc)))
    return {
        "d_l_mpc": float(d_l),
        "d_s_mpc": float(d_s),
        "d_ls_mpc": float(d_ls),
        "theta_e_mas": float(np.degrees(theta) * 3.6e6),
        "lens_time_seconds": settings.lens_time_seconds,
    }


def calculate(request: dict[str, Any]) -> dict[str, Any]:
    """Calculate every selected visualization from one interface state."""
    settings = LensSettings.from_request(request)
    solver_settings = request.get("solver_settings", {})
    n_frequency = int(np.clip(solver_settings.get("frequency_samples", 140), 48, 360))
    sky_resolution = int(np.clip(solver_settings.get("sky_resolution", 90), 48, 180))
    bgo_order = int(np.clip(solver_settings.get("bgo_order", 2), 0, 10))
    time_samples = int(np.clip(solver_settings.get("time_samples", 140), 64, 320))
    tolerance = float(solver_settings.get("glow_tolerance", 0.02))
    time_budget = float(solver_settings.get("glow_budget", 2.0))

    parameters = request.get("parameters", {})
    f_min = max(float(parameters.get("f_min", 10.0)), 1e-3)
    f_max = max(float(parameters.get("f_max", 1000.0)), f_min * 1.01)
    frequency = np.geomspace(f_min, f_max, n_frequency)
    w = 2 * np.pi * frequency * settings.lens_time_seconds
    # BGO is asymptotic and becomes singular at zero.  A small positive floor
    # keeps every solver finite when a very low physical frequency is selected.
    w_safe = np.maximum(w, 1e-4)

    images = _geometry(settings, bgo_order)
    if not images:
        raise ValueError("No geometric-optics images were found for these parameters")

    selected_solvers = request.get("solvers") or ["exact", "bgo"]
    curves: dict[str, dict[str, Any]] = {}
    messages: list[str] = []
    for solver_name in selected_solvers:
        try:
            if solver_name == "geometric":
                values = _geometric_amplification(w_safe, images)
                label = "Geometric optics"
            elif solver_name == "bgo":
                values = bgo_amplification(
                    w_safe,
                    settings.model,
                    settings.source,
                    strength=settings.strength,
                    shear=settings.shear,
                    order=bgo_order,
                    convergence=settings.convergence,
                )
                label = f"BGO order {bgo_order}"
            elif solver_name == "glow":
                values = _glow_reference(settings, w_safe, tolerance, time_budget)
                label = "GLoW reference"
            else:
                values = _exact_or_semianalytic(settings, w_safe)
                label = "Analytic" if settings.model == "PM" else "Semianalytic SIS"
                if abs(settings.shear1) + abs(settings.shear2) > 0:
                    messages.append(f"{label} is radial and does not include the selected shear; choose GLoW or BGO for shear.")
            curve = _complex_curve(values)
            t, response = _time_response(w_safe, values, time_samples)
            curve["time"] = _safe_float_array(t)
            curve["time_response"] = _safe_float_array(response)
            curve["label"] = label
            curves[solver_name] = curve
        except Exception as error:
            # One expensive solver should not prevent the remaining panels from
            # updating.  The compact status line reports the scientific reason.
            messages.append(f"{solver_name}: {error}")

    first_delay = images[0]["tau"]
    image_rows = []
    for index, image in enumerate(images):
        hessian_magnification = 1.0 / image["determinant"]
        image_rows.append({
            "order": index + 1,
            "x": np.asarray(image["x"], dtype=float).tolist(),
            "magnification": float(hessian_magnification),
            "log_magnification": float(np.clip(np.log10(abs(hessian_magnification) + 1e-12), -1, 3)),
            "type": _image_type(image["morse"]),
            "delay_dimensionless": float(image["tau"] - first_delay),
            "delay_seconds": float((image["tau"] - first_delay) * settings.lens_time_seconds),
            "shape": _image_shape(settings, image["x"]),
        })

    field = _field(settings, sky_resolution)
    maximum_delay = max((row["delay_dimensionless"] for row in image_rows), default=1.0)
    contour_maximum = max(2 * maximum_delay, 0.25)
    axis = np.linspace(field["extent"][0], field["extent"][1], field["resolution"])
    delay_grid = np.asarray(field["delay"]).reshape(field["resolution"], field["resolution"])
    relative_delay_grid = delay_grid - first_delay
    field["delay"] = _safe_float_array(relative_delay_grid.ravel())
    field["delay_contours"] = _contour_segments(axis, relative_delay_grid, np.linspace(0, contour_maximum, 7).tolist())

    return {
        "model": settings.model,
        "model_name": MODEL_NAMES[settings.model],
        "frequency_hz": _safe_float_array(frequency),
        "w": _safe_float_array(w),
        "curves": curves,
        "images": image_rows,
        "source": settings.source.tolist(),
        "field": field,
        "cosmology": _cosmology(settings),
        "messages": messages,
    }
