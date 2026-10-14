"""Canonical image curves and a two-parameter wave-optics surrogate.

The frequency variable is ``w`` and the source position is ``y``. For each
fixed ``y``, write

    F(w,y) = C(w,y) + sum_i p_i(w,y) G_i(w,y),
    p_i = exp(i*w*t_i - i*pi*n_i/2),
    G_i = sqrt(|mu_i|) + W(w,y)*B_i(w,y) + S_i(log10(w),y)/(1+w).

``B_i`` is the first- or second-order stationary-phase correction for an
ordinary image, ``C`` is the separate SIS center contribution, and ``W``
switches the divergent BGO series off at low frequency. Set order=0 to omit
BGO. The splines always tend to zero relative to geometric optics.

To choose one representative of the otherwise nonunique image curves, use
the same spline knots and the same positive quadratic norm at every ``y``:

    R = sum_i integral [eta0*|u_i|^2 + eta1*|u_i'|^2
                        + eta2*|u_i''|^2] d(log10 w),
    u_i = S_i(log10 w)/(1+w).

Minimize R subject to a bound on the weighted RMS error in F. This is the
Sobolev-orthogonal representative of the image-decomposition gauge freedom.
For a fixed number of knots the solution is unique because eta0 > 0. The
single-error-constraint solution is found by bisection on its Lagrange
multiplier, with only Hermitian linear solves. See the report accompanying
the notebook for details and validation.
"""

import os

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.linalg import block_diag, solve

from bgo.expansion import amplification as bgo_amplification, isolated_parts
from decomposition.isolated import isolated_lens


def geometry(model: str, y: float) -> tuple[np.ndarray, np.ndarray]:
    """Return sqrt(|mu_i|) and delays for the isolated two-image lenses."""
    if model == 'PM':
        root = np.sqrt(y*y+4)
        mu_plus = 0.5+(y*y+2)/(2*y*root)
        mu_minus = 0.5-(y*y+2)/(2*y*root)
        delay = y*root/2 + np.log((root+y)/(root-y))
    elif model == 'SIS':
        mu_plus, mu_minus = 1+1/y, 1-1/y
        delay = 2*y
    else:
        raise ValueError("model must be 'PM' or 'SIS'")
    return np.sqrt([mu_plus, abs(mu_minus)]), np.array([0.0, delay])


def image_phases(w: np.ndarray, delays: np.ndarray) -> np.ndarray:
    """Minimum and saddle Morse factors, including their arrival times."""
    return np.exp(1j*np.asarray(w)[None, :]*delays[:, None]) * np.array([1, -1j])[:, None]


def bgo_start(w: np.ndarray, model: str, y: float, wave: np.ndarray,
              tolerance: float = 0.01) -> float:
    """Earliest frequency after which second-order BGO stays within tolerance.

    The ground truth is used only while *building* the surrogate. If no
    frequency passes, set the start to the upper end, so BGO remains off
    throughout the represented frequency band.
    """
    approximation = bgo_amplification(w, model, [y, 0], order=2)
    error = abs(approximation-wave)/np.maximum(abs(wave), 1e-12)
    remaining_maximum = np.maximum.accumulate(error[::-1])[::-1]
    valid = np.flatnonzero(remaining_maximum <= tolerance)
    if not len(valid):
        return float(w[-1])
    index = valid[0]
    if index == 0:
        return float(w[0])
    # Interpolate the crossing to avoid steps at the sampled frequencies.
    previous, current = remaining_maximum[index-1:index+1]
    fraction = (previous-tolerance)/(previous-current) if previous > current else 1.0
    return float(np.exp(np.log(w[index-1])*(1-fraction)+np.log(w[index])*fraction))


def bgo_window(w: np.ndarray, start: float) -> np.ndarray:
    """Smoothstep from zero at ``start`` to one at ``4*start`` in log w."""
    coordinate = np.clip(np.log(np.asarray(w)/start)/np.log(4), 0, 1)
    return coordinate**2*(3-2*coordinate)


@dataclass
class CanonicalBasis:
    """A shared spline space and a positive smoothness matrix."""
    w_min: float
    w_max: float
    knot_count: int = 18
    eta0: float = 0.02
    eta1: float = 1.0
    eta2: float = 0.02

    def __post_init__(self):
        self.knots = np.linspace(np.log10(self.w_min), np.log10(self.w_max),
                                 self.knot_count)
        self.spline = CubicSpline(self.knots, np.eye(self.knot_count), axis=0)
        x = np.linspace(self.knots[0], self.knots[-1], max(300, 20*self.knot_count))
        w = 10**x
        base = self.spline(x)
        first = self.spline(x, 1)
        second = self.spline(x, 2)
        envelope = 1/(1+w)
        envelope_first = -np.log(10)*w/(1+w)**2
        envelope_second = np.log(10)**2*w*(w-1)/(1+w)**3
        b0 = base*envelope[:, None]
        b1 = first*envelope[:, None]+base*envelope_first[:, None]
        b2 = (second*envelope[:, None]+2*first*envelope_first[:, None]
              +base*envelope_second[:, None])
        weight = np.ones(len(x))
        weight[[0, -1]] = 0.5
        weight /= weight.sum()
        self.one_image_norm = sum(
            strength*(basis.T*weight) @ basis
            for strength, basis in ((self.eta0, b0), (self.eta1, b1), (self.eta2, b2))
        )

    def evaluate(self, w: np.ndarray) -> np.ndarray:
        """Matrix taking one image's complex knot values to its correction."""
        return self.spline(np.log10(w))/(1+np.asarray(w)[:, None])


def asymptotic_base(w: np.ndarray, model: str, y: float, order: int,
                    start: float) -> tuple[np.ndarray, np.ndarray]:
    """Return both image factors and the separate SIS center term."""
    magnitudes, _ = geometry(model, y)
    image = np.broadcast_to(magnitudes[:, None], (2, len(w))).astype(complex).copy()
    center = np.zeros(len(w), dtype=complex)
    if order:
        corrections, center_correction = isolated_parts(w, model, y, order)
        window = bgo_window(w, start)
        image += window[None, :]*corrections
        center = window*center_correction
    return image, center


def minimum_smoothness_fit(w: np.ndarray, wave: np.ndarray, phases: np.ndarray,
                           base: np.ndarray, center: np.ndarray, basis: CanonicalBasis,
                           rms_limit: float = 0.01) -> tuple[np.ndarray, float, float]:
    """Canonical curves for any number of images at an RMS complex-error limit.

    All images enter the norm. The data constraint is

        mean(|F_fit-F_wave|^2 / max(|F_wave|, 0.1)^2) <= rms_limit^2.

    ``phases`` and ``base`` have shape (number of images, number of w values).
    Bisection finds the largest ridge multiplier satisfying the error limit;
    that solution minimizes the positive quadratic norm in the feasible set.
    If the spline space cannot reach the limit, return its least-squares fit
    and report the achieved error. The same frequency knots and norm must be
    used at each lens-parameter value to obtain a stable convention.
    """
    target = wave-center-np.sum(phases*base, axis=0)
    b = basis.evaluate(w)
    image_count = len(phases)
    matrix = np.concatenate([phases[i, :, None]*b for i in range(image_count)], axis=1)
    norm = block_diag(*[basis.one_image_norm for _ in range(image_count)])
    scale = np.maximum(abs(wave), 0.1)
    weighted_matrix = matrix/scale[:, None]
    weighted_target = target/scale
    gram = weighted_matrix.conj().T @ weighted_matrix/len(w)
    rhs = weighted_matrix.conj().T @ weighted_target/len(w)

    def solution(multiplier):
        values = solve(gram+multiplier*norm, rhs, assume_a='her')
        error = np.sqrt(np.mean(abs(weighted_matrix @ values-weighted_target)**2))
        return values, float(error)

    # The tiny positive multiplier resolves rank loss if the frequency samples
    # leave a nearly invisible image-mixing direction.
    values, error = solution(1e-12)
    if error > rms_limit:
        return values.reshape(image_count, -1), error, 1e-12
    lower, upper = 1e-12, 1.0
    _, upper_error = solution(upper)
    while upper_error < rms_limit and upper < 1e12:
        lower, upper = upper, upper*10
        _, upper_error = solution(upper)
    for _ in range(38):
        midpoint = np.sqrt(lower*upper)
        candidate, midpoint_error = solution(midpoint)
        if midpoint_error <= rms_limit:
            lower, values, error = midpoint, candidate, midpoint_error
        else:
            upper = midpoint
    return values.reshape(image_count, -1), error, lower


def canonical_fit(w: np.ndarray, wave: np.ndarray, model: str, y: float,
                  basis: CanonicalBasis, order: int, start: float,
                  rms_limit: float = 0.01) -> tuple[np.ndarray, float, float]:
    """Apply the general canonical rule to one isolated PM or SIS lens."""
    _, delays = geometry(model, y)
    phases = image_phases(w, delays)
    base, center = asymptotic_base(w, model, y, order, start)
    return minimum_smoothness_fit(w, wave, phases, base, center, basis, rms_limit)


@dataclass
class FullSurrogate:
    """Interpolated canonical image curves over source position and frequency."""
    model: str
    order: int
    y_train: np.ndarray
    knots: np.ndarray
    values: np.ndarray
    starts: np.ndarray

    def components(self, w: np.ndarray, y: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the two G curves and the separately phased SIS center term."""
        w = np.atleast_1d(np.asarray(w, dtype=float))
        x_y = np.log(self.y_train)
        coefficients = CubicSpline(x_y, self.values, axis=0)(np.log(y))
        start = float(np.exp(PchipInterpolator(x_y, np.log(self.starts))(np.log(y))))
        base, center = asymptotic_base(w, self.model, y, self.order, start)
        correction = CubicSpline(self.knots, coefficients, axis=1)(np.log10(w))
        return base+correction/(1+w)[None, :], center

    def amplification(self, w: np.ndarray, y: float) -> np.ndarray:
        """Evaluate F without calling the exact wave-optics calculation."""
        w = np.atleast_1d(np.asarray(w, dtype=float))
        curves, center = self.components(w, y)
        _, delays = geometry(self.model, y)
        return center+np.sum(image_phases(w, delays)*curves, axis=0)

    def save(self, path: str):
        """Save all numbers needed to evaluate the surrogate later."""
        np.savez_compressed(path, model=self.model, order=self.order,
                            y_train=self.y_train, knots=self.knots,
                            values=self.values, starts=self.starts)

    @classmethod
    def load(cls, path: str):
        """Read a saved surrogate without needing the training waveforms."""
        with np.load(path) as data:
            return cls(str(data['model']), int(data['order']), data['y_train'],
                       data['knots'], data['values'], data['starts'])


def build_surrogate(model: str, order: int, y_train: np.ndarray, w_train: np.ndarray,
                    basis: CanonicalBasis, rms_limit: float = 0.01,
                    waves: np.ndarray | None = None,
                    starts: np.ndarray | None = None
                    ) -> tuple[FullSurrogate, np.ndarray, np.ndarray]:
    """Fit source positions independently under one fixed canonical convention."""
    values, errors, fitted_starts = [], [], []
    if waves is None:
        waves = np.array([isolated_lens(model, w_train, float(y)).wave for y in y_train])
    for index, (y, wave) in enumerate(zip(y_train, waves)):
        start = (float(starts[index]) if starts is not None else
                 bgo_start(w_train, model, float(y), wave)) if order else w_train[-1]
        fitted, error, _ = canonical_fit(w_train, wave, model, float(y),
                                         basis, order, start, rms_limit)
        values.append(fitted)
        errors.append(error)
        fitted_starts.append(start)
    surrogate = FullSurrogate(model, order, np.asarray(y_train), basis.knots,
                              np.asarray(values), np.asarray(fitted_starts))
    return surrogate, np.asarray(errors), np.asarray(fitted_starts)


def error_measures(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Complex, amplitude, and circular-phase errors on a frequency grid."""
    relative = abs(prediction-truth)/np.maximum(abs(truth), 0.1)
    amplitude = abs(abs(prediction)/np.maximum(abs(truth), 0.1)-1)
    phase = abs(np.angle(prediction*np.conj(truth)))
    return {'complex_rms': float(np.sqrt(np.mean(relative**2))),
            'complex_max': float(np.max(relative)),
            'amplitude_rms': float(np.sqrt(np.mean(amplitude**2))),
            'amplitude_max': float(np.max(amplitude)),
            'phase_rms': float(np.sqrt(np.mean(phase**2))),
            'phase_max': float(np.max(phase))}
