"""Smooth image contributions to isolated point-mass and SIS wave amplification.

For two images, with the minimum arrival time subtracted, the model is

    F(w) = G_min(w) + G_sad(w) exp[i w Delta T - i pi/2].

The factors G tend to the *positive* square roots of the absolute image
magnifications. The saddle's Morse phase is written outside G. See
Takahashi & Nakamura (2003), https://arxiv.org/abs/astro-ph/0305055,
for the point-mass wave formula and geometric limit. The SIS radial
integral follows Eq. 65 of GLoW, https://arxiv.org/abs/2409.04606.
"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

from dataclasses import dataclass

import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.special import wofz

from bgo.expansion import amplification as bgo_amplification


@dataclass
class LensData:
    name: str
    y: float
    delay: float
    magnitudes: np.ndarray
    wave: np.ndarray
    geometric: np.ndarray


def isolated_lens(name: str, w: np.ndarray, y: float = 0.5) -> LensData:
    """Return wave and geometric amplification for an isolated PM or SIS.

    The minimum has zero delay. The saddle has delay ``delay`` and Morse
    phase ``-pi/2``. SIS here means a circular singular isothermal sphere
    with Einstein radius one, and requires ``0 < y < 1`` for two images.
    """
    w = np.asarray(w)
    if name == "PM":
        root = np.sqrt(y*y + 4)
        x_plus, x_minus = (y + root)/2, (y - root)/2
        phi_plus = (x_plus-y)**2/2 - np.log(abs(x_plus))
        phi_minus = (x_minus-y)**2/2 - np.log(abs(x_minus))
        delay = phi_minus - phi_plus
        mu_plus = 0.5 + (y*y+2)/(2*y*root)
        mu_minus = 0.5 - (y*y+2)/(2*y*root)
        wave = []
        with mp.workdps(35):
            for frequency in w:
                u = mp.mpf(str(frequency))/2
                phase = mp.exp(mp.pi*u/2 + 1j*u*(mp.log(u)-2*phi_plus))
                wave.append(complex(phase * mp.gamma(1-1j*u)
                                    * mp.hyp1f1(1j*u, 1, 1j*u*y*y)))
        wave = np.asarray(wave)
    elif name == "SIS":
        if not 0 < y < 1:
            raise ValueError("This two-image SIS example requires 0 < y < 1.")
        delay = 2*y
        mu_plus, mu_minus = 1+1/y, 1-1/y
        phi_plus = -y-0.5
        nodes, weights = leggauss(512)
        theta = np.pi*(nodes+1)
        b = 1+y*np.cos(theta)
        wave = []
        for frequency in w:
            z = np.exp(5j*np.pi/4)*b*np.sqrt(frequency/2)
            j = np.sqrt(np.pi/(2*frequency))*np.exp(1j*np.pi/4)*wofz(z)
            radial = b*j + 1j/frequency
            unshifted = frequency/(2j)*np.dot(weights, radial)
            wave.append(unshifted*np.exp(1j*frequency*(y*y/2-phi_plus)))
        wave = np.asarray(wave)
    else:
        raise ValueError("name must be 'PM' or 'SIS'")

    magnitudes = np.sqrt([mu_plus, abs(mu_minus)])
    geometric = magnitudes[0] - 1j*magnitudes[1]*np.exp(1j*w*delay)
    return LensData(name, y, delay, magnitudes, wave, geometric)


@dataclass
class Decomposition:
    """Fitted cubic splines for the two image contributions."""
    knots: np.ndarray
    values: np.ndarray
    magnitudes: np.ndarray
    delay: float
    loss: float
    fit_error: float
    length: float
    acceleration: float
    score: float
    gradient_norm: float

    def components(self, w: np.ndarray) -> np.ndarray:
        """Return G_min and G_sad at positive dimensionless frequencies."""
        return self.magnitudes[:, None] + CubicSpline(self.knots, self.values, axis=1)(
            np.log10(w))/(1+np.asarray(w))[None, :]

    def amplification(self, w: np.ndarray) -> np.ndarray:
        g = self.components(w)
        return g[0] - 1j*g[1]*np.exp(1j*w*self.delay)


def bgo_onset(w: np.ndarray, data: LensData, order: int, tolerance: float = 0.01) -> float:
    """First frequency after which the cumulative BGO error stays below ``tolerance``.

    The comparison uses the *total* PM or SIS BGO amplification. For SIS this
    includes the singular-center series, which is needed at first order.
    """
    approximation = bgo_amplification(w, data.name, [data.y, 0], order=order)
    relative_error = abs(approximation-data.wave)/abs(data.wave)
    remaining_maximum = np.maximum.accumulate(relative_error[::-1])[::-1]
    valid = np.flatnonzero(remaining_maximum <= tolerance)
    return float(w[valid[0]]) if len(valid) else float('nan')


def smooth_bgo_window(w: np.ndarray, start: float, full: float) -> np.ndarray:
    """Cubic smoothstep in log frequency: zero below start, one above full."""
    fraction = np.clip(np.log(w/start)/np.log(full/start), 0, 1)
    return fraction**2*(3-2*fraction)


def fit_components(w: np.ndarray, data: LensData, knot_counts=range(4, 13),
                   length_weight: float = 0.0001, acceleration_weight: float = 0,
                   knot_weight: float = 0.000003, bgo_order: int = 0,
                   bgo_full: float | None = None, bgo_weight: float = 0.01,
                   bgo_max: float | None = None
                   ) -> tuple[Decomposition, list[Decomposition]]:
    """Fit both G curves and choose their common number of spline knots.

    Let x=log10(w), and let S_i(x) be a cubic spline through ``k``
    complex values. Set G_i(w)=sqrt(|mu_i|)+S_i(x)/(1+w), which guarantees
    the geometric limit because a cubic extrapolation grows only as
    ``log(w)^3``, slower than ``1+w``. For every candidate k, minimize

        mean |F_data - G_1 - G_2 exp(i w Delta T-i pi/2)|^2
        + lambda_L sum_i integral sqrt(epsilon^2+|dG_i/dx|^2) dx / range(x)
        + lambda_A sum_i integral sqrt(epsilon^2+|d²G_i/dx²|^2) dx / range(x).

    Thus the lengths and, when requested, accelerations of *both curves*
    enter the cost. The small epsilon makes these terms differentiable.
    When ``bgo_order`` is 1 or 2, a soft penalty also draws reconstructed
    F toward BGO above ``bgo_full``. The BGO
    correction is smoothly zeroed below ``bgo_full/2``; the spline itself
    always describes deviations from geometric optics. Candidate knot
    counts are compared with ``knot_weight * 2*k``; 2*k is
    the number of complex knot values. Knot locations are equally spaced
    in log frequency; only their number and their complex values vary.
    """
    w = np.asarray(w)
    x = np.log10(w)
    phase = -1j*np.exp(1j*w*data.delay)
    candidates = []
    if bgo_order:
        if bgo_full is None:
            raise ValueError('Set bgo_full from a checked BGO accuracy range.')
        if bgo_max is None:
            bgo_max = float(w[-1])
        bgo_w = np.geomspace(bgo_full/2, bgo_max, 180)
        bgo_x = np.log10(bgo_w)
        bgo_window = smooth_bgo_window(bgo_w, bgo_full/2, bgo_full)
        bgo_target = bgo_amplification(bgo_w, data.name, [data.y, 0],
                                       order=bgo_order)
        high_phase = -1j*np.exp(1j*bgo_w*data.delay)
    for count in knot_counts:
        knots = np.linspace(x[0], x[-1], count)
        basis = CubicSpline(knots, np.eye(count), axis=0)
        b = basis(x)/(1+w[:, None])
        # A denser quadrature grid makes the length independent of the data spacing.
        qx = np.linspace(x[0], x[-1], max(200, 30*count))
        qw = 10**qx
        db = basis(qx, 1)/(1+qw[:, None]) - (
            np.log(10)*qw/(1+qw)**2)[:, None]*basis(qx)
        second_envelope = np.log(10)**2*qw*(qw-1)/(1+qw)**3
        ddb = (basis(qx, 2)/(1+qw[:, None])
               - 2*(np.log(10)*qw/(1+qw)**2)[:, None]*basis(qx, 1)
               + second_envelope[:, None]*basis(qx))
        qweight = np.ones(len(qx))
        qweight[[0, -1]] = 0.5
        qweight /= qweight.sum()
        eps = 1e-5
        if bgo_order:
            bbgo = basis(bgo_x)/(1+bgo_w[:, None])

        def unpack(parameters):
            return parameters.reshape(2, 2, count)[:, 0] + 1j*parameters.reshape(2, 2, count)[:, 1]

        def objective(parameters):
            values = unpack(parameters)
            curves = data.magnitudes[:, None] + values @ b.T
            residual = curves[0] + phase*curves[1] - data.wave
            fit = np.mean(abs(residual)**2)

            slopes = values @ db.T
            speed = np.sqrt(eps**2 + abs(slopes)**2)
            length = np.sum(speed @ qweight)
            accelerations = values @ ddb.T
            acceleration_speed = np.sqrt(eps**2 + abs(accelerations)**2)
            acceleration = np.sum(acceleration_speed @ qweight)
            cost = fit + length_weight*length + acceleration_weight*acceleration
            if bgo_order:
                high_curves = data.magnitudes[:, None] + values @ bbgo.T
                bgo_difference = high_curves[0]+high_phase*high_curves[1]-bgo_target
                cost += bgo_weight*np.mean(bgo_window*abs(bgo_difference)**2)

            # Exact derivative of the data term, the smoothed curve lengths,
            # with both image contributions varied together.
            grad = np.empty_like(values)
            for i, ph in enumerate((np.ones_like(phase), phase)):
                grad[i] = 2*np.mean((np.conj(ph)*residual)[:, None]*b, axis=0)
                grad[i] += length_weight * ((qweight*slopes[i]/speed[i]) @ db)
                grad[i] += acceleration_weight * (
                    (qweight*accelerations[i]/acceleration_speed[i]) @ ddb)
                if bgo_order:
                    high_factor = np.ones_like(high_phase) if i == 0 else high_phase
                    difference = np.conj(high_factor)*bgo_difference
                    grad[i] += 2*bgo_weight*np.mean(
                        (bgo_window*difference)[:, None]*bbgo, axis=0)
            gradient = np.stack((grad.real, grad.imag), axis=1).ravel()
            return cost, gradient

        start = np.zeros((2, 2, count))
        result = minimize(objective, start.ravel(), method="L-BFGS-B", jac=True,
                          options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-9})
        values = unpack(result.x)
        curves = data.magnitudes[:, None] + values @ b.T
        fit_error = np.sqrt(np.mean(abs(curves[0]+phase*curves[1]-data.wave)**2))
        length = np.sum(np.sqrt(eps**2+abs(values @ db.T)**2) @ qweight)
        acceleration = np.sum(np.sqrt(eps**2+abs(values @ ddb.T)**2) @ qweight)
        score = result.fun + knot_weight*2*count
        candidates.append(Decomposition(knots, values, data.magnitudes, data.delay, result.fun,
                                        fit_error, length, acceleration, score,
                                        float(np.max(abs(result.jac)))))
    return min(candidates, key=lambda item: item.score), candidates
