"""Adaptive, resolution-checked GLoW reference amplifications.

The numerical reference is accepted only when two consecutive contour
resolutions agree in relative discrete L2 norm.  The timing limit applies to
one evaluation of the *complete* requested frequency array, rather than to
each frequency or to the sum of the convergence checks.
"""

import os

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import time

import numpy as np
from glow import freq_domain_c, lenses, time_domain_c

def _source_and_shear(source: float | np.ndarray,
                      shear: float | tuple[float, float]
                      ) -> tuple[float, float, float]:
    """Rotate the source onto GLoW's positive x axis.

    GLoW's time-domain classes take a scalar source radius.  A shear is a
    spin-two quantity, so a coordinate rotation by ``phi`` rotates
    ``gamma1+i gamma2`` by ``exp(-2 i phi)``.
    """
    source = np.asarray(source, dtype=float)
    if source.ndim == 0:
        y1, y2 = float(source), 0.0
    elif source.shape == (2,):
        y1, y2 = map(float, source)
    else:
        raise ValueError('source must be a scalar radius or a two-component position')
    if np.ndim(shear) == 0:
        gamma1, gamma2 = float(shear), 0.0
    else:
        gamma1, gamma2 = map(float, shear)
    phi = np.arctan2(y2, y1)
    cosine, sine = np.cos(2*phi), np.sin(2*phi)
    rotated_gamma1 = gamma1*cosine+gamma2*sine
    rotated_gamma2 = -gamma1*sine+gamma2*cosine
    return float(np.hypot(y1, y2)), rotated_gamma1, rotated_gamma2


def _lens(model: str, strength: float, convergence: float,
          gamma1: float, gamma2: float):
    """Construct the circular lens plus convergence and shear directly."""
    base = (lenses.Psi_PointLens if model == 'PM' else lenses.Psi_SIS)(
        {'psi0': strength})
    if convergence == gamma1 == gamma2 == 0:
        return base
    field = lenses.Psi_Ext({'kappa': convergence, 'gamma1': gamma1, 'gamma2': gamma2})
    return lenses.CombinedLens({'lenses': [base, field]})


def _analytic_reference(model: str, y: float, strength: float,
                        convergence: float, w: np.ndarray) -> np.ndarray:
    """Exact PM or semianalytic radial SIS result under mass-sheet scaling."""
    scale = 1-convergence
    transformed_y = y/np.sqrt(scale)
    if model == 'PM':
        if strength != 1:
            raise ValueError('The analytic PM reference expects unit strength')
        solver = freq_domain_c.Fw_AnalyticPointLens_C(
            transformed_y, {'parallel': False})
    else:
        solver = freq_domain_c.Fw_SemiAnalyticSIS_C(
            transformed_y, {'parallel': False}, psi0=strength/np.sqrt(scale))
    return solver(w)/scale


def _regularized_transform(response, w: np.ndarray) -> np.ndarray:
    """Apply GLoW's image-regularized FFT to a sampled time response."""
    transform = freq_domain_c.Fw_FFT_C(
        response, {'wmin': float(min(w)), 'wmax': float(max(w)),
                   'eval_mode': 'exact', 'parallel': False,
                   'N_above_discard': 7})
    return transform(w)


def adaptive_glow_reference(model: str, source: float | np.ndarray, strength: float,
                            shear: float | tuple[float, float], convergence: float,
                            w: np.ndarray, relative_l2_limit: float = 0.01,
                            seconds_per_array: float = 1.0
                            ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return a GLoW curve with a measured resolution uncertainty.

    The uncertainty is ``||F_high-F_low||_2/||F_high||_2`` for consecutive
    resolution levels. Numerical calculations continue until this is at
    most ``relative_l2_limit``. The accepted high-resolution evaluation may
    not exceed ``seconds_per_array`` for the complete frequency array.
    Failure to satisfy both requirements raises ``RuntimeError`` rather than
    returning an under-resolved curve.
    """
    w = np.asarray(w, dtype=float)
    start = time.perf_counter()
    budget = seconds_per_array
    y, gamma1, gamma2 = _source_and_shear(source, shear)
    if gamma1 == gamma2 == 0:
        values = _analytic_reference(model, y, strength, convergence, w)
        elapsed = time.perf_counter()-start
        return values, np.zeros_like(w), {
            'method': 'analytic PM' if model == 'PM' else 'semianalytic radial SIS',
            'relative_l2_uncertainty': 0.0, 'elapsed_seconds': elapsed,
            'array_evaluation_seconds': elapsed, 'levels': []}

    lens = _lens(model, strength, convergence, gamma1, gamma2)
    levels = [{'Nt': value, 'tmax': 1000, 'parallel': False}
              for value in (3000, 4500, 6500, 9000, 12000)]

    previous = None
    records = []
    for settings in levels:
        level_start = time.perf_counter()
        response = time_domain_c.It_MultiContour_C(lens, y, settings)
        values = _regularized_transform(response, w)
        elapsed = time.perf_counter()-start
        level_seconds = time.perf_counter()-level_start
        record = {'settings': settings, 'level_seconds': level_seconds,
                  'cumulative_seconds': elapsed}
        if not np.all(np.isfinite(values)):
            record['finite'] = False
            records.append(record)
            break
        if previous is not None:
            spread = abs(values-previous)
            relative_l2 = float(np.linalg.norm(spread)/np.linalg.norm(values))
            record['relative_l2_uncertainty'] = relative_l2
            records.append(record)
            if relative_l2 <= relative_l2_limit and level_seconds <= budget:
                return values, spread, {
                    'method': 'multi-contour and regularized FFT',
                    'relative_l2_uncertainty': relative_l2,
                    'elapsed_seconds': elapsed,
                    'array_evaluation_seconds': level_seconds, 'levels': records}
        else:
            records.append(record)
        if level_seconds > budget:
            break
        previous = values

    attained = records[-1].get('relative_l2_uncertainty', np.inf)
    elapsed = time.perf_counter()-start
    raise RuntimeError(
        f'GLoW resolution target {relative_l2_limit:.3g} was not reached within '
        f'{seconds_per_array:.3g} s for the complete frequency array; '
        f'attained {attained:.3g} after {elapsed:.2f} s of resolution checks')

