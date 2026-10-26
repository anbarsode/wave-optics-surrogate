"""Adaptive, lens-independent decomposition into smooth image factors.

The supplied geometric-optics data define

    F_geo(w) = sum_i sqrt(|mu_i|) exp[i (w*t_i + phi_i)].

``phi_i`` is the Morse *phase shift in radians*, for example ``-pi/2``
for a saddle. The fitted factors are

    G_i(w) = sqrt(|mu_i|) + W(w) sqrt(|mu_i|)
             * sum_(k=1)^K B_ik (i/w)^k + S_i(log10(w))/(1+w).

The BGO coefficients and their smooth turn-on are optional. User-requested
ghosts have a fitted delay and phase and a baseline ``sqrt(mu_g)/(1+w)``;
they therefore disappear in the geometric limit. Each ``S_i`` is a complex
cubic spline, and the same frequency knots are used for images and ghosts.
The positive amplitude and curvature penalties select one reproducible
member of the otherwise non-unique image split. See ``DECOMPOSER.md`` for
the objective, knot-selection rule, and the limits of the greedy search.
"""

import os

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import maximum_filter1d
from scipy.optimize import minimize
from scipy.signal import find_peaks


def _phases(w: np.ndarray, delays: np.ndarray, morse_shifts: np.ndarray) -> np.ndarray:
    """Geometric propagation and Morse factors for any number of images."""
    return np.exp(1j*(delays[:, None]*w[None, :]+morse_shifts[:, None]))


def _bgo_window(w: np.ndarray, start: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Smooth BGO turn-on and its derivative with respect to log10(w)."""
    if start is None:
        return np.zeros_like(w), np.zeros_like(w)
    coordinate = np.log(w/start)/np.log(4)
    clipped = np.clip(coordinate, 0, 1)
    window = clipped**2*(3-2*clipped)
    derivative = 6*clipped*(1-clipped)*np.log(10)/np.log(4)
    derivative[(coordinate <= 0) | (coordinate >= 1)] = 0
    return window, derivative


def _base(w: np.ndarray, magnitudes: np.ndarray,
          coefficients: np.ndarray | None, start: float | None,
          ghost_mask: np.ndarray | None = None,
          derivative: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Geometric/BGO image factors and vanishing ghost factors.

    A ghost magnitude is the squared coefficient of ``1/(1+w)``.  This
    convention lets a ghost describe a diffracted contribution without
    adding a new geometric-optics image at infinite frequency.
    """
    amplitude = np.sqrt(abs(magnitudes))[:, None]
    base = np.broadcast_to(amplitude, (len(magnitudes), len(w))).astype(complex).copy()
    slope = np.zeros_like(base)
    if ghost_mask is not None and np.any(ghost_mask):
        base[ghost_mask] /= 1+w
        slope[ghost_mask] = (-np.log(10)*w/(1+w)**2)[None, :]*amplitude[ghost_mask]
    if coefficients is not None:
        window, window_slope = _bgo_window(w, start)
        correction = np.zeros_like(base)
        correction_slope = np.zeros_like(base)
        for order, column in enumerate(coefficients.T, start=1):
            term = amplitude*column[:, None]*(1j/w[None, :])**order
            correction += term
            correction_slope -= order*np.log(10)*term
        base += window[None, :]*correction
        slope += window_slope[None, :]*correction+window[None, :]*correction_slope
    return (base, slope) if derivative else base


def _bgo_start(w: np.ndarray, wave: np.ndarray, delays: np.ndarray,
               magnitudes: np.ndarray, morse_shifts: np.ndarray,
               coefficients: np.ndarray, known_term: np.ndarray, tolerance: float,
               image_fraction: float) -> float:
    """Require accurate total F and small BGO terms for every image.

    The full-wave test alone can accept a large error in a faint image.
    Both conditions are checked on the supplied frequency array. The BGO
    turn-on starts at the first frequency satisfying both conditions for
    every later sample. If none passes, BGO stays off in the fitted band.
    """
    full = known_term+np.sum(_phases(w, delays, morse_shifts)
                             *_base(w, magnitudes, coefficients, start=w[0]/4), axis=0)
    total_error = abs(full-wave)/np.maximum(abs(wave), 1e-12)
    later_error = np.maximum.accumulate(total_error[::-1])[::-1]
    orders = np.arange(1, coefficients.shape[1]+1)
    relative_terms = np.sum(abs(coefficients)[:, :, None]
                            /w[None, None, :]**orders[None, :, None], axis=1)
    largest_image_term = np.max(relative_terms, axis=0)
    valid = np.flatnonzero((later_error <= tolerance)
                           & (largest_image_term <= image_fraction))
    return float(w[valid[0]]) if len(valid) else float(w[-1])


@dataclass
class ImageDecomposition:
    """Cubic-spline image factors and the data needed to reconstruct F."""

    knots: np.ndarray                         # Positions in log10(w).
    values: np.ndarray                        # Complex spline values: image x knot.
    delays: np.ndarray
    magnifications: np.ndarray                # Signed geometric magnifications.
    morse_shifts: np.ndarray                   # Phase shifts in radians.
    bgo_coefficients: np.ndarray | None       # B_ik, with columns k=1,2,...
    bgo_start: float | None
    relative_l2: float
    length: float
    peak_counts: tuple[int, int]              # Peaks of |F| and |F_geo|.
    history: tuple[tuple[int, float], ...]    # (knot count, relative L2).
    known_w: np.ndarray | None = None          # Samples for a separate known term.
    known_term: np.ndarray | None = None       # For example, an SIS center contribution.
    physical_image_count: int | None = None
    ghost_mask: np.ndarray | None = None
    smoothness_cost: float = np.inf             # Full fixed-knot objective at the answer.
    ghost_prefit_relative_l2: float | None = None
    ghost_delay_resolution: float | None = None
    ghost_spectral_support: np.ndarray | None = None
    ghost_residual_spectral_strength: np.ndarray | None = None
    ghost_spectrum_delays: np.ndarray | None = None
    ghost_residual_spectrum: np.ndarray | None = None
    ghost_component_spectra: np.ndarray | None = None

    def components(self, w: np.ndarray) -> np.ndarray:
        """Return G_i(w), with shape (number of images, number of frequencies)."""
        w = np.atleast_1d(np.asarray(w, dtype=float))
        correction = CubicSpline(self.knots, self.values, axis=1)(np.log10(w))
        return (_base(w, self.magnifications, self.bgo_coefficients, self.bgo_start,
                      self.ghost_mask)
                +correction/(1+w)[None, :])

    @property
    def ghost_count(self) -> int:
        """Number of fitted non-geometric terms."""
        return 0 if self.ghost_mask is None else int(np.count_nonzero(self.ghost_mask))

    @property
    def ghost_delays(self) -> np.ndarray:
        """Fitted ghost delays."""
        return self.delays[-self.ghost_count:] if self.ghost_count else np.array([])

    @property
    def ghost_phases(self) -> np.ndarray:
        """Fitted constant ghost phase shifts in radians."""
        return self.morse_shifts[-self.ghost_count:] if self.ghost_count else np.array([])

    @property
    def ghost_magnifications(self) -> np.ndarray:
        """Squared amplitudes of the fitted ``1/(1+w)`` ghost baselines."""
        return self.magnifications[-self.ghost_count:] if self.ghost_count else np.array([])

    def amplification(self, w: np.ndarray) -> np.ndarray:
        """Recompose F from the fitted image factors."""
        w = np.atleast_1d(np.asarray(w, dtype=float))
        answer = np.sum(_phases(w, self.delays, self.morse_shifts)*self.components(w), axis=0)
        if self.known_term is not None:
            answer += CubicSpline(np.log10(self.known_w), self.known_term)(np.log10(w))
        return answer


def _basis(knots: np.ndarray, w: np.ndarray, qx: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spline values, quadrature values, speeds, and accelerations."""
    spline = CubicSpline(knots, np.eye(len(knots)), axis=0)
    b = spline(np.log10(w))/(1+w[:, None])
    qw = 10**qx
    envelope = 1/(1+qw)
    envelope_first = -np.log(10)*qw/(1+qw)**2
    envelope_second = np.log(10)**2*qw*(qw-1)/(1+qw)**3
    base = spline(qx)
    first = spline(qx, 1)
    u = base*envelope[:, None]
    du = first*envelope[:, None]+base*envelope_first[:, None]
    ddu = (spline(qx, 2)*envelope[:, None]+2*first*envelope_first[:, None]
           +base*envelope_second[:, None])
    return b, u, du, ddu


def _shared_ghost_delays(w: np.ndarray, wave: np.ndarray,
                         prefit: ImageDecomposition,
                         physical_delays: np.ndarray,
                         physical_magnifications: np.ndarray,
                         physical_morse_shifts: np.ndarray,
                         bgo_coefficients: np.ndarray | None,
                         bgo_start: float | None,
                         count: int,
                         delay_bounds: tuple[float, float] | None,
                         search_points: int,
                         minimum_residual_strength: float,
                         minimum_shared_support: float,
                         ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray, np.ndarray]:
    """Find ghost delays from the prefit residual and diagnose shared peaks.

    A missing coherent term with delay ``t_g`` is distributed among the
    physical image factors by a fit that has no ghost.  In the contribution
    from image ``i`` it appears as

        exp(i*w*t_i) delta G_i(w) ~ c_i exp(i*w*t_g).

    The Fourier transforms of the *contributions*, rather than of ``G_i``
    alone, therefore peak at the same absolute delay ``t_g``.  A non-unique
    decomposition can nevertheless place most of the missing term in only
    one or two factors.  A shared peak is therefore useful confirmation, but
    it is not a necessary condition.

    Ghost delays are selected from the Fourier transform of the residual left
    by the cheap no-ghost prefit: this is precisely the part that the ordinary
    image factors did not capture.  The individual image spectra are retained
    as a separate diagnostic.  All transforms use a uniform frequency grid,
    remove a straight-line trend, apply a Hann window, and divide out the
    ghost's known ``1/(1+w)`` envelope.  Delays within one Fourier bin of a
    supplied physical image are excluded.
    """
    image_count = len(physical_delays)
    base = _base(w, physical_magnifications, bgo_coefficients, bgo_start)
    phase = _phases(w, physical_delays, physical_morse_shifts)
    # The fitted spline represents S_i=(1+w)(G_i-G_i^base).  Removing the
    # known 1/(1+w) envelope before the transform turns a ghost into a narrow
    # Fourier peak instead of a broad low-delay feature.
    contributions = phase*(prefit.components(w)[:image_count]-base)*(1+w)[None, :]

    # FFT frequencies are meaningful only on a uniform w grid.  A moderate
    # interpolation grid is sufficient; zero padding then gives a smooth
    # numerical representation of the continuous Fourier transform.
    uniform_count = max(512, 4*len(w))
    uniform_w = np.linspace(w[0], w[-1], uniform_count)
    values = CubicSpline(w, contributions, axis=1)(uniform_w)
    coordinate = np.linspace(0, 1, uniform_count)
    trend = values[:, :1]*(1-coordinate)+values[:, -1:] * coordinate
    values = values-trend
    window = np.hanning(uniform_count)
    n_fft = max(4096, int(search_points),
                2**int(np.ceil(np.log2(8*uniform_count))))
    spectrum = np.fft.fft(values*window[None, :], n=n_fft, axis=1)
    delay_axis = 2*np.pi*np.fft.fftfreq(n_fft, uniform_w[1]-uniform_w[0])
    amplitudes = abs(spectrum)
    order = np.argsort(delay_axis)
    delay_axis = delay_axis[order]
    amplitudes = amplitudes[:, order]
    delay_resolution = 2*np.pi/(w[-1]-w[0])
    amplitudes /= np.maximum(np.max(amplitudes, axis=1, keepdims=True), 1e-30)
    # Peaks need only agree to the actual delay resolution.  Within that
    # neighbourhood, score a candidate by the weakest image.  A high score
    # therefore means that every physical-image correction contains the
    # oscillation; a large peak in only one or two images cannot dominate.
    delay_step = float(np.median(np.diff(delay_axis)))
    neighbourhood = max(3, 2*int(np.ceil(0.5*delay_resolution/delay_step))+1)
    nearby_amplitudes = maximum_filter1d(amplitudes, size=neighbourhood,
                                         axis=1, mode='nearest')
    residual_values = (wave-prefit.amplification(w))*(1+w)
    uniform_residual = CubicSpline(w, residual_values)(uniform_w)
    residual_trend = (uniform_residual[0]*(1-coordinate)
                      +uniform_residual[-1]*coordinate)
    residual_spectrum = abs(np.fft.fft(
        (uniform_residual-residual_trend)*window, n=n_fft))[order]
    residual_spectrum /= max(float(np.max(residual_spectrum)), 1e-30)

    if delay_bounds is None:
        lower, upper = float(delay_axis.min()), float(delay_axis.max())
    else:
        lower, upper = map(float, delay_bounds)
        if not lower < upper:
            raise ValueError('ghost_delay_bounds must be an increasing pair')
    allowed = (delay_axis >= lower) & (delay_axis <= upper)
    for delay in physical_delays:
        allowed &= abs(delay_axis-delay) > delay_resolution
    local_maximum = np.zeros_like(allowed)
    local_maximum[find_peaks(residual_spectrum)[0]] = True
    allowed &= local_maximum

    # Pick separated maxima of the residual spectrum.  The exclusion width is
    # the main-lobe scale of the Hann-windowed transform, which prevents a
    # second ghost from being initialized on a side lobe of the first.
    guesses = []
    support = []
    residual_strength = []
    candidate_strength = residual_spectrum.copy()
    candidate_strength[~allowed] = -np.inf
    separation = 2*delay_resolution
    while len(guesses) < count:
        index = int(np.argmax(candidate_strength))
        if not np.isfinite(candidate_strength[index]):
            break
        shared_support = nearby_amplitudes[:, index].copy()
        strong_residual = residual_spectrum[index] >= minimum_residual_strength
        strong_shared_peak = np.min(shared_support) >= minimum_shared_support
        if not (strong_residual or strong_shared_peak):
            candidate_strength[index] = -np.inf
            continue
        delay = float(delay_axis[index])
        guesses.append(delay)
        support.append(shared_support)
        residual_strength.append(float(residual_spectrum[index]))
        candidate_strength[abs(delay_axis-delay) <= separation] = -np.inf
    support_array = np.asarray(support).reshape(-1, image_count)
    return (np.asarray(guesses), delay_resolution, support_array,
            np.asarray(residual_strength), delay_axis, residual_spectrum,
            amplitudes)


def _estimate_ghost_coefficients(w: np.ndarray, residual: np.ndarray,
                                 delays: np.ndarray
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Solve all ghost coefficients together for fixed Fourier delays.

    For fixed delays the ghost model is linear,

        c exp(i*w*t)/(1+w).

    so the complex coefficients for all ghosts are obtained in one least
    squares solve.  Their phases are ``arg(c)`` and magnitudes are ``|c|^2``.
    """
    if not len(delays):
        empty = np.array([], dtype=float)
        return empty, empty
    templates = np.exp(1j*w[:, None]*delays[None, :])/(1+w[:, None])
    coefficients = np.linalg.lstsq(templates, residual, rcond=None)[0]
    return np.angle(coefficients), abs(coefficients)**2


def _fit_knots(w: np.ndarray, wave: np.ndarray, known_term: np.ndarray,
               delays: np.ndarray,
               magnifications: np.ndarray, morse_shifts: np.ndarray,
               coefficients: np.ndarray | None, start: float | None,
               ghost_mask: np.ndarray,
               knots: np.ndarray, l2_limit: float, fit_weight: float,
               amplitude_weight: float, curvature_weight: float,
               reference: ImageDecomposition | None,
               reference_weight: float,
               previous: ImageDecomposition | None,
               optimizer_maxiter: int) -> ImageDecomposition:
    """Fit one knot set; the data constraint is convex for fixed knots."""
    image_count, count = len(delays), len(knots)
    xq = np.linspace(knots[0], knots[-1], max(240, 24*count))
    wq = 10**xq
    qweight = np.ones(len(xq))/max(1, len(xq)-1)
    qweight[[0, -1]] *= 0.5
    b, uq, duq, dduq = _basis(knots, w, xq)
    phase = _phases(w, delays, morse_shifts)
    baseline = _base(w, magnifications, coefficients, start, ghost_mask)
    target = wave-known_term-np.sum(phase*baseline, axis=0)
    matrix = np.concatenate([phase[i, :, None]*b for i in range(image_count)], axis=1)
    wave_norm2 = np.vdot(wave, wave).real
    base_q, base_slope = _base(wq, magnifications, coefficients, start, ghost_mask,
                               derivative=True)
    reference_q = reference.components(wq) if reference is not None else None

    def relative_error(values: np.ndarray) -> float:
        residual = matrix @ values.ravel()-target
        return float(np.linalg.norm(residual)/np.sqrt(wave_norm2))

    # Refit the preceding smooth curves at the new knot positions. This is
    # the initial condition after either adding or removing a knot.
    earlier = previous if previous is not None else reference
    if earlier is None:
        initial = np.zeros((image_count, count), dtype=complex)
    else:
        knot_w = 10**knots
        initial = (earlier.components(knot_w)
                   -_base(knot_w, magnifications, coefficients, start,
                          ghost_mask))*(1+knot_w)[None, :]

    # Least squares tells us whether this knot space can meet the requested
    # error. If it cannot, use its smallest-L2 solution to locate the next
    # knot; a length penalty must not hide a feasible lower-error fit.
    least_squares = np.linalg.lstsq(matrix, target, rcond=None)[0].reshape(image_count, count)
    least_error = relative_error(least_squares)
    if least_error > l2_limit*(1+1e-9):
        values = least_squares
    else:
        # Start inside the allowed error ball rather than on its boundary.
        # This gives the constrained optimizer a numerically useful feasible
        # direction even when l2_limit is of order 1e-4.
        interior_limit = 0.5*(least_error+l2_limit)
        zero = np.zeros((image_count, count), dtype=complex)
        if relative_error(zero) <= l2_limit:
            smooth_feasible = zero
        else:
            # Find the most strongly ridge-regularized feasible solution.
            # This avoids the enormous cancelling coefficients produced by
            # an unregularized pseudoinverse when two image phases are nearly
            # indistinguishable over the supplied frequency interval.
            left_vectors, singular_values, right_vectors = np.linalg.svd(
                matrix, full_matrices=False)
            projected = left_vectors.conj().T @ target

            def ridge_solution(penalty: float) -> np.ndarray:
                factors = singular_values/(singular_values**2+penalty)
                flat = right_vectors.conj().T @ (factors*projected)
                return flat.reshape(image_count, count)

            scale = max(float(singular_values[0]**2), 1e-30)
            infeasible_penalty = scale
            if relative_error(ridge_solution(infeasible_penalty)) <= interior_limit:
                feasible_penalty = infeasible_penalty
                while relative_error(ridge_solution(infeasible_penalty)) <= interior_limit:
                    feasible_penalty = infeasible_penalty
                    infeasible_penalty *= 10
            else:
                while relative_error(ridge_solution(infeasible_penalty)) > interior_limit:
                    infeasible_penalty /= 10
                feasible_penalty = infeasible_penalty
                infeasible_penalty *= 10
            for _ in range(50):
                penalty = np.sqrt(feasible_penalty*infeasible_penalty)
                if relative_error(ridge_solution(penalty)) <= interior_limit:
                    feasible_penalty = penalty
                else:
                    infeasible_penalty = penalty
            smooth_feasible = ridge_solution(feasible_penalty)

        if relative_error(initial) > l2_limit:
            initial = smooth_feasible

        def unpack(parameters: np.ndarray) -> np.ndarray:
            return parameters.reshape(image_count, 2, count)[:, 0] + 1j*parameters.reshape(
                image_count, 2, count)[:, 1]

        def pack(values: np.ndarray) -> np.ndarray:
            return np.stack((values.real, values.imag), axis=1).ravel()

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            values = unpack(parameters)
            residual = matrix @ values.ravel()-target
            error2 = np.vdot(residual, residual).real/wave_norm2
            data_gradient = 2*matrix.conj().T @ residual/wave_norm2
            gradient = data_gradient.reshape(image_count, count)*fit_weight/l2_limit**2
            cost = fit_weight*error2/l2_limit**2

            correction = values @ uq.T
            slopes = base_slope+values @ duq.T
            speed = np.sqrt(1e-10+abs(slopes)**2)
            cost += np.sum(speed @ qweight)
            gradient += ((qweight[None, :]*slopes/speed) @ duq)

            cost += amplitude_weight*np.sum(abs(correction)**2 @ qweight)
            gradient += 2*amplitude_weight*((qweight[None, :]*correction) @ uq)

            acceleration = values @ dduq.T
            cost += curvature_weight*np.sum(abs(acceleration)**2 @ qweight)
            gradient += 2*curvature_weight*((qweight[None, :]*acceleration) @ dduq)

            if reference_q is not None and reference_weight:
                deviation = base_q+correction-reference_q
                cost += reference_weight*np.sum(abs(deviation)**2 @ qweight)
                gradient += 2*reference_weight*((qweight[None, :]*deviation) @ uq)
            return float(cost), pack(gradient)

        def constraint(parameters: np.ndarray) -> float:
            values = unpack(parameters)
            residual = matrix @ values.ravel()-target
            return float(1-np.vdot(residual, residual).real/(wave_norm2*l2_limit**2))

        def constraint_gradient(parameters: np.ndarray) -> np.ndarray:
            values = unpack(parameters)
            residual = matrix @ values.ravel()-target
            derivative = -2*matrix.conj().T @ residual/(wave_norm2*l2_limit**2)
            return pack(derivative.reshape(image_count, count))

        # SLSQP occasionally leaves the feasible region while exploring an
        # ill-conditioned multi-image split. Keep the lowest-cost feasible
        # iterate, including the canonical ridge starting point. This is a
        # deterministic safeguard within the stated optimization, rather
        # than a switch to a different fitting rule after failure.
        best_values = initial.copy()
        best_cost = objective(pack(initial))[0]

        def remember_feasible(parameters: np.ndarray) -> None:
            nonlocal best_values, best_cost
            candidate = unpack(parameters)
            if relative_error(candidate) <= l2_limit*(1+1e-7):
                candidate_cost = objective(parameters)[0]
                if candidate_cost < best_cost:
                    best_values = candidate.copy()
                    best_cost = candidate_cost

        answer = minimize(objective, pack(initial), jac=True, method='SLSQP',
                          constraints={'type': 'ineq', 'fun': constraint,
                                       'jac': constraint_gradient},
                          callback=remember_feasible,
                          options={'maxiter': optimizer_maxiter, 'ftol': 1e-10})
        remember_feasible(answer.x)
        values = best_values

    correction = values @ uq.T
    slopes = base_slope+values @ duq.T
    acceleration = values @ dduq.T
    length = float(np.sum(np.sqrt(1e-10+abs(slopes)**2) @ qweight))
    smoothness_cost = length
    smoothness_cost += amplitude_weight*float(np.sum(abs(correction)**2 @ qweight))
    smoothness_cost += curvature_weight*float(np.sum(abs(acceleration)**2 @ qweight))
    smoothness_cost += fit_weight*(relative_error(values)/l2_limit)**2
    if reference_q is not None and reference_weight:
        deviation = base_q+correction-reference_q
        smoothness_cost += reference_weight*float(np.sum(abs(deviation)**2 @ qweight))
    return ImageDecomposition(knots, values, delays, magnifications, morse_shifts,
                              coefficients, start, relative_error(values), length, (0, 0), (),
                              w if np.any(known_term) else None,
                              known_term if np.any(known_term) else None,
                              int(np.count_nonzero(~ghost_mask)), ghost_mask.copy(),
                              smoothness_cost)


def decompose(w: np.ndarray, wave: np.ndarray, delays: np.ndarray,
              magnifications: np.ndarray, morse_shifts: np.ndarray,
              l2_limit: float,
              bgo_coefficients: np.ndarray | None = None,
              known_term: np.ndarray | None = None,
              ghost_count: int = 0,
              ghost_delay_bounds: tuple[float, float] | None = None,
              ghost_search_points: int = 401,
              ghost_prefit_l2: float = 0.01,
              ghost_delay_guesses: np.ndarray | None = None,
              ghost_min_residual_strength: float = 0.1,
              ghost_min_shared_support: float = 0.1,
              min_knots: int = 3, max_initial_knots: int = 20,
              max_knots: int | None = None,
              bgo_start: float | None = None, bgo_tolerance: float = 0.01,
              bgo_image_fraction: float = 0.25,
              fit_weight: float = 0.01, amplitude_weight: float = 0.02,
              curvature_weight: float = 0.02,
              reference: ImageDecomposition | None = None,
              reference_weight: float = 0.02,
              optimizer_maxiter: int = 500,
              factor_scale_limit: float | None = None) -> ImageDecomposition:
    """Find smooth image factors and adapt their common cubic-spline knots.

    ``l2_limit`` bounds ``||F_fit-F||_2/||F||_2`` on the supplied samples.
    ``known_term`` is an optional separate sampled contribution to F, such
    as a known SIS center term. ``ghost_count`` is the maximum number of
    additional vanishing terms to infer.  A preliminary fit without ghosts,
    at relative error ``ghost_prefit_l2``, exposes an uncaptured oscillation in
    its residual.  The ghost delays are the strongest residual Fourier peaks
    separated from every supplied image delay.  Fourier spectra of the
    physical image contributions are also recorded to test whether the same
    delay is shared among them.  The ghost phase and amplitude are then solved
    analytically. Candidates weaker than both Fourier support thresholds are
    discarded. Its factor baseline is ``sqrt(mu_g)/(1+w)``.
    The initial knot count is ``clip(P_wave+2|P_wave-P_geo|, min_knots,
    max_initial_knots)``, where each P counts strict local maxima of ``|F|``.
    Every candidate uses uniformly spaced knots in ``log10(w)``. Increasing
    or decreasing the knot count refits the preceding spline analytically at
    the new uniform knot positions before optimizing the factor values. A
    previous decomposition may be supplied as a reference for nearby lens
    settings.

    ``factor_scale_limit`` optionally rejects decompositions whose largest
    factor magnitude exceeds this multiple of the largest physical geometric
    amplitude. The uniform knot count is increased until both the L2 and
    factor-scale conditions hold. Set ``optimizer_maxiter=0`` to use the
    analytic ridge refit directly, without the iterative SLSQP improvement.

    If the requested accuracy cannot be reached by ``max_knots``, raise a
    RuntimeError with the best attained error rather than returning a fit
    that silently violates the limit.
    """
    w = np.asarray(w, dtype=float)
    wave = np.asarray(wave, dtype=complex)
    delays = np.asarray(delays, dtype=float)
    magnifications = np.asarray(magnifications, dtype=float)
    morse_shifts = np.asarray(morse_shifts, dtype=float)
    if w.ndim != 1 or len(w) < 4 or np.any(w <= 0) or np.any(np.diff(w) <= 0):
        raise ValueError('w must contain at least four increasing positive samples')
    if wave.shape != w.shape:
        raise ValueError('wave and w must have the same shape')
    known_term = (np.asarray(known_term, dtype=complex) if known_term is not None
                  else np.zeros_like(w, dtype=complex))
    if known_term.shape != w.shape:
        raise ValueError('known_term and w must have the same shape')
    if delays.ndim != 1 or not len(delays):
        raise ValueError('at least one image is required')
    if not (delays.shape == magnifications.shape == morse_shifts.shape):
        raise ValueError('delays, magnifications, and morse_shifts must have the same shape')
    max_initial_knots = min(max_initial_knots, len(w))
    if not 3 <= min_knots <= max_initial_knots:
        raise ValueError('require 3 <= min_knots <= max_initial_knots')
    if l2_limit <= 0:
        raise ValueError('l2_limit must be positive')
    if factor_scale_limit is not None and factor_scale_limit <= 0:
        raise ValueError('factor_scale_limit must be positive or None')
    if optimizer_maxiter < 0:
        raise ValueError('optimizer_maxiter must be non-negative')
    if int(ghost_count) != ghost_count or ghost_count < 0:
        raise ValueError('ghost_count must be a non-negative integer')
    if ghost_prefit_l2 <= 0:
        raise ValueError('ghost_prefit_l2 must be positive')
    if not 0 <= ghost_min_residual_strength <= 1:
        raise ValueError('ghost_min_residual_strength must lie between zero and one')
    if not 0 <= ghost_min_shared_support <= 1:
        raise ValueError('ghost_min_shared_support must lie between zero and one')
    ghost_count = int(ghost_count)
    physical_image_count = len(delays)
    if bgo_coefficients is not None:
        bgo_coefficients = np.asarray(bgo_coefficients, dtype=complex)
        if bgo_coefficients.ndim != 2 or bgo_coefficients.shape[0] != len(delays):
            raise ValueError('bgo_coefficients must have shape (number of images, order)')
        if bgo_start is None:
            bgo_start = _bgo_start(w, wave, delays, magnifications, morse_shifts,
                                   bgo_coefficients, known_term, bgo_tolerance,
                                   bgo_image_fraction)

    # A cheap physical-image-only fit reveals a missing coherent contribution
    # as the same absolute-delay Fourier peak in every image contribution.
    ghost_delay_resolution = None
    ghost_spectral_support = None
    ghost_residual_spectral_strength = None
    ghost_spectrum_delays = None
    ghost_residual_spectrum = None
    ghost_component_spectra = None
    ghost_prefit_relative_l2 = None
    if ghost_delay_guesses is not None:
        ghost_delays = np.asarray(ghost_delay_guesses, dtype=float)
        if ghost_delays.shape != (ghost_count,):
            raise ValueError('ghost_delay_guesses must have one entry per requested ghost')
    elif ghost_count:
        prefit = decompose(
            w=w, wave=wave, delays=delays, magnifications=magnifications,
            morse_shifts=morse_shifts, l2_limit=ghost_prefit_l2,
            bgo_coefficients=bgo_coefficients, known_term=known_term,
            ghost_count=0, min_knots=min_knots,
            max_initial_knots=max_initial_knots, max_knots=len(w),
            bgo_start=bgo_start, bgo_tolerance=bgo_tolerance,
            bgo_image_fraction=bgo_image_fraction,
            fit_weight=fit_weight, amplitude_weight=amplitude_weight,
            curvature_weight=curvature_weight, reference=reference,
            reference_weight=reference_weight, optimizer_maxiter=0,
            # This is a diagnostic fit whose purpose is to expose the shared
            # oscillation.  Applying the final factor-scale rejection here can
            # prevent a no-ghost fit from reaching even the loose 1% target.
            factor_scale_limit=None)
        (ghost_delays, ghost_delay_resolution, ghost_spectral_support,
         ghost_residual_spectral_strength, ghost_spectrum_delays,
         ghost_residual_spectrum, ghost_component_spectra) = _shared_ghost_delays(
            w, wave, prefit, delays, magnifications, morse_shifts,
            bgo_coefficients, bgo_start, ghost_count,
            ghost_delay_bounds, ghost_search_points,
            ghost_min_residual_strength, ghost_min_shared_support)
        ghost_prefit_relative_l2 = prefit.relative_l2
    else:
        ghost_delays = np.array([], dtype=float)

    # A requested count is an upper limit when delays are inferred: weak
    # Fourier peaks do not justify adding a coherent term.  Explicit user
    # guesses retain exactly the requested count.
    ghost_count = len(ghost_delays)

    # Once the delays are known, solve the constant ghost coefficients from
    # what remains after the physical geometric/BGO terms.
    physical_base = _base(w, magnifications, bgo_coefficients, bgo_start)
    residual = (wave-known_term
                -np.sum(_phases(w, delays, morse_shifts)*physical_base, axis=0))
    ghost_phases, ghost_magnifications = _estimate_ghost_coefficients(
        w, residual, ghost_delays)
    if ghost_count:
        delays = np.r_[delays, ghost_delays]
        morse_shifts = np.r_[morse_shifts, ghost_phases]
        magnifications = np.r_[magnifications, ghost_magnifications]
        if bgo_coefficients is not None:
            bgo_coefficients = np.vstack((bgo_coefficients,
                                          np.zeros((ghost_count,
                                                    bgo_coefficients.shape[1]))))
    ghost_mask = np.zeros(len(delays), dtype=bool)
    ghost_mask[physical_image_count:] = True
    if max_knots is None:
        max_knots = min(len(w), 80)
    if max_knots < min_knots:
        raise ValueError('max_knots must be at least min_knots')
    max_initial_knots = min(max_initial_knots, max_knots)

    baseline_signal = np.sum(_phases(w, delays, morse_shifts)
                             *_base(w, magnifications, bgo_coefficients,
                                    bgo_start, ghost_mask), axis=0)

    def peak_count(curve: np.ndarray) -> int:
        amplitude = abs(curve)
        return int(np.count_nonzero((amplitude[1:-1] > amplitude[:-2])
                                    & (amplitude[1:-1] > amplitude[2:])))

    peaks = peak_count(wave), peak_count(baseline_signal)
    count = int(np.clip(peaks[0]+2*abs(peaks[0]-peaks[1]),
                        min_knots, max_initial_knots))
    x = np.log10(w)
    knots = np.linspace(x[0], x[-1], count)
    history: list[tuple[int, float]] = []

    def fit(current_knots: np.ndarray, preceding: ImageDecomposition | None,
            maxiter: int) -> ImageDecomposition:
        result = _fit_knots(w, wave, known_term, delays, magnifications, morse_shifts,
                            bgo_coefficients, bgo_start, ghost_mask, current_knots, l2_limit,
                            fit_weight, amplitude_weight, curvature_weight,
                            reference, reference_weight, preceding, maxiter)
        history.append((len(current_knots), result.relative_l2))
        result.peak_counts = peaks
        result.history = tuple(history)
        result.ghost_prefit_relative_l2 = ghost_prefit_relative_l2
        result.ghost_delay_resolution = ghost_delay_resolution
        result.ghost_spectral_support = ghost_spectral_support
        result.ghost_residual_spectral_strength = ghost_residual_spectral_strength
        result.ghost_spectrum_delays = ghost_spectrum_delays
        result.ghost_residual_spectrum = ghost_residual_spectrum
        result.ghost_component_spectra = ghost_component_spectra
        return result

    geometric_scale = float(np.max(np.sqrt(abs(magnifications[:physical_image_count]))))

    def factor_scale(result: ImageDecomposition) -> float:
        sample_w = np.geomspace(w[0], w[-1], max(240, 24*len(result.knots)))
        return float(np.max(abs(result.components(sample_w)))/geometric_scale)

    def acceptable(result: ImageDecomposition) -> bool:
        accurate = result.relative_l2 <= l2_limit*(1+1e-7)
        well_scaled = (factor_scale_limit is None
                       or factor_scale(result) <= factor_scale_limit)
        return accurate and well_scaled

    # Knot-count changes use the analytic ridge solution. Only the final
    # selected count is optionally improved with the iterative optimizer.
    current = fit(knots, None, 0)
    if acceptable(current):
        # Reduce only the integer knot count. The new values are initialized
        # analytically by evaluating the preceding cubic splines at the new
        # uniformly spaced knots inside _fit_knots.
        while len(current.knots) > min_knots:
            proposed_knots = np.linspace(x[0], x[-1], len(current.knots)-1)
            proposed = fit(proposed_knots, current, 0)
            if not acceptable(proposed):
                break
            current = proposed
    else:
        # Increase only the integer knot count, retaining uniform spacing.
        while not acceptable(current) and len(current.knots) < max_knots:
            proposed_knots = np.linspace(x[0], x[-1], len(current.knots)+1)
            current = fit(proposed_knots, current, 0)
        if not acceptable(current):
            raise RuntimeError(
                f'The L2/scale conditions were not reached with {len(current.knots)} '
                f'knots; attained L2={current.relative_l2:.4g}, '
                f'factor scale={factor_scale(current):.4g}')

    if optimizer_maxiter:
        optimized = fit(current.knots, current, optimizer_maxiter)
        if not acceptable(optimized):
            raise RuntimeError('The final smoothness optimization violated the accepted '
                               'L2 or factor-scale condition')
        current = optimized
    current.history = tuple(history)
    return current
