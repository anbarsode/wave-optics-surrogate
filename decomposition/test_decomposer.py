"""Numerical checks for adaptive knots and the general image input arrays."""

import numpy as np

from decomposition.decomposer import decompose
from decomposition.isolated import isolated_lens
from decomposition.surrogate import geometry


def test_geometric_signal_removes_unneeded_knots():
    """Ten interference peaks need only three knots when G_i are constant."""
    w = np.linspace(0.05, 30, 180)
    delays = np.array([0.0, 2.0])
    magnifications = np.array([2.0, -1.0])
    morse_shifts = np.array([0.0, -np.pi/2])
    phase = np.exp(1j*(delays[:, None]*w+morse_shifts[:, None]))
    wave = np.sum(phase*np.sqrt(abs(magnifications))[:, None], axis=0)
    result = decompose(w, wave, delays, magnifications, morse_shifts, 0.001,
                       max_initial_knots=15)
    assert result.peak_counts == (10, 10)
    assert result.history[0][0] == 10
    assert len(result.knots) == 3
    assert result.relative_l2 < 1e-10


def test_three_images_add_knots_where_needed():
    """Localized image corrections require additions and meet the L2 limit."""
    w = np.geomspace(0.02, 20, 150)
    x = np.log10(w)
    delays = np.array([0.0, 0.7, 1.9])
    magnifications = np.array([2.0, -0.8, 0.3])
    morse_shifts = np.array([0.0, -np.pi/2, -np.pi])
    correction = np.array([
        0.15j*np.exp(-((x+0.7)/0.15)**2),
        0.08*np.exp(-((x+0.3)/0.13)**2),
        0.05j*np.exp(-((x-0.2)/0.18)**2),
    ])
    phase = np.exp(1j*(delays[:, None]*w+morse_shifts[:, None]))
    wave = np.sum(phase*(np.sqrt(abs(magnifications))[:, None]+correction), axis=0)
    result = decompose(w, wave, delays, magnifications, morse_shifts, 0.004,
                       max_initial_knots=5, max_knots=20)
    assert len(result.knots) > result.history[0][0]
    assert result.relative_l2 <= 0.004*(1+1e-7)
    np.testing.assert_allclose(np.diff(result.knots), np.diff(result.knots)[0])
    np.testing.assert_allclose(result.amplification(w),
                               np.sum(phase*result.components(w), axis=0))


def test_optional_bgo_coefficients():
    """The fitted factors retain the supplied stationary-phase convention."""
    w = np.geomspace(0.05, 20, 120)
    delays = np.array([0.0, 1.0])
    magnifications = np.array([2.0, -0.5])
    morse_shifts = np.array([0.0, -np.pi/2])
    coefficients = np.array([[0.05, 0.01], [0.2, 0.03]])
    coordinate = np.clip(np.log(w)/np.log(4), 0, 1)
    window = coordinate**2*(3-2*coordinate)
    correction = sum(coefficients[:, order-1, None]*(1j/w[None, :])**order
                     for order in (1, 2))
    phase = np.exp(1j*(delays[:, None]*w+morse_shifts[:, None]))
    center = 0.02*np.exp(0.3j*w)/(1+w)
    wave = center+np.sum(phase*np.sqrt(abs(magnifications))[:, None]
                         *(1+window[None, :]*correction), axis=0)
    result = decompose(w, wave, delays, magnifications, morse_shifts, 0.003,
                       bgo_coefficients=coefficients, known_term=center,
                       bgo_start=1.0)
    assert result.relative_l2 <= 0.003*(1+1e-7)
    assert result.bgo_start == 1.0
    assert result.known_term is not None
    np.testing.assert_allclose(result.components(np.array([10000.0]))[:, 0],
                               np.sqrt(abs(magnifications)), rtol=1e-3)


def test_exact_pm_and_sis_inputs():
    """The lens-independent interface also fits the project's two examples."""
    w = np.geomspace(0.01, 20, 120)
    for model in ('PM', 'SIS'):
        wave = isolated_lens(model, w, 0.5).wave
        amplitudes, delays = geometry(model, 0.5)
        signed_magnifications = np.array([amplitudes[0]**2, -amplitudes[1]**2])
        result = decompose(w, wave, delays, signed_magnifications,
                           np.array([0.0, -np.pi/2]), 0.01,
                           max_initial_knots=12, max_knots=24)
        assert result.relative_l2 <= 0.01*(1+1e-7)


def test_unreachable_limit_is_reported():
    """A three-knot cap must not silently return a fit above the limit."""
    w = np.geomspace(0.02, 20, 100)
    x = np.log10(w)
    delays = np.array([0.0])
    magnifications = np.array([1.0])
    morse_shifts = np.array([0.0])
    wave = 1+0.1j*np.exp(-((x+0.4)/0.12)**2)
    try:
        decompose(w, wave, delays, magnifications, morse_shifts, 1e-4,
                  max_initial_knots=3, max_knots=3)
    except RuntimeError as error:
        assert 'not reached' in str(error)
    else:
        raise AssertionError('An unattainable error limit was silently accepted')


def test_user_selected_ghosts_recover_extra_phases():
    """Explicit initial delays preserve two known coherent contributions."""
    w = np.linspace(0.2, 30, 220)
    physical = np.ones_like(w, dtype=complex)
    ghost_1 = 0.35/(1+w)*np.exp(1j*(0.73*w+0.4))
    ghost_2 = 0.18/(1+w)*np.exp(1j*(-0.31*w-0.8))
    wave = physical+ghost_1+ghost_2
    result = decompose(w, wave, np.array([0.0]), np.array([1.0]), np.array([0.0]),
                       0.002, ghost_count=2, ghost_delay_bounds=(-1.5, 1.5),
                       ghost_delay_guesses=np.array([0.73, -0.31]),
                       max_initial_knots=8, max_knots=20)
    assert result.ghost_count == 2
    assert result.relative_l2 <= 0.002*(1+1e-7)
    assert result.components(w).shape == (3, len(w))
    np.testing.assert_allclose(np.sort(result.ghost_delays), [-0.31, 0.73], atol=0.05)
    assert np.all(result.ghost_magnifications > 0)


def test_fourier_ghost_delay_finds_remaining_oscillation():
    """The residual spectrum recovers an uncaptured delay to its FFT resolution."""
    w = np.linspace(0.2, 30, 220)
    ghost_delay = 0.73
    wave = 1+0.35/(1+w)*np.exp(1j*(ghost_delay*w+0.4))
    result = decompose(w, wave, np.array([0.0]), np.array([1.0]), np.array([0.0]),
                       0.002, ghost_count=1, ghost_delay_bounds=(-1.5, 1.5),
                       max_initial_knots=8, max_knots=20)
    assert abs(result.ghost_delays[0]-ghost_delay) <= result.ghost_delay_resolution
    assert result.ghost_residual_spectral_strength[0] > 0.5
    assert result.relative_l2 <= 0.002*(1+1e-7)


if __name__ == '__main__':
    test_geometric_signal_removes_unneeded_knots()
    test_three_images_add_knots_where_needed()
    test_optional_bgo_coefficients()
    test_exact_pm_and_sis_inputs()
    test_unreachable_limit_is_reported()
    test_user_selected_ghosts_recover_extra_phases()
    test_fourier_ghost_delay_finds_remaining_oscillation()
    print('All adaptive-decomposer checks passed.')
