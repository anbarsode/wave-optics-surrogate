"""Numerical checks for the image split and reusable surrogate files."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from bgo.expansion import amplification as bgo_amplification, isolated_parts
from decomposition.surrogate import (CanonicalBasis, FullSurrogate, geometry,
                                     image_phases, minimum_smoothness_fit)


def test_bgo_parts_rebuild_full_expansion():
    """The separately stored SIS center term must not be lost or double counted."""
    w = np.geomspace(0.2, 30, 50)
    for model in ('PM', 'SIS'):
        magnitudes, delays = geometry(model, 0.5)
        corrections, center = isolated_parts(w, model, 0.5, order=2)
        rebuilt = center + np.sum(image_phases(w, delays)
                                  *(magnitudes[:, None]+corrections), axis=0)
        complete = bgo_amplification(w, model, [0.5, 0], order=2)
        np.testing.assert_allclose(rebuilt, complete, rtol=1e-12, atol=1e-12)


def test_three_image_canonical_fit():
    """One positive norm fixes all three image corrections without initial guesses."""
    w = np.geomspace(0.01, 20, 120)
    basis = CanonicalBasis(0.01, 20, knot_count=8)
    delays = np.array([0.0, 0.4, 1.1])
    phases = np.exp(1j*w[None, :]*delays[:, None])
    base = np.ones((3, len(w)), dtype=complex)
    center = np.zeros(len(w), dtype=complex)
    true_values = np.array([[0.05*np.exp(1j*j)*(i+1)/8 for j in range(8)]
                            for i in range(3)])
    wave = np.sum(phases*(base+true_values @ basis.evaluate(w).T), axis=0)
    fitted, error, _ = minimum_smoothness_fit(w, wave, phases, base, center,
                                               basis, rms_limit=1e-4)
    reconstructed = np.sum(phases*(base+fitted @ basis.evaluate(w).T), axis=0)
    assert error <= 1e-4*(1+1e-8)
    assert np.max(abs(reconstructed-wave)/abs(wave)) < 5e-4
    repeated, _, _ = minimum_smoothness_fit(w, wave, phases, base, center,
                                             basis, rms_limit=1e-4)
    np.testing.assert_allclose(repeated, fitted, atol=1e-12)


def test_saved_surrogate_reproduces_evaluation():
    """The compact saved representation evaluates without training waveforms."""
    knots = np.linspace(-2, np.log10(20), 6)
    source_positions = np.array([0.3, 0.5, 0.7, 0.9])
    coefficients = np.ones((4, 2, 6), dtype=complex)*0.02j
    original = FullSurrogate('SIS', 0, source_positions, knots, coefficients,
                             np.ones(4)*100)
    with TemporaryDirectory() as directory:
        path = Path(directory)/'surrogate.npz'
        original.save(path)
        loaded = FullSurrogate.load(path)
        w = np.geomspace(0.01, 20, 50)
        np.testing.assert_allclose(loaded.amplification(w, 0.6),
                                   original.amplification(w, 0.6))


if __name__ == '__main__':
    test_bgo_parts_rebuild_full_expansion()
    test_three_image_canonical_fit()
    test_saved_surrogate_reproduces_evaluation()
    print('All surrogate checks passed.')
