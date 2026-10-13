"""Independent checks of image finding and the asymptotic coefficients."""

import os

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from glow import lenses, time_domain_c, freq_domain_c

from expansion import _image_data, amplification


def main():
    # Takahashi (2004), Eq. 27, is an independent first-order PM expression.
    for y in (0.2, 0.3, 0.8, 1.5, 3.0):
        for image in _image_data('PM', [y, 0], 1, (0, 0)):
            x = image['x'][0]
            published = (4*x*x-1)/(3*(x*x+1)**3*(x*x-1))
            assert abs(image['first']-published) < 1e-9

    # All distinct ordinary-image counts, including rotated external shear.
    shear = (0.05, 0.03)
    for model, y, count in (('PM', 0.3, 2), ('PM', 0.03, 4),
                            ('SIS', 0.3, 2), ('SIS', 0.03, 4), ('SIS', 1.5, 1)):
        assert len(_image_data(model, [y, 0], 1, shear)) == count

    # Rotating both source and shear must leave F unchanged. This also checks
    # non-unit lens strengths and sources away from the coordinate axes.
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    source = np.array([0.23, 0.11])
    shear_matrix = np.array([[shear[0], shear[1]], [shear[1], -shear[0]]])
    rotated_shear = rotation @ shear_matrix @ rotation.T
    for model in ('PM', 'SIS'):
        original = amplification(np.array([10., 30., 100.]), model, source,
                                 strength=0.8, shear=shear, order=2)
        rotated = amplification(np.array([10., 30., 100.]), model, rotation @ source,
                                strength=0.8, order=2,
                                shear=(rotated_shear[0, 0], rotated_shear[0, 1]))
        assert np.max(abs(original-rotated)) < 1e-10

    # The full sixth-order sheared formula must be covariant under rotation.
    for model in ('PM', 'SIS'):
        frequencies = np.array([80., 160., 320.])
        original = amplification(frequencies, model, source, strength=0.8,
                                 shear=shear, order=6)
        rotated = amplification(frequencies, model, rotation @ source,
                                strength=0.8, order=6,
                                shear=(rotated_shear[0, 0], rotated_shear[0, 1]))
        assert np.max(abs(original-rotated)) < 1e-8

    # Exact GLoW routines check orders through six in their asymptotic regime.
    w = np.array([20., 40., 80.])
    for model, solver in (('PM', freq_domain_c.Fw_AnalyticPointLens_C(0.3, {'parallel': False})),
                          ('SIS', freq_domain_c.Fw_SemiAnalyticSIS_C(0.3, {'parallel': False}))):
        exact = solver(w)
        errors = [abs(amplification(w, model, [0.3, 0], order=k)-exact) for k in range(7)]
        assert np.all(errors[2] < errors[1])
        assert np.all(errors[1] < errors[0])
        assert errors[6][-1] < errors[2][-1]
        print(model, 'two-image errors at w=20,40,80:')
        for order, error in enumerate(errors):
            print('  order', order, np.round(error, 7))

    # The one-image SIS checks the opposite branch of the center term.
    solver = freq_domain_c.Fw_SemiAnalyticSIS_C(1.5, {'parallel': False})
    exact = solver(w)
    error_two = abs(amplification(w, 'SIS', [1.5, 0], order=2)-exact)
    error_six = abs(amplification(w, 'SIS', [1.5, 0], order=6)-exact)
    assert error_six[-1] < error_two[-1]


    # The sheared high-frequency result is checked with GLoW's regularized
    # FFT. Its direct transform of a sampled time response fails this check
    # for the same settings; see reference_diagnostic.py.
    for model, shear in (('PM', (0.05, 0.0)), ('SIS', (0.05, 0.03))):
        base = (lenses.Psi_PointLens if model == 'PM' else lenses.Psi_SIS)({'psi0': 1})
        lens = lenses.CombinedLens({'lenses': [
            base, lenses.Psi_Ext({'gamma1': shear[0], 'gamma2': shear[1]})]})
        response = time_domain_c.It_MultiContour_C(lens, 0.3,
                         {'Nt': 10000, 'tmax': 1e4, 'parallel': False})
        transform = freq_domain_c.Fw_FFT_C(response,
                         {'wmin': 10, 'wmax': 100, 'eval_mode': 'exact',
                          'parallel': False})
        frequencies = np.array([40., 80.])
        exact = transform(frequencies)
        sixth = amplification(frequencies, model, [0.3, 0],
                              shear=shear, order=6)
        assert np.max(abs(sixth-exact)) < 1e-3


if __name__ == '__main__':
    main()
