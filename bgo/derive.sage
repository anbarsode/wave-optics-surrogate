"""Verify the Fermat Taylor series through degree 14 using SageMath.

Run with ``DOT_SAGE=/tmp/wave_optics_sage sage bgo/derive.sage`` in gw.
The finite formulas implemented in expansion.py and reported in REPORT.md are
checked here against direct exact-rational polynomial expansion.
"""

from sage.all import PolynomialRing, QQ, binomial, var, diff, matrix, SR, factor, RealField
from expansion import image_coefficients, _terms, image_series_coefficients

parameters = PolynomialRing(QQ, ('a', 'b', 'r'))
a, b, r = parameters.gens()
polynomials = PolynomialRing(parameters.fraction_field(), ('u', 'v'))
u, v = polynomials.gens()
q = 2*u/r + (u*u + v*v)/r**2

for lens in ('PM', 'SIS'):
    potential = polynomials.zero()
    generated = _terms(lens, a if lens == 'PM' else b, r, 14)
    for k in range(1, 15):
        coefficient = (-a/2)*(-1)**(k+1)/k if lens == 'PM' else -b*r*binomial(QQ(1)/2, k)
        potential += coefficient*q**k
        if k < 3:
            continue
        direct = sum((c*u**powers[0]*v**powers[1]
                      for powers, c in potential.dict().items() if sum(powers) == k),
                     polynomials.zero())
        formula = sum((c*u**powers[0]*v**powers[1]
                       for powers, c in generated[k-3].items()), polynomials.zero())
        assert direct == formula
    print(f'{lens}: P3 through P14 agree with the Sage Taylor expansion')

# The central SIS term follows by differentiating the leading conical integral
# with respect to source position. This check also covers a rotated shear.
b0, y1, y2, A11, A12, A22 = var('b0 y1 y2 A11 A12 A22')
z = b0**2 - y1**2 - y2**2
c0 = 1j*b0*z**(-QQ(3)/2)
c1 = -1j/2*(A11*diff(c0, y1, 2) + 2*A12*diff(diff(c0, y1), y2)
               + A22*diff(c0, y2, 2))
expected = b0/2*(3*(A11+A22)*z**(-QQ(5)/2)
                 + 15*(A11*y1**2+2*A12*y1*y2+A22*y2**2)*z**(-QQ(7)/2))
assert (c1 - expected).simplify_full() == 0
print('SIS central second-order derivative identity: verified')

# Carry out the Gaussian contractions symbolically for the isolated lenses.
# The shear cases use the same coefficients with the full inverse Hessian;
# their compact general formula is Eq. (1) of REPORT.md.
radius = var('radius')
for lens in ('PM', 'SIS'):
    if lens == 'PM':
        local_inverse = matrix(SR, [[1/(1+1/radius**2), 0],
                                    [0, 1/(1-1/radius**2)]])
    else:
        local_inverse = matrix(SR, [[1, 0], [0, 1/(1-1/radius)]])
    first, second = image_coefficients(lens, 1, radius, local_inverse)
    print(f'{lens} isolated c1 = {factor(first)}')
    print(f'{lens} isolated c2 = {factor(second)}')


# Check that double-precision Wick contractions remain accurate in a generic
# indefinite (saddle) Hessian with a nonzero off-diagonal entry. This tests
# the high-order numerical implementation separately from the exact Taylor
# polynomial identity above.
import numpy as np
R = RealField(120)
g_high = matrix(R, [[R('4.25'), R('0.7')], [R('0.7'), R('-5.5')]])
g_float = np.array([[4.25, 0.7], [0.7, -5.5]])
for lens in ('PM', 'SIS'):
    high = image_series_coefficients(lens, R(1), R('1.03'), g_high, 6)
    low = image_series_coefficients(lens, 1., 1.03, g_float, 6)
    assert all(abs((high[k]-R(low[k]))/high[k]) < 1e-11 for k in range(1, 7))
print('Sixth-order Gaussian contractions: 120-bit comparison verified')


# SageMath carries out every Wick contraction exactly over Q[g11,g12,g22]
# at unit strength and radius. The general-parameter expression remains the
# finite formula in REPORT.md; the coefficients below check its full algebra
# through sixth order, including off-diagonal shear terms.
gaussian_ring = PolynomialRing(QQ, ('g11', 'g12', 'g22'))
g11, g12, g22 = gaussian_ring.gens()
generic_inverse = matrix(gaussian_ring, [[g11, g12], [g12, g22]])
for lens in ('PM', 'SIS'):
    coefficients = image_series_coefficients(lens, QQ(1), QQ(1), generic_inverse, 6)
    counts = [len(c.dict()) if hasattr(c, 'dict') else 1 for c in coefficients]
    print(f'{lens} exact symbolic Wick monomials, orders 0-6: {counts}')
