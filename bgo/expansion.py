"""Stationary-phase amplification through sixth order for PM and SIS lenses.

The full derivation, including the SIS conical-center series, is in REPORT.md.
All coefficients are finite sums of lens parameters, image positions, and H^-1.
The first-order framework follows Takahashi (2004), Appendix B:
https://arxiv.org/abs/astro-ph/0402165
"""

import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

from functools import lru_cache
from math import comb, factorial, pi, sqrt
from fractions import Fraction
import numpy as np


def _multiply(p, q):
    """Multiply sparse polynomials in radial and transverse displacements."""
    result = {}
    for (i, j), a in p.items():
        for (k, l), b in q.items():
            key = (i+k, j+l)
            result[key] = result.get(key, 0) + a*b
    return result


def _terms(model, strength, radius, max_order=6):
    """Return P3,...,Pmax as homogeneous local Fermat polynomials.

    For PM, Pn is the degree-n part of -a log(sqrt((r+u)^2+v^2)).
    For SIS, it is the degree-n part of -b sqrt((r+u)^2+v^2).
    External shear is quadratic and adds no terms here.
    """
    result = []
    r = radius
    for n in range(3, max_order+1):
        terms = {}
        for j in range(n//2+1):
            h = n-2*j
            if model == 'PM':
                c = (Fraction((-1)**n, n) if j == 0 else
                     Fraction((-1)**(j+h)*comb(n-1, h), 2*j))
                terms[(h, 2*j)] = strength*c.numerator/(c.denominator*r**n)
            elif model == 'SIS' and j:
                # binom(1/2,j), evaluated exactly before conversion to float.
                bj = Fraction(1)
                for k in range(1, j+1):
                    bj *= Fraction(3-2*k, 2*k)
                c = -bj*(-1)**h*comb(2*j+h-2, h)
                terms[(h, 2*j)] = strength*c.numerator/(c.denominator*r**(n-1))
        result.append(terms)
    return tuple(result)


def _partitions(total, largest=None):
    """Tuples (m3,...,m2k+2) with sum (n-2)*mn = total."""
    if largest is None:
        largest = total
    if largest == 0:
        if total == 0:
            yield ()
        return
    for count in range(total//largest+1):
        for rest in _partitions(total-count*largest, largest-1):
            yield rest + (count,)


def image_series_coefficients(model, strength, radius, inverse_hessian, order=6):
    """Return B0,...,BK for image factor sum (i/w)^k Bk.

    Bk = sum_{sum(n-2)m_n=2k} (-1)^M/<product m_n!>
         times the Wick average of product P_n**m_n, M=sum m_n.
    This formula follows by x-x_j=z/sqrt(w), Taylor expansion of the
    nonquadratic phase, and Gaussian integration. It includes all diagrams.
    """
    if not 0 <= order <= 6:
        raise ValueError('order must be from zero to six')
    if order == 0:
        return (1,)
    terms = _terms(model, strength, radius, 2*order+2)
    g11, g12, g22 = inverse_hessian[0, 0], inverse_hessian[0, 1], inverse_hessian[1, 1]

    @lru_cache(None)
    def moment(i, j):
        if i+j == 0:
            return 1
        if (i+j) % 2:
            return 0
        if i:
            return (i-1)*g11*moment(i-2, j) + j*g12*moment(i-1, j-1)
        return (j-1)*g22*moment(0, j-2)

    powers = [[{(0, 0): 1}] for _ in terms]
    for index, polynomial in enumerate(terms):
        for _ in range((2*order)//(index+1)):
            powers[index].append(_multiply(powers[index][-1], polynomial))
    coefficients = [1]
    for k in range(1, order+1):
        total = 0
        for counts in _partitions(2*k):
            polynomial = {(0, 0): 1}
            multiplier = Fraction((-1)**sum(counts), 1)
            for index, count in enumerate(counts):
                if count:
                    polynomial = _multiply(polynomial, powers[index][count])
                    multiplier /= factorial(count)
            total += multiplier.numerator*sum(c*moment(*ij) for ij, c in polynomial.items())/multiplier.denominator
        coefficients.append(total)
    return tuple(coefficients)


def image_coefficients(model, strength, radius, inverse_hessian):
    """Legacy convention: image factor is 1+i*c1/w+c2/w²."""
    _, b1, b2 = image_series_coefficients(model, strength, radius, inverse_hessian, 2)
    return b1, -b2


def _images(model, source, strength, shear):
    """Find all ordinary images from the quartic radial lens equation."""
    gamma1, gamma2 = shear
    gamma = np.hypot(gamma1, gamma2)
    angle = np.arctan2(gamma2, gamma1)/2 if gamma else 0.0
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    y = rotation.T @ np.asarray(source, dtype=float)
    lam = np.array([1-gamma, 1+gamma])
    t = np.poly1d([1.0, 0.0])
    d0, d1 = float(lam[0])-t, float(lam[1])-t
    radial = t if model == 'PM' else t*t
    power = 1 if model == 'PM' else 2
    polynomial = radial*(float(y[0]**2)*d1*d1+float(y[1]**2)*d0*d0) - strength**power*d0*d0*d1*d1
    candidates = []
    for root in polynomial.r:
        if abs(root.imag) > 1e-8*max(1, abs(root.real)) or root.real <= 0:
            continue
        value = float(root.real)
        if np.min(abs(lam-value)) < 1e-8:
            continue
        candidates.append(y/(lam-value))
    if gamma:
        for i in range(2):
            if abs(y[i]) > 1e-10 or lam[i] <= 0:
                continue
            j = 1-i
            value = lam[i]
            xj = y[j]/(lam[j]-value)
            radius = sqrt(strength/value) if model == 'PM' else strength/value
            xi2 = radius**2-xj**2
            if xi2 >= -1e-10:
                for sign in (-1, 1):
                    x = np.zeros(2)
                    x[i], x[j] = sign*sqrt(max(0, xi2)), xj
                    candidates.append(x)
    images = []
    for local in candidates:
        x = rotation @ local
        r = np.linalg.norm(x)
        factor = strength/r**2 if model == 'PM' else strength/r
        matrix = np.array([[1-gamma1, -gamma2], [-gamma2, 1+gamma1]])
        residual = matrix @ x - factor*x - source
        if np.linalg.norm(residual) > 1e-6 or any(np.linalg.norm(x-z) < 1e-7 for z in images):
            continue
        images.append(x)
    return images


def _image_data(model, source, strength, shear, order=2):
    gamma1, gamma2 = shear
    A = np.array([[1-gamma1, -gamma2], [-gamma2, 1+gamma1]])
    data = []
    for x in _images(model, source, strength, shear):
        r = np.linalg.norm(x)
        er = x/r
        et = np.array([-er[1], er[0]])
        basis = np.column_stack((er, et))
        if model == 'PM':
            hessian = A - strength/r**2*np.eye(2) + 2*strength/r**2*np.outer(er, er)
            lens_potential = strength*np.log(r)
        else:
            hessian = A - strength/r*(np.eye(2)-np.outer(er, er))
            lens_potential = strength*r
        eigenvalues = np.linalg.eigvalsh(hessian)
        determinant = np.linalg.det(hessian)
        tau = np.dot(x-source, x-source)/2 - lens_potential - np.dot(x, (np.eye(2)-A) @ x)/2
        inverse_local = basis.T @ np.linalg.inv(hessian) @ basis
        coefficients = image_series_coefficients(model, strength, r, inverse_local, order)
        data.append(dict(x=x, tau=tau, determinant=determinant,
                         morse=int(np.count_nonzero(eigenvalues < 0)),
                         coefficients=coefficients,
                         first=coefficients[1] if order >= 1 else None,
                         second=-coefficients[2] if order >= 2 else None))
    data.sort(key=lambda item: item['tau'])
    return data


def _differentiate(terms, coordinate):
    """Differentiate y1^i y2^j (b²-y²-i0)^(-p) symbolically."""
    result = {}
    for (i, j, p), coefficient in terms.items():
        powers = [i, j]
        if powers[coordinate]:
            new = powers.copy()
            new[coordinate] -= 1
            key = (new[0], new[1], p)
            result[key] = result.get(key, 0) + coefficient*powers[coordinate]
        new = powers.copy()
        new[coordinate] += 1
        key = (new[0], new[1], p+1)
        result[key] = result.get(key, 0) + 2*p*coefficient
    return result


def _center_coefficients(source, strength, shear, order):
    """Coefficients of w^-1,...,w^-K from the SIS conical center.

    D=A11*d11+2*A12*d12+A22*d22. The m-th coefficient is
    (-i/2)^m D^m[i*b*(b²-y²-i0)^(-3/2)]/m!.
    """
    y = np.asarray(source, dtype=float)
    gamma1, gamma2 = shear
    A = np.array([[1-gamma1, -gamma2], [-gamma2, 1+gamma1]])
    terms = {(0, 0, Fraction(3, 2)): 1}
    z = complex(strength**2-y@y, -1e-14)
    coefficients = []
    for m in range(order):
        value = sum(c*y[0]**i*y[1]**j*z**(-float(p)) for (i, j, p), c in terms.items())
        coefficients.append(1j*strength*(-0.5j)**m/factorial(m)*value)
        next_terms = {}
        for i, j, factor in ((0, 0, A[0, 0]), (0, 1, 2*A[0, 1]), (1, 1, A[1, 1])):
            differentiated = _differentiate(_differentiate(terms, i), j)
            for key, value in differentiated.items():
                next_terms[key] = next_terms.get(key, 0) + factor*value
        terms = next_terms
    return coefficients


def amplification(w, model, source, strength=1.0, shear=(0.0, 0.0), order=6,
                  include_sis_center=True):
    """Amplification through order zero to six beyond geometric optics.

    Frequencies w>0. Ordinary-image terms are invalid at caustics; the SIS
    center series is invalid at |y|=b. See REPORT.md for uniform limits.
    """
    source = np.asarray(source, dtype=float)
    w = np.asarray(w, dtype=float)
    if np.any(w <= 0):
        raise ValueError('w must be positive')
    data = _image_data(model, source, strength, shear, order)
    if not data:
        raise ValueError('No ordinary images found')
    answer = np.zeros_like(w, dtype=complex)
    reference = data[0]['tau']
    for image in data:
        factor = sum((1j/w)**k*c for k, c in enumerate(image['coefficients']))
        answer += (factor/np.sqrt(abs(image['determinant']))
                   * np.exp(1j*w*(image['tau']-reference)-0.5j*pi*image['morse']))
    if model == 'SIS' and order >= 1 and include_sis_center:
        center = sum(c/w**(m+1) for m, c in enumerate(
            _center_coefficients(source, strength, shear, order)))
        answer += np.exp(1j*w*(source@source/2-reference))*center
    return answer
