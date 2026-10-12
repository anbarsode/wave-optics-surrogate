"""
Based on
    Takahashi & Nakamura (2003), Wave Effects in Gravitational Lensing of Gravitational Waves from Chirping Binaries.
Written by chatgpt.
Checked that result matches visually with ... by Ankur on ...
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma, hyp1f1
from mpmath import hyp1f1 as hyp1f1_single
hyp1f1 = np.vectorize(
    lambda a, b, z: complex(hyp1f1_single(a, b, z)),
    otypes=[np.complex128]
)
from astropy.constants import M_sun, G, c

@dataclass(frozen=True)
class Image:
    """
    A single lensed image.
    """
    x: float               # image position
    phi: float             # dimensionless Fermat potential
    delay: float           # physical arrival time (s)
    mu: float              # signed magnification
    morse: int             # 0 for "minimum" or 1 for "saddle"


class PointLens:
    """
    Point-mass gravitational lens.

    Parameters
    ----------
    M_Lz : float
        Redshifted lens mass in solar masses.
    """

    def __init__(self, M_Lz):
        self.M_Lz = float(M_Lz)
        self.M = self.M_Lz * M_sun.value

    @property
    def t0(self):
        """
        Characteristic lens time

            t0 = 4GM/c^3
        """
        return 4 * G.value * self.M / c.value**3

    def f_to_w(self, f):
        """
        Convert frequency (Hz) to dimensionless frequency.
        """
        return 2 * np.pi * np.asarray(f) * self.t0
    
    def w_to_f(self, w):
        """
        Convert dimensionless frequency to frequency (Hz).
        """
        return np.asarray(w) / (2 * np.pi * self.t0)

    # -----------------------------------------------------
    # Lens equation
    # -----------------------------------------------------

    @staticmethod
    def image_positions(y):
        y = np.asarray(y)

        r = np.sqrt(y**2 + 4)

        return (
            0.5 * (y + r),
            0.5 * (y - r),
        )

    @staticmethod
    def fermat_potential(x, y):
        return 0.5 * (x - y) ** 2 - np.log(np.abs(x))

    @staticmethod
    def magnifications(y):
        r = np.sqrt(y**2 + 4)

        mu_plus = 0.5 + (y**2 + 2) / (2 * y * r)
        mu_minus = 0.5 - (y**2 + 2) / (2 * y * r)

        return mu_plus, mu_minus

    # -----------------------------------------------------
    # Images
    # -----------------------------------------------------

    def images(self, y):
        """
        Returns
        -------
        (minimum_image, saddle_image)
        """

        xp, xm = self.image_positions(y)

        mup, mum = self.magnifications(y)

        phip = self.fermat_potential(xp, y)
        phim = self.fermat_potential(xm, y)

        return (
            Image(
                x=xp,
                phi=phip,
                delay=self.t0 * phip,
                mu=mup,
                morse=0,
            ),
            Image(
                x=xm,
                phi=phim,
                delay=self.t0 * phim,
                mu=mum,
                morse=1,
            ),
        )

    # -----------------------------------------------------
    # Convenience methods
    # -----------------------------------------------------

    def minimum_image(self, y):
        return self.images(y)[0]

    def saddle_image(self, y):
        return self.images(y)[1]

    def time_delay(self, y):
        """
        Dimensionless Fermat potential.
        """
        p, m = self.images(y)
        return m.phi - p.phi

    def time_delay_seconds(self, y):
        return self.t0 * self.time_delay(y)

    def total_magnification(self, y):
        p, m = self.images(y)
        return abs(p.mu) + abs(m.mu)

    # -----------------------------------------------------
    # Wave optics
    # -----------------------------------------------------

    def amplification(self, f, y):
        """
        Exact wave-optics amplification factor.
        """

        w = self.f_to_w(f)

        phi = self.minimum_image(y).phi

        prefactor = (
            np.exp(np.pi * w / 4)
            * np.exp(
                0.5j * w
                * (np.log(w / 2) - 2 * phi)
            )
            * gamma(1 - 0.5j * w)
        )

        hyper = hyp1f1(
            0.5j * w,
            1,
            0.5j * w * y**2,
        )

        return prefactor * hyper

    # -----------------------------------------------------
    # Geometric optics
    # -----------------------------------------------------

    def geometric_optics_amplification(self, f, y):
        """
        High-frequency approximation.
        """

        p, s = self.images(y)
        dt = s.delay - p.delay
        phase_s = np.exp(
            2j * np.pi * f * dt
            - 0.5j * np.pi
        )

        return (
            np.sqrt(abs(p.mu))
            + np.sqrt(abs(s.mu)) * phase_s
        )
