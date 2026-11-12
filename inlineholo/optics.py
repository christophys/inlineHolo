# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Electron-optical basics: wavelength, interaction constant, aberration function,
partial-coherence envelopes.

Conventions
-----------
* Fourier transform: ``Psi(q) = sum_r psi(r) exp(-2 pi i q.r)`` (numpy convention),
  spatial frequencies ``q`` in 1/nm, lengths in nm, angles in rad.
* Aberrations follow the Krivanek notation C_{n,m} (identical to the NXem
  ``NXcorrector_cs/tableau`` names ``c_n_m_a`` / ``c_n_m_b``)::

      chi(theta, phi) = 2 pi / lambda * sum_{n,m} theta^(n+1)/(n+1)
                        * (C_nm_a cos(m phi) + C_nm_b sin(m phi)),   theta = lambda |q|

  so that ``C1 = C_1_0`` is the defocus (positive = overfocus, CEOS/Krivanek
  convention; Scherzer focus is negative for positive C3), ``A1 = (C_1_2_a, C_1_2_b)``
  is two-fold astigmatism, ``C3 = C_3_0`` is spherical aberration, ...
* The wave in a plane of defocus ``C1`` is ``psi_C1 = F^-1[ Psi * exp(-i chi) ]``,
  which for pure defocus is Fresnel propagation over the distance ``z = C1``.
* Partial spatial coherence: Gaussian distribution of illumination tilts with
  standard deviation ``alpha`` (rad) *per axis*.  (A Gaussian with 1/e half-width
  alpha_e corresponds to alpha = alpha_e / sqrt(2).)
* Partial temporal coherence: Gaussian distribution of defocus with standard
  deviation ``focal_spread`` (nm).
"""
from __future__ import annotations

import numpy as np

# physical constants (CODATA 2018)
_h = 6.62607015e-34
_c = 299792458.0
_m0 = 9.1093837015e-31
_e = 1.602176634e-19


def wavelength_nm(voltage_V: float) -> float:
    """Relativistic electron wavelength in nm for acceleration voltage in V."""
    E = _e * voltage_V
    lam = _h / np.sqrt(2 * _m0 * E * (1 + E / (2 * _m0 * _c ** 2)))
    return lam * 1e9


def interaction_constant(voltage_V: float) -> float:
    """Interaction constant sigma = C_E in rad / (V nm)."""
    E = _e * voltage_V
    lam = wavelength_nm(voltage_V) * 1e-9
    gamma_factor = (_m0 * _c ** 2 + E) / (2 * _m0 * _c ** 2 + E)
    sigma = 2 * np.pi / (lam * voltage_V) * gamma_factor  # rad / (V m)
    return sigma * 1e-9


# names of supported aberration coefficients (Krivanek notation)
ABERRATION_NAMES = (
    "c_1_0", "c_1_2_a", "c_1_2_b",
    "c_2_1_a", "c_2_1_b", "c_2_3_a", "c_2_3_b",
    "c_3_0", "c_3_2_a", "c_3_2_b", "c_3_4_a", "c_3_4_b",
    "c_4_1_a", "c_4_1_b", "c_4_3_a", "c_4_3_b", "c_4_5_a", "c_4_5_b",
    "c_5_0", "c_5_2_a", "c_5_2_b", "c_5_4_a", "c_5_4_b", "c_5_6_a", "c_5_6_b",
)


def _parse(name):
    parts = name.split("_")
    n, m = int(parts[1]), int(parts[2])
    ab = parts[3] if len(parts) > 3 else "a"
    return n, m, ab


def astig_to_ab(mag: float, angle_rad: float):
    """Convert two-fold astigmatism (magnitude, azimuth of the axis) to (C12a, C12b).

    ``chi_A1 = pi lambda |A1| q^2 cos(2(phi - angle))``.
    """
    return mag * np.cos(2 * angle_rad), mag * np.sin(2 * angle_rad)


def ab_to_astig(a: float, b: float):
    return float(np.hypot(a, b)), float(0.5 * np.arctan2(b, a))


def chi_polar(theta, phi, aberrations: dict, lam: float):
    """Aberration function chi (rad) for polar angle theta (rad) and azimuth phi."""
    chi = np.zeros_like(theta, dtype=float)
    for name, val in aberrations.items():
        if not val:
            continue
        n, m, ab = _parse(name)
        ang = np.cos(m * phi) if ab == "a" else np.sin(m * phi)
        if m == 0:
            ang = 1.0
        chi = chi + val * theta ** (n + 1) / (n + 1) * ang
    return 2 * np.pi / lam * chi


def chi_gradient_theta(theta, phi, aberrations: dict, lam: float):
    """Gradient of chi with respect to the 2D angle vector (rad/rad); returns (gx, gy)."""
    A = np.zeros_like(theta, dtype=float)   # d chi / d theta
    B = np.zeros_like(theta, dtype=float)   # (1/theta) d chi / d phi
    for name, val in aberrations.items():
        if not val:
            continue
        n, m, ab = _parse(name)
        if m == 0:
            A = A + val * theta ** n
            continue
        if ab == "a":
            A = A + val * theta ** n * np.cos(m * phi)
            B = B - val * theta ** n / (n + 1) * m * np.sin(m * phi)
        else:
            A = A + val * theta ** n * np.sin(m * phi)
            B = B + val * theta ** n / (n + 1) * m * np.cos(m * phi)
    k = 2 * np.pi / lam
    gx = k * (np.cos(phi) * A - np.sin(phi) * B)
    gy = k * (np.sin(phi) * A + np.cos(phi) * B)
    return gx, gy


class FourierGrid:
    """Spatial-frequency grid for an (ny, nx) image with given pixel size (nm)."""

    def __init__(self, shape, pixel_size_nm: float):
        self.shape = tuple(int(s) for s in shape)
        self.pixel = float(pixel_size_nm)
        ny, nx = self.shape
        self.qy = np.fft.fftfreq(ny, d=self.pixel)[:, None] * np.ones((1, nx))
        self.qx = np.fft.fftfreq(nx, d=self.pixel)[None, :] * np.ones((ny, 1))
        self.q2 = self.qx ** 2 + self.qy ** 2
        self.q = np.sqrt(self.q2)
        self.phi = np.arctan2(self.qy, self.qx)
        self.q_nyquist = 0.5 / self.pixel

    def chi(self, aberrations, lam, tilt=(0.0, 0.0)):
        """chi(q + k0) - chi(k0) for an illumination tilt k0 = tilt/lambda (rad)."""
        tx, ty = tilt
        thx = lam * self.qx + tx
        thy = lam * self.qy + ty
        th = np.hypot(thx, thy)
        ph = np.arctan2(thy, thx)
        c = chi_polar(th, ph, aberrations, lam)
        if tx or ty:
            c = c - chi_polar(np.array(np.hypot(tx, ty)), np.array(np.arctan2(ty, tx)),
                              aberrations, lam)
        return c

    def grad_chi(self, aberrations, lam):
        return chi_gradient_theta(lam * self.q, self.phi, aberrations, lam)

    def temporal_envelope(self, lam, focal_spread_nm):
        if not focal_spread_nm:
            return np.ones(self.shape)
        return np.exp(-0.5 * (np.pi * lam * focal_spread_nm * self.q2) ** 2)

    def spatial_envelope(self, aberrations, lam, alpha_rad):
        if not alpha_rad:
            return np.ones(self.shape)
        gx, gy = self.grad_chi(aberrations, lam)
        return np.exp(-0.5 * alpha_rad ** 2 * (gx ** 2 + gy ** 2))

    def aperture(self, lam, aperture_rad=None, soft_px=1.0):
        if aperture_rad is None:
            return np.ones(self.shape)
        qa = aperture_rad / lam
        dq = soft_px / (self.shape[0] * self.pixel)
        return 0.5 * (1 - np.tanh((self.q - qa) / dq))


def gauss_hermite_1d(n: int):
    """Nodes/weights for averaging f(x) over a standard normal distribution."""
    if n <= 1:
        return np.array([0.0]), np.array([1.0])
    x, w = np.polynomial.hermite.hermgauss(n)
    return np.sqrt(2) * x, w / np.sqrt(np.pi)


def tilt_nodes(alpha_rad: float, n: int):
    """2D Gauss-Hermite product nodes for a Gaussian tilt distribution (std alpha per axis)."""
    if not alpha_rad or n <= 1:
        return np.zeros((1, 2)), np.ones(1)
    x, w = gauss_hermite_1d(n)
    X, Y = np.meshgrid(x, x)
    W = np.outer(w, w)
    nodes = alpha_rad * np.stack([X.ravel(), Y.ravel()], axis=1)
    return nodes, W.ravel()
