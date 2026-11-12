# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Strain mapping by dark-field inline holography (DIH).

In DIH (Koch, Oezdoel & van Aken 2010; Oezdoel et al. 2010, 2012; Park et al. 2022) a focal
series of dark-field images is recorded with a single Bragg reflection g aligned with the
optic axis and selected by a small objective aperture. The phase of the reconstructed
diffracted wave contains the geometric phase

    phi_g(r) = -2 pi g . u(r)    (+ sigma Delta V_MIP t  and dynamical contributions)

so that, with grad phi_g computed directly from the complex wave (no phase unwrapping),

    eps_gg = -(g_hat . grad phi_g) / (2 pi |g|)

and, with two non-collinear reflections g1, g2, the full 2D displacement gradient
d u_i / d x_j = -(1/2 pi) [G^-1 (grad phi)]_ij  (rows of G are g1, g2), whose symmetric part is
the strain tensor and antisymmetric part the rigid rotation (Hytch et al. 1998, geometric
phase analysis).
"""
from __future__ import annotations

import numpy as np

from .forward import fft2, ifft2
from .optics import FourierGrid


def phase_gradient(wave, pixel_nm, sigma_px=0.0):
    """grad(arg psi) in rad/nm from the complex wave (wrap free); optional Gaussian smoothing."""
    g = FourierGrid(wave.shape, pixel_nm)
    F = fft2(wave)
    if sigma_px:
        F = F * np.exp(-2 * np.pi ** 2 * (sigma_px * pixel_nm) ** 2 * g.q2)
        w = ifft2(F)
    else:
        w = wave
    dx = ifft2(2j * np.pi * g.qx * F)
    dy = ifft2(2j * np.pi * g.qy * F)
    a2 = np.maximum(np.abs(w) ** 2, 1e-6 * np.max(np.abs(w)) ** 2)
    return np.imag(np.conj(w) * dy) / a2, np.imag(np.conj(w) * dx) / a2   # (d/dy, d/dx)


def strain_1d(wave_g, g_vec, pixel_nm, sigma_px=0.0, ref_mask=None, mip_phase=None):
    """Normal strain along g (eps_gg) and the shear-like term along g_perp from one reflection.

    g_vec : (g_y, g_x) in 1/nm.  ref_mask: boolean region of unstrained reference crystal,
    whose mean strain is subtracted.  mip_phase: optional phase map due to the mean inner
    potential (sigma Delta V t) to be removed first.
    Returns (eps_gg, deps_perp) where deps_perp = d u_g / d x_perp.
    """
    w = wave_g if mip_phase is None else wave_g * np.exp(-1j * mip_phase)
    gy, gx = g_vec
    gn = np.hypot(gx, gy)
    ey, ex = gy / gn, gx / gn
    dpy, dpx = phase_gradient(w, pixel_nm, sigma_px)
    eps = -(ex * dpx + ey * dpy) / (2 * np.pi * gn)
    perp = -(-ey * dpx + ex * dpy) / (2 * np.pi * gn)
    if ref_mask is not None:
        eps = eps - np.mean(eps[ref_mask])
        perp = perp - np.mean(perp[ref_mask])
    return eps, perp


def strain_2d(wave_g1, wave_g2, g1, g2, pixel_nm, sigma_px=0.0, ref_mask=None):
    """Full 2D strain tensor (eps_xx, eps_yy, eps_xy) and rotation from two reflections."""
    p1y, p1x = phase_gradient(wave_g1, pixel_nm, sigma_px)
    p2y, p2x = phase_gradient(wave_g2, pixel_nm, sigma_px)
    G = np.array([[g1[1], g1[0]], [g2[1], g2[0]]])          # rows: (g_x, g_y)
    Ginv = np.linalg.inv(G)
    # grad phi_k = -2 pi sum_i g_k,i d u_i/d x_j  ->  D = -(1/2pi) G^-1 P
    P = np.array([[p1x, p1y], [p2x, p2y]])                   # P[k, j]
    D = -np.einsum("ik,kj...->ij...", Ginv, P) / (2 * np.pi)  # D[i, j] = d u_i / d x_j
    exx, eyy = D[0, 0], D[1, 1]
    exy = 0.5 * (D[0, 1] + D[1, 0])
    rot = 0.5 * (D[1, 0] - D[0, 1])
    if ref_mask is not None:
        exx, eyy, exy, rot = (a - np.mean(a[ref_mask]) for a in (exx, eyy, exy, rot))
    return exx, eyy, exy, rot
