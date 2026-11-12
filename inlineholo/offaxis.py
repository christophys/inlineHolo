# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Simulation and sideband reconstruction of off-axis electron holograms
(used for comparison and as start wave of hybrid inline/off-axis holography)."""
from __future__ import annotations

import numpy as np

from .forward import fft2, ifft2
from .optics import FourierGrid


def carrier_on_grid(shape, pixel_nm, q_c, angle_deg=0.0):
    """Carrier frequency vector (1/nm) rounded to the nearest FFT grid point."""
    ny, nx = shape
    qx = q_c * np.cos(np.deg2rad(angle_deg))
    qy = q_c * np.sin(np.deg2rad(angle_deg))
    mx = np.round(qx * nx * pixel_nm)
    my = np.round(qy * ny * pixel_nm)
    return my / (ny * pixel_nm), mx / (nx * pixel_nm)


def simulate_offaxis(wave, pixel_nm, q_c=0.25, angle_deg=0.0, mu=0.3, dose_per_pixel=None,
                     mtf=True, seed=0):
    """Off-axis hologram I = 1 + |psi|^2 + mu (psi e^{-2 pi i q_c.r} + c.c.).

    ``mu`` is the fringe contrast (partial coherence, instabilities); a sinc detector MTF
    damps the fringes further. ``dose_per_pixel`` is the mean number of electrons per
    pixel (Poisson noise), None = noise free.  The reference wave is an ideal plane wave.
    Returns (hologram, (qy, qx)).
    """
    ny, nx = wave.shape
    qy, qx = carrier_on_grid(wave.shape, pixel_nm, q_c, angle_deg)
    y, x = np.mgrid[0:ny, 0:nx] * pixel_nm
    carrier = np.exp(-2j * np.pi * (qx * x + qy * y))
    I = 1 + np.abs(wave) ** 2 + 2 * mu * np.real(wave * carrier)
    if mtf:
        g = FourierGrid(wave.shape, pixel_nm)
        I = np.real(ifft2(fft2(I) * np.sinc(g.qx * pixel_nm) * np.sinc(g.qy * pixel_nm)))
    if dose_per_pixel is not None:
        rng = np.random.default_rng(seed)
        I = rng.poisson(np.maximum(I, 0) / np.mean(I) * dose_per_pixel).astype(float)
    return I, (qy, qx)


def reconstruct_offaxis(hologram, pixel_nm, carrier, r_ap=None, mu=None, order=8):
    """Sideband reconstruction: Fourier mask of radius ``r_ap`` (1/nm, default |q_c|/3)
    around -q_c, shift to the origin, inverse FFT.  Returns the complex wave normalized to
    the mean intensity of the hologram's reference part."""
    qy, qx = carrier
    g = FourierGrid(hologram.shape, pixel_nm)
    if r_ap is None:
        r_ap = np.hypot(qx, qy) / 3
    H = fft2(hologram / np.mean(hologram))
    ny, nx = hologram.shape
    sy = int(round(-qy * ny * pixel_nm))
    sx = int(round(-qx * nx * pixel_nm))
    side = np.roll(H, shift=(-sy, -sx), axis=(0, 1))       # sideband at -q_c moved to origin
    mask = 1.0 / (1.0 + (g.q / r_ap) ** (2 * order))
    w = ifft2(side * mask)
    # normalize amplitude: mean |w| -> mean |psi| ~ 1 (unknown mu)
    if mu is None:
        w = w / np.mean(np.abs(w))
    else:
        w = w * 2 / mu
    return w
