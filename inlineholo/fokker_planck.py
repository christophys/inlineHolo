# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Fokker-Planck generalization of the transport-of-intensity equation.

Paganin & Morgan (Sci. Rep. 2019), Paganin, Pelliccia & Morgan (Phys. Rev. A 108, 013517,
2023) and Paganin & Morgan (arXiv:2601.14645, 2026) augment the coherent energy flow of the
TIE by a *diffusive* flow.  To second order in the propagation distance (defocus) dz,

    I(dz) = I0 - (dz/k) div[(1-F) I0 grad(phi)] + dz^2 lap[D I0] + dz^2 (coherent 2nd-order terms)

with the dimensionless diffusion field D(x, y) = (theta_0^2 + F theta_s^2)/2: theta_0 is the
illumination (source-size) semi-angle, theta_s the rms small-angle scattering angle of
spatially unresolved micro-structure and F the scattered fraction.

Consequences for focal-series reconstruction implemented here
------------------------------------------------------------
* The dz^2 terms cancel in symmetric differences I(+dz) - I(-dz); for asymmetric or one-sided
  series they bias a first-order TIE.  :func:`tie_fp` uses the three planes closest to
  focus (any signs) and the quadratic fit (= the data function of Leatham et al. 2023 /
  Paganin & Morgan 2026, Eqs. 99 and 102), which removes them.
* For a uniform D = theta_0^2/2 the diffusion term is the small-dz limit of the Gaussian
  intensity blur with sigma = |dz| theta_0 used by the flux-preserving FRWR model.  A
  *position-dependent* D(x, y) is an additional, dark-field-like information channel that
  can be retrieved from the same focal series: :func:`diffusion_field` fits the residuals
  between the measured series and the coherent (or flux-preserving) images of a
  reconstructed wave with dz^2 lap(D I0) (+ an optional defocus-independent background).
"""
from __future__ import annotations

import numpy as np

from .forward import ForwardModel, fft2, ifft2
from .optics import FourierGrid
from .params import ImagingParameters
from .reconstruct import Reconstruction, _inv_laplace, _grad, _div


def fp_forward(I0, phi, D, dz, lam, pixel, F=0.0):
    """Finite-difference Fokker-Planck TIE forward model (Paganin & Morgan 2026, Eq. 21)."""
    grid = FourierGrid(I0.shape, pixel)
    k = 2 * np.pi / lam
    gx, gy = _grad(phi, grid)
    w = (1.0 - F) * I0
    coh = _div(w * gx, w * gy, grid)
    Fq = fft2(D * I0)
    lap = np.real(ifft2(-4 * np.pi ** 2 * grid.q2 * Fq))
    return I0 - dz / k * coh + dz ** 2 * lap


def _three_nearest(z, max_planes=3):
    return np.argsort(np.abs(z))[:max_planes]


def quadratic_fit(data, z, idx=None):
    """Per-pixel LSQ fit I(z) = a + b z + c z^2 (all planes in idx)."""
    z = np.asarray(z, float)
    idx = np.arange(len(z)) if idx is None else np.asarray(idx)
    V = np.vander(z[idx], 3, increasing=True)
    coef = np.linalg.pinv(V) @ np.asarray(data)[idx].reshape(len(idx), -1)
    shp = np.asarray(data).shape[1:]
    return coef[0].reshape(shp), coef[1].reshape(shp), coef[2].reshape(shp)


def tie_fp(data, params: ImagingParameters, planes="nearest3", q_reg=None, return_diffusion=True,
           Ifloor=0.05, d_reg=None):
    """TIE consistent with the Fokker-Planck extension (dz^2 terms eliminated).

    planes : 'nearest3' (three planes closest to focus, any sign of defocus) or an index list
    Returns a Reconstruction; ``extra['D']`` contains the diffusion field estimated from the
    quadratic coefficient after subtraction of the coherent second-order term of the
    retrieved wave (meaningful for small defocus only).
    """
    z = np.asarray(params.defoci_nm, float)
    idx = _three_nearest(z) if isinstance(planes, str) else np.asarray(planes)
    if len(idx) < 3:
        raise ValueError("tie_fp needs at least three planes")
    a, b, c = quadratic_fit(data, z, idx)
    lam = params.wavelength_nm
    k = 2 * np.pi / lam
    grid = FourierGrid(a.shape, params.pixel_size_nm)
    if q_reg is None:
        q_reg = 0.5 / (max(a.shape) * params.pixel_size_nm)
    aux = -k * _inv_laplace(b, grid, q_reg)
    gx, gy = _grad(aux, grid)
    Ir = np.maximum(a, Ifloor * np.mean(a))
    phi = _inv_laplace(_div(gx / Ir, gy / Ir, grid), grid, q_reg)
    wave = np.sqrt(np.maximum(a, 0)) * np.exp(1j * phi)
    rec = Reconstruction(wave, "tie_fp", dict(planes=idx.tolist(), q_reg=q_reg))
    if return_diffusion:
        # coherent second-order coefficient of the retrieved wave (central difference)
        h = max(np.min(np.abs(z[idx][z[idx] != 0])) if np.any(z[idx] != 0) else 1.0, 1e-3) * 0.25
        p2 = ImagingParameters(voltage_V=params.voltage_V, pixel_size_nm=params.pixel_size_nm,
                               defoci_nm=[-h, 0.0, h])
        I = ForwardModel(p2, a.shape, "coherent", use_mtf=False).forward(wave)
        c_coh = (I[0] + I[2] - 2 * I[1]) / (2 * h ** 2)
        r = c - c_coh
        u = _inv_laplace(r, grid, q_reg if d_reg is None else d_reg)  # lap(u) = r
        D = u / Ir
        rec.extra = {"D": D, "quadratic_coefficient": c, "coherent_quadratic": c_coh}
    return rec


def diffusion_field(data, params: ImagingParameters, wave, model="blur", max_dz=None, reg=1e-8,
                    background=True, q_min=None):
    """Diffusion-field retrieval from a focal series and a reconstructed wave.

    The residuals r_n = I_exp,n - I_model,n(wave) are fitted, independently for every spatial
    frequency, by  r_n(q) = -4 pi^2 q^2 dz_n^2 U(q) + B(q)  with U = D I0 and an optional
    defocus-independent background B (the 'incoherent background' of FRWR).  With
    ``model='blur'`` the source-size blur (alpha) is part of the model, so that D is the
    specimen contribution F theta_s^2 / 2; with ``model='coherent'`` it includes theta_0^2/2.
    Only planes with |dz| <= max_dz are used (the expansion is second order in dz).
    ``reg`` is a Tikhonov floor relative to the mean of sum_n a_n^2 (noise-free: 1e-8; noisy
    data: increase it and/or suppress low frequencies with ``q_min``, since the transfer of
    the diffusion term vanishes as q^2 dz^2).
    Returns dict(D=..., U=..., B=..., planes=...).
    """
    data = np.asarray(data, float)
    z = np.asarray(params.defoci_nm, float)
    idx = np.arange(len(z)) if max_dz is None else np.where(np.abs(z) <= max_dz)[0]
    p = params.subset(idx)
    I = ForwardModel(p, data.shape[1:], model).forward(wave)
    R = fft2(data[idx] - I)
    grid = FourierGrid(data.shape[1:], params.pixel_size_nm)
    a = -4 * np.pi ** 2 * grid.q2[None] * (z[idx] ** 2)[:, None, None]       # (n, ny, nx)
    if background:
        n = len(idx)
        S11 = np.sum(a * a, 0)
        S12 = np.sum(a, 0)
        S22 = float(n)
        T1 = np.sum(a * R, 0)
        T2 = np.sum(R, 0)
        lam_reg = reg * np.mean(S11)
        det = (S11 + lam_reg) * S22 - S12 ** 2
        U = (S22 * T1 - S12 * T2) / det
        Bq = ((S11 + lam_reg) * T2 - S12 * T1) / det
    else:
        S11 = np.sum(a * a, 0)
        U = np.sum(a * R, 0) / (S11 + reg * np.mean(S11))
        Bq = np.zeros_like(U)
    if q_min is not None:
        U = U * (grid.q >= q_min)
    U[0, 0] = 0
    u = np.real(ifft2(U))
    B = np.real(ifft2(Bq))
    I0 = np.abs(wave) ** 2
    D = u / np.maximum(I0, 0.05 * np.mean(I0))
    return dict(D=D, U=u, B=B, planes=idx.tolist())
