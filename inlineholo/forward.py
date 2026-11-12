# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Forward models for focal-series image formation and their adjoints (gradients).

Four levels of approximation are implemented (thin-object / projection approximation):

``coherent``
    I_n = |F^-1[ Psi T_n ]|^2 with T_n = A exp(-i chi_n).
``envelope`` (quasi-coherent)
    as ``coherent`` but T_n multiplied by the temporal and spatial coherence envelopes
    E_t(q) E_s,n(q).  This is the classical "envelope on the wave" approximation, exact
    for the linear-imaging (single-sideband) terms and approximate for the non-linear
    (wave-wave interference) terms.
``blur`` (flux preserving, FRWR-type)
    I_n = G_n (x) |F^-1[ Psi T_n E_t ]|^2 + I_bg, where partial spatial coherence is
    treated as a defocus-dependent Gaussian blur of the *intensity* (std
    sigma_n = |C1_n| alpha).  This preserves the total flux (Koch 2008).
``incoherent``
    rigorous incoherent superposition of coherent images over a Gauss-Hermite
    quadrature of defocus values (temporal coherence) and illumination tilts
    (spatial coherence):  I_n = sum_j w_j |F^-1[ Psi T_{n,j} ]|^2.  For thin objects
    this is equivalent to imaging with the full transmission cross-coefficient (TCC).

All models optionally include a detector MTF and an objective aperture.
"""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from .optics import FourierGrid, gauss_hermite_1d, tilt_nodes
from .params import ImagingParameters

WORKERS = -1


def fft2(a):
    return sfft.fft2(a, workers=WORKERS)


def ifft2(a):
    return sfft.ifft2(a, workers=WORKERS)


def detector_mtf(grid: FourierGrid, mtf: dict | None):
    """Detector MTF in the Fourier domain of the image (applied to intensities)."""
    if not mtf:
        return None
    kind = mtf.get("type", "sinc")
    p = grid.pixel
    if kind == "sinc":
        return np.sinc(grid.qx * p) * np.sinc(grid.qy * p)
    if kind == "gauss":
        s = mtf.get("sigma_px", 0.5) * p
        return np.exp(-2 * np.pi ** 2 * s ** 2 * grid.q2)
    if kind == "array":
        return np.asarray(mtf["values"])
    raise ValueError(kind)


class ForwardModel:
    KINDS = ("coherent", "envelope", "blur", "incoherent")

    def __init__(self, params: ImagingParameters, shape, kind: str = "envelope",
                 n_defocus: int = 5, n_tilt: int = 3, use_mtf: bool = True):
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}")
        self.p = params
        self.kind = kind
        self.grid = g = FourierGrid(shape, params.pixel_size_nm)
        lam = params.wavelength_nm
        ap = g.aperture(lam, params.aperture_rad)
        Et = g.temporal_envelope(lam, params.focal_spread_nm)
        self.mtf = detector_mtf(g, params.mtf) if use_mtf else None
        self.nodes = []      # per image: list of (T, weight)
        self.blur = []       # per image: Fourier-space intensity kernel or None
        for i in range(params.n_images):
            ab = params.aberrations_for(i)
            if kind == "coherent":
                T = ap * np.exp(-1j * g.chi(ab, lam))
                self.nodes.append([(T, 1.0)])
                self.blur.append(self.mtf)
            elif kind == "envelope":
                Es = g.spatial_envelope(ab, lam, params.convergence_rad)
                T = ap * Et * Es * np.exp(-1j * g.chi(ab, lam))
                self.nodes.append([(T, 1.0)])
                self.blur.append(self.mtf)
            elif kind == "blur":
                T = ap * Et * np.exp(-1j * g.chi(ab, lam))
                sig = abs(ab["c_1_0"]) * params.convergence_rad
                K = np.exp(-2 * np.pi ** 2 * sig ** 2 * g.q2)
                if self.mtf is not None:
                    K = K * self.mtf
                self.nodes.append([(T, 1.0)])
                self.blur.append(K)
            else:  # incoherent
                dz, wz = gauss_hermite_1d(n_defocus if params.focal_spread_nm else 1)
                tn, wt = tilt_nodes(params.convergence_rad, n_tilt)
                lst = []
                for z, w1 in zip(dz * params.focal_spread_nm, wz):
                    abz = dict(ab)
                    abz["c_1_0"] = ab["c_1_0"] + z
                    for t, w2 in zip(tn, wt):
                        T = ap * np.exp(-1j * g.chi(abz, lam, tilt=tuple(t)))
                        lst.append((T, w1 * w2))
                self.nodes.append(lst)
                self.blur.append(self.mtf)

    @property
    def n_ffts_per_eval(self):
        return sum(len(n) for n in self.nodes)

    # ------------------------------------------------------------------ forward
    def propagate(self, psi, i, node=0):
        """Coherent wave in image plane i (first quadrature node)."""
        T, _ = self.nodes[i][node]
        return ifft2(fft2(psi) * T)

    def forward(self, psi, Psi=None, keep=False):
        if Psi is None:
            Psi = fft2(psi)
        out = np.empty((self.p.n_images,) + self.grid.shape)
        fields = [] if keep else None
        for i, lst in enumerate(self.nodes):
            I = np.zeros(self.grid.shape)
            fl = []
            for T, w in lst:
                f = ifft2(Psi * T)
                I += w * (f.real ** 2 + f.imag ** 2)
                if keep:
                    fl.append(f)
            K = self.blur[i]
            if K is not None:
                I = np.real(ifft2(fft2(I) * K))
            out[i] = I
            if keep:
                fields.append(fl)
        return (out, fields) if keep else out

    # ------------------------------------------------------------- loss + grad
    def loss_and_grad(self, psi, data, loss="lsq", weights=None, background=None, eps=1e-6,
                      mask=None):
        """Return (L, g) with g = dL/dRe(psi) + i dL/dIm(psi) (= 2 dL/dpsi*).

        ``data`` is the focal series normalized to unit mean intensity per image.
        mask: optional 2D weight map (e.g. 1 inside the measured field of view, 0 in a
              padding region), multiplies the per-pixel loss.
        loss: 'lsq'      L = 1/2 mean (I - D)^2              (Gaussian noise, MAL/FRWR)
              'amplitude' L = 1/2 mean (sqrt I - sqrt D)^2   (variance-stabilized)
              'poisson'  L = mean (I - D log I)              (Poisson maximum likelihood)
        """
        Psi = fft2(psi)
        I, fields = self.forward(psi, Psi, keep=True)
        if background is not None:
            I = I + background
        N = data.size if mask is None else float(np.sum(mask)) * len(data)
        wts = np.ones(self.p.n_images) if weights is None else np.asarray(weights)
        G = np.zeros(self.grid.shape, complex)
        Ltot = 0.0
        for i in range(self.p.n_images):
            Ii, Di = I[i], data[i]
            if loss == "lsq":
                r = Ii - Di
                Lmap = 0.5 * r ** 2
                fp = r
            elif loss == "amplitude":
                s = np.sqrt(np.maximum(Ii, 0) + eps)
                r = s - np.sqrt(np.maximum(Di, 0))
                Lmap = 0.5 * r ** 2
                fp = r / (2 * s)
            elif loss == "poisson":
                Ic = np.maximum(Ii, eps)
                Lmap = Ic - Di * np.log(Ic)
                fp = 1 - Di / Ic
            else:
                raise ValueError(loss)
            L = np.sum(Lmap) if mask is None else np.sum(mask * Lmap)
            Ltot += wts[i] * L
            fp = wts[i] * fp / N
            if mask is not None:
                fp = fp * mask
            K = self.blur[i]
            if K is not None:
                fp = np.real(ifft2(fft2(fp) * np.conj(K)))
            for (T, w), f in zip(self.nodes[i], fields[i]):
                G += w * np.conj(T) * fft2(fp * f)
        g = 2 * ifft2(G)
        return Ltot / N, g


def normalize_stack(stack, reference="mean"):
    """Normalize each image to unit mean; returns (normalized, scale per image)."""
    stack = np.asarray(stack, float)
    if reference == "mean":
        s = stack.reshape(len(stack), -1).mean(axis=1)
    else:
        s = np.full(len(stack), float(reference))
    return stack / s[:, None, None], s
