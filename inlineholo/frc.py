# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Fourier ring correlation (FRC) and related quality metrics.

FRC(q) = Re sum_{ring q} F1 F2* / sqrt( sum |F1|^2 sum |F2|^2 )
(Saxton & Baumeister 1982; van Heel 1987).  For complex exit waves the full complex
spectra are correlated, so amplitude and phase errors are both penalized.  Threshold
curves after van Heel & Schatz (2005); for a comparison against a noise-free ground
truth the 0.5 criterion is used (equivalent to the 1/7 criterion for half data sets).
"""
from __future__ import annotations

import numpy as np

from .forward import fft2


def ring_indices(shape, pixel_size, n_bins=None):
    ny, nx = shape
    qy = np.fft.fftfreq(ny, d=pixel_size)[:, None]
    qx = np.fft.fftfreq(nx, d=pixel_size)[None, :]
    q = np.sqrt(qx ** 2 + qy ** 2)
    qmax = 0.5 / pixel_size
    if n_bins is None:
        n_bins = min(ny, nx) // 2
    edges = np.linspace(0, qmax, n_bins + 1)
    idx = np.digitize(q, edges) - 1
    idx[(idx < 0) | (idx >= n_bins)] = -1
    centers = 0.5 * (edges[1:] + edges[:-1])
    return idx, centers


def tukey_window(shape, alpha=0.2):
    from scipy.signal.windows import tukey
    return np.outer(tukey(shape[0], alpha), tukey(shape[1], alpha))


def align_global_phase(a, b, subtract_mean=False):
    """Multiply a by the global phase factor that best matches b (least squares)."""
    c = np.vdot(a, b)
    return a * (c / abs(c) if abs(c) > 0 else 1.0)


def frc(a, b, pixel_size=1.0, n_bins=None, window=None, remove_dc=True):
    """Return (q, frc, n_points_per_ring) for two (complex or real) images."""
    a = np.asarray(a)
    b = np.asarray(b)
    if window is not None:
        w = tukey_window(a.shape, window) if np.isscalar(window) else window
        a = (a - a.mean()) * w
        b = (b - b.mean()) * w
    Fa, Fb = fft2(a), fft2(b)
    if remove_dc:
        Fa[0, 0] = Fb[0, 0] = 0
    idx, q = ring_indices(a.shape, pixel_size, n_bins)
    m = idx >= 0
    nb = len(q)
    num = np.bincount(idx[m], weights=np.real(Fa * np.conj(Fb))[m], minlength=nb)
    da = np.bincount(idx[m], weights=np.abs(Fa[m]) ** 2, minlength=nb)
    db = np.bincount(idx[m], weights=np.abs(Fb[m]) ** 2, minlength=nb)
    npts = np.bincount(idx[m], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = num / np.sqrt(da * db)
    return q, np.nan_to_num(f), npts


def half_bit_threshold(npts):
    n = np.maximum(npts, 1)
    return (0.2071 + 1.9102 / np.sqrt(n)) / (1.2071 + 0.9102 / np.sqrt(n))


def one_bit_threshold(npts):
    n = np.maximum(npts, 1)
    return (0.5 + 2.4142 / np.sqrt(n)) / (1.5 + 1.4142 / np.sqrt(n))


def resolution_from_frc(q, f, threshold=0.5, q_min=None, smooth=3, mode="last"):
    """Resolution from an FRC curve.

    mode='first': first frequency (>= q_min) where the FRC drops below the threshold;
    mode='last' : highest frequency at which the (smoothed) FRC is still >= threshold.
    The 'last' variant is robust against the poorly transferred low spatial frequencies
    typical of inline holography.  Returns (q_res, d_res = 1/q_res).
    """
    f = np.asarray(f, float)
    if smooth and smooth > 1:
        k = np.ones(int(smooth)) / int(smooth)
        fs = np.convolve(np.pad(f, (smooth // 2, smooth - 1 - smooth // 2), mode="edge"), k, "valid")
    else:
        fs = f
    thr = np.broadcast_to(threshold, fs.shape)
    start = 1 if q_min is None else max(1, int(np.searchsorted(q, q_min)))
    ok = fs >= thr
    if mode == "last":
        idx = np.where(ok[start:])[0]
        if len(idx) == 0:
            return np.nan, np.nan
        i = start + idx[-1]
        if i >= len(q) - 1:
            return q[-1], 1 / q[-1]
        f0, f1 = fs[i] - thr[i], fs[i + 1] - thr[i + 1]
    else:
        idx = np.where(~ok[start:])[0]
        if len(idx) == 0:
            return q[-1], 1 / q[-1]
        i = start + idx[0] - 1
        if i < start:
            return np.nan, np.nan
        f0, f1 = fs[i] - thr[i], fs[i + 1] - thr[i + 1]
    t = f0 / (f0 - f1) if f0 != f1 else 0.0
    qr = q[i] + np.clip(t, 0, 1) * (q[i + 1] - q[i])
    return qr, 1 / qr


def compare_to_truth(rec, truth, pixel_size, q_max=None, n_bins=None, threshold=0.5):
    """FRC-based comparison of a reconstructed wave with the ground truth.

    The unobservable global phase is removed first.  Returns a dict with the FRC
    curve, the resolution (highest frequency with FRC >= threshold), the mean FRC up to
    q_max and in three bands (low: <0.1 q_max, mid: 0.1-0.4, high: 0.4-1), the
    amplitude-weighted rms phase error and the normalized complex rms error.
    """
    r = align_global_phase(rec, truth)
    q, f, n = frc(r, truth, pixel_size, n_bins)
    qn = 0.5 / pixel_size
    if q_max is None:
        q_max = qn
    sel = (q > 0) & (q <= q_max) & (n > 1)
    qr, dr = resolution_from_frc(q, f, threshold, mode="last")
    bands = {"frc_low": (0, 0.1), "frc_mid": (0.1, 0.4), "frc_high": (0.4, 1.0)}
    out = dict(q=q, frc=f, npts=n, q_res=qr, d_res=dr, mean_frc=float(np.mean(f[sel])))
    valid = n > 1                      # ring 0 contains only the (removed) DC component
    for k, (a, b) in bands.items():
        m = (q > a * q_max) & (q <= b * q_max) & valid
        out[k] = float(np.mean(f[m])) if m.any() else np.nan
    dphi = np.angle(r * np.conj(truth))
    w = np.abs(truth) ** 2
    out["rms_phase"] = float(np.sqrt(np.sum(w * dphi ** 2) / np.sum(w)))
    s = np.vdot(r, truth) / np.vdot(r, r)
    out["nrmse"] = float(np.linalg.norm(s * r - truth) / np.linalg.norm(truth - truth.mean()))
    return out


def binomial_split(counts, rng=None):
    """Split a stack of counts into two statistically independent halves (Poisson thinning)."""
    rng = np.random.default_rng(rng)
    c = np.asarray(np.round(counts), dtype=np.int64)
    a = rng.binomial(np.maximum(c, 0), 0.5)
    return a, c - a
