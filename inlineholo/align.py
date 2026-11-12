# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Image registration of focal series (drift correction)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .forward import fft2, ifft2


def fourier_shift(img, shift):
    """Shift an image by (dy, dx) pixels (periodic, sub-pixel)."""
    ny, nx = img.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    ph = np.exp(-2j * np.pi * (ky * shift[0] + kx * shift[1]))
    out = ifft2(fft2(img) * ph)
    return out.real if np.isrealobj(img) else out


def _xcorr_shift(a, b, upsample=20, sigma_hp=None):
    from skimage.registration import phase_cross_correlation
    if sigma_hp:
        a = a - gaussian_filter(a, sigma_hp)
        b = b - gaussian_filter(b, sigma_hp)
    s, _, _ = phase_cross_correlation(a, b, upsample_factor=upsample, normalization=None)
    return s


def register_stack(stack, defoci=None, reference=None, upsample=20, sigma_hp=None,
                   sequential=True):
    """Align a focal series.

    Images are registered pairwise to their neighbour in defocus order (shifts are
    accumulated) when ``sequential`` - this is more robust because neighbouring
    images look alike.  Returns (aligned_stack, shifts[N, 2]).
    """
    stack = np.asarray(stack, float)
    n = len(stack)
    order = np.argsort(defoci) if defoci is not None else np.arange(n)
    if reference is None:
        reference = int(order[len(order) // 2]) if defoci is None else int(np.argmin(np.abs(defoci)))
    shifts = np.zeros((n, 2))
    pos = list(order).index(reference)
    if sequential:
        for direction in (1, -1):
            k = pos
            while 0 <= k + direction < n:
                i, j = order[k], order[k + direction]
                s = _xcorr_shift(stack[i], stack[j], upsample, sigma_hp)
                shifts[j] = shifts[i] + s
                k += direction
    else:
        for j in range(n):
            if j != reference:
                shifts[j] = _xcorr_shift(stack[reference], stack[j], upsample, sigma_hp)
    out = np.stack([fourier_shift(stack[i], shifts[i]) for i in range(n)])
    return out, shifts


def frwr3_align(stack, params, binning=2, config=None, margin_px=60, max_shift_px=40, fit_distortions=True,
                final_loops=10, crop=True, backend="auto", callback=None):
    """Drift and defocus-dependent distortion correction with FRWR3 (Koch, Micron 63, 2014).

    FRWR3 is run on a ``binning``-times binned copy of the (normalized) series with the fit of
    image rotation, magnification, two-fold (elliptical) magnification, spiral and pincushion
    distortion switched on; the distortions and (sub-pixel) frame offsets are then applied to the
    full-resolution frames, which are cropped to the largest rectangle covered by all frames.

    Returns (aligned stack, info) with info = dict(distortions (n, 6; rotation [rad], zoom,
    zoomX, zoomY, spiral and pincushion [per 1024 px]), shifts_px (n, 2; x, y), crop (top, left,
    bottom, right), refined_defoci_nm, reference_plane, R_final, result (the FRWR3Result)).
    """
    from . import frwr3
    from .params import ImagingParameters
    st = np.asarray(stack, float)
    n, NY, NX = st.shape
    b = max(1, int(binning))
    ny, nx = NY // b * b, NX // b * b
    work = st[:, :ny, :nx].reshape(n, ny // b, b, nx // b, b).mean((2, 4)) if b > 1 else st[:, :ny, :nx].copy()
    work = work / work.mean()
    pb = ImagingParameters(voltage_V=params.voltage_V, pixel_size_nm=params.pixel_size_nm * b,
                           defoci_nm=np.asarray(params.defoci_nm, float), aberrations=dict(params.aberrations),
                           astig_a_nm=params.astig_a_nm, astig_b_nm=params.astig_b_nm,
                           focal_spread_nm=params.focal_spread_nm, convergence_rad=params.convergence_rad,
                           aperture_rad=params.aperture_rad)
    m = max(4, int(round(margin_px / b)))
    roi = (m, m, work.shape[1] - m, work.shape[2] - m)
    cfg = {"print level": 0, "padding": 0.25, "fit distortions": int(bool(fit_distortions)), "fit rotation": 1,
           "fit magnification": 1, "zoomFactor_mm range": 2, "rot_rad_mm range": 2, "zoomX_mm range": 0.5,
           "zoomY_mm range": 0.5, "spiral_kpix_mm range": 1, "pincushion_kpix_mm range": 1,
           "max shift": int(max_shift_px / b) + 4, "Nedge": max(8, 50 // b), "Nedge for smoothing": max(8, 50 // b),
           "intermediate loops": 4, "final loops": int(final_loops), "residue loops": 0, "fit defocus": 2,
           "init loops 2": 3, "init loops 3": 3, "bandpass filter high": 0.2, "exponent": 1.0,
           "distortion frequency": 30, "distortion refine": 0, "data normalized": 1, "backend": backend}
    if params.convergence_rad:
        cfg["alpha_mrad"] = params.convergence_rad * np.sqrt(2) * 1e3
    if params.aperture_rad:
        cfg["objective aperture (mrad)"] = params.aperture_rad * 1e3
        cfg["objective aperture edge (mrad)"] = 0.1 * params.aperture_rad * 1e3
    cfg.update(config or {})
    res = frwr3.run_frwr3(work, pb, cfg, roi=roi, callback=callback)
    if getattr(res, "stopped", False):
        return None, {"stopped": True, "result": res}
    fo = res.frame_offsets_subpixel if res.frame_offsets_subpixel is not None else res.frame_offsets
    full = st if (ny, nx) == (NY, NX) else st[:, :ny, :nx]
    out, info = frwr3.align_frames(full, res.distortions, fo, res.reference_plane, binning=b, crop=crop)
    info.update(refined_defoci_nm=np.asarray(res.defoci_nm, float), R_final=float(res.rvalues[-1][2]),
                result=res, roi_binned=roi)
    try:
        info.update(initial_wave(res, b, info.get("crop", (0, 0, ny, nx))[:2], out.shape[1:]))
    except Exception as e:  # noqa: BLE001  (the alignment itself is still valid)
        info["wave_error"] = repr(e)
    return out, info


def _fourier_upsample(w, b):
    """Band-limited (periodic) interpolation of a complex array by an integer factor b."""
    if b == 1:
        return np.array(w, complex)
    ny, nx = w.shape
    F = np.fft.fftshift(np.fft.fft2(w))
    G = np.zeros((ny * b, nx * b), complex)
    y0, x0 = (ny * b) // 2 - ny // 2, (nx * b) // 2 - nx // 2      # zero frequency stays at the centre
    G[y0:y0 + ny, x0:x0 + nx] = F
    return np.fft.ifft2(np.fft.ifftshift(G)) * b * b


def _reflect_index(i, n):
    if n == 1:
        return np.zeros_like(i)
    p = 2 * n - 2
    i = np.abs(i) % p
    return np.where(i >= n, p - i, i)


def initial_wave(res, binning, crop_tl, shape):
    """FRWR3 wave of the (binned) alignment run at the resolution and geometry of the aligned stack.

    The padded wave is Fourier-interpolated by ``binning`` and shifted by (b - 1)/2 pixels (centre
    of a binned pixel), then placed in the coordinates of the aligned, cropped stack (the
    reference frame).  Returns dict(wave: exit wave on the region reconstructed by FRWR3 (the ROI)
    inside the aligned stack, wave_box: (top, left, bottom, right) of that region in the aligned
    stack, wave_start: wave on the whole aligned-stack area (outside the padded wave: mirrored),
    used as start wave for a refinement with all frames).
    """
    b = int(binning)
    wp = np.asarray(res.wave_padded, complex)
    py, px = res.pad_offset
    sy, sx = res.wave.shape
    ref = res.reference_plane
    fx, fy = np.asarray(res.frame_offsets, float)[ref]
    up = _fourier_upsample(wp, b)
    if b > 1:
        up = fourier_shift(up, ((b - 1) / 2, (b - 1) / 2))
    t, l = crop_tl
    oy, ox = int(round((fy - py) * b)) - t, int(round((fx - px) * b)) - l    # up[0, 0] in aligned coordinates
    H, W = shape
    rr = _reflect_index(np.arange(H) - oy, up.shape[0])
    cc = _reflect_index(np.arange(W) - ox, up.shape[1])
    start = up[np.ix_(rr, cc)]
    top, left = oy + py * b, ox + px * b
    bt, rt = max(top, 0), max(left, 0)
    bb, rb = min(top + sy * b, H), min(left + sx * b, W)
    return {"wave": start[bt:bb, rt:rb].copy(), "wave_box": (int(bt), int(rt), int(bb), int(rb)),
            "wave_start": start}
