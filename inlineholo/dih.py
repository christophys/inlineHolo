# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Many-beam dark-field inline holography (DIH): strain from several reflections.

Python port of the DigitalMicrograph scripts of the FRWR tools (C. T. Koch, "DIH-GPA" menu:
``createDFstack.s``, ``DFHoloAllBeamsGPA.s``, ``DFHoloAllBeamsGPA_computeDisplacement.s``).

Each reflection ``g_j`` contributes a reconstructed dark-field wave (amplitude ``A_j``, phase
``phi_j``), e.g. the exit wave of a DF focal series reconstructed with FRWR.  The geometric phase
is ``phi_j = -2 pi g_j . u + const`` so that, per pixel and for each derivative direction d,

    d phi_j / d x_d = -2 pi  sum_i g_ji  d u_i / d x_d .

With N >= 2 non-collinear reflections the displacement gradient D_id = d u_i / d x_d is obtained
by weighted linear least squares over the reflections (weights ``w_j = A_j^2``, the DF
intensity, as in the DM script):

    D_(.d) = -(1 / 2 pi) (sum_j w_j g_j g_j^T)^-1  sum_j w_j g_j d phi_j / d x_d .

Before the fit, the amplitude-weighted mean phase gradient of every reflection in a reference
region is subtracted (``reference correction`` of the DM script): it removes the arbitrary
phase ramps that a slight mis-alignment of the DF focal series produces and makes the
reference region unstrained by definition.  The results are e_xx = D_xx, e_yy = D_yy, the
symmetric shear e_xy = (D_xy + D_yx)/2 and the rigid rotation w = (D_yx - D_xy)/2 (rad,
counter-clockwise positive), optionally band-limited by a round mask of radius
1/resolution with a cosine edge (``smooth`` = fraction of the radius).  With one reflection
only the strain along g, e_gg = -(g_hat . grad phi) / (2 pi |g|), is available.

Coordinates: x to the right (image columns), y upwards (against the row index), z = x × y
towards the viewer, as in the DM scripts.  2D reflection vectors are computed from the unit
cell (a, b, c, alpha, beta, gamma), the zone axis [uvw] (real-space direction, parallel to z)
and the reflection (hkl) that defines the x direction; an additional in-plane rotation of the
crystal relative to the image x axis can be given.

``displacement_from_phases`` is the variant ``DFHoloAllBeamsGPA_computeDisplacement.s``: the
(unwrapped) phases themselves, after removing a plane fitted in the reference region, are
combined into the displacement field u = -(1/2 pi) (sum w g g^T)^-1 sum w g phi.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

#: lattice presets used in the DM scripts (nm, degrees)
LATTICES = {
    "Si": (0.5431, 0.5431, 0.5431, 90.0, 90.0, 90.0),
    "GaN": (0.3189, 0.3189, 0.5185, 90.0, 90.0, 120.0),
    "AlN": (0.3111, 0.3111, 0.4978, 90.0, 90.0, 120.0),
    "SrTiO3": (0.3905, 0.3905, 0.3905, 90.0, 90.0, 90.0),
    "Ge": (0.5658, 0.5658, 0.5658, 90.0, 90.0, 90.0),
    "GaAs": (0.5653, 0.5653, 0.5653, 90.0, 90.0, 90.0),
    "test (cubic 0.456 nm)": (0.456, 0.456, 0.456, 90.0, 90.0, 90.0),
}


# ----------------------------------------------------------------------------- lattice
def cell_vectors(a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
    """Real-space basis (rows a, b, c) in nm, a along x, b in the xy plane."""
    al, be, ga = np.deg2rad([alpha, beta, gamma])
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c ** 2 - cx ** 2 - cy ** 2, 0.0))
    return np.array([va, vb, np.array([cx, cy, cz])])


def reciprocal_vectors(cell):
    """Reciprocal basis (rows a*, b*, c*) in 1/nm (no 2 pi): a_i . b*_j = delta_ij."""
    A = np.asarray(cell, float)
    return np.linalg.inv(A).T


def g_vectors_2d(hkls, lattice, zone=(0, 0, 1), x_hkl=(1, 0, 0), rotation_deg=0.0):
    """2D reflection vectors (N, 2) [g_x, g_y] in 1/nm in the image frame.

    lattice : (a, b, c, alpha, beta, gamma) in nm / degrees
    zone    : zone axis [uvw] (real-space direction, becomes the image z axis)
    x_hkl   : reflection whose projection defines the image x axis
    rotation_deg : rotation of the crystal relative to the image x axis (counter-clockwise)
    Reflections with a component along the zone axis (not in the zero-order Laue zone) are
    projected onto the image plane.
    """
    cell = cell_vectors(*lattice)
    rec = reciprocal_vectors(cell)
    z = np.asarray(zone, float) @ cell
    z = z / np.linalg.norm(z)
    gx = np.asarray(x_hkl, float) @ rec
    gx = gx - (gx @ z) * z
    if np.linalg.norm(gx) < 1e-12:
        raise ValueError(f"the x reflection {tuple(x_hkl)} is parallel to the zone axis {tuple(zone)}")
    ex = gx / np.linalg.norm(gx)
    ey = np.cross(z, ex)
    G = np.asarray(hkls, float).reshape(-1, 3) @ rec
    g2 = np.c_[G @ ex, G @ ey]
    r = np.deg2rad(rotation_deg)
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    return g2 @ R.T


def parse_hkl(text):
    """'[2 0 0]', '(-1,1,0)', '200', '[-110]', '2 0 0' -> (h, k, l) or None."""
    s = str(text)
    m = re.search(r"[\[(]\s*(-?\d+)[\s,]+(-?\d+)[\s,]+(-?\d+)\s*[\])]", s)
    if m:
        return tuple(int(v) for v in m.groups())
    m = re.search(r"[\[(]\s*((?:-?\d){3})\s*[\])]", s)
    if not m:
        m = re.fullmatch(r"\s*((?:-?\d){3})\s*", s)
    if m:
        return tuple(int(v) for v in re.findall(r"-?\d", m.group(1)))
    m = re.fullmatch(r"\s*(-?\d+)[\s,]+(-?\d+)[\s,]+(-?\d+)\s*", s)
    if m:
        return tuple(int(v) for v in m.groups())
    for tok in re.split(r"[_\s.]+", s):                 # e.g. wave_220, DF_-111_phase
        if re.fullmatch(r"(?:-?\d){3}", tok):
            return tuple(int(v) for v in re.findall(r"-?\d", tok))
    return None


# ----------------------------------------------------------------------------- data
@dataclass
class DFReflection:
    """One dark-field data set: amplitude and phase (radian) of the reconstructed DF wave."""
    hkl: tuple
    amplitude: np.ndarray
    phase: np.ndarray
    pixel_nm: float
    name: str = ""
    shift_px: tuple = (0.0, 0.0)          # (dx, dy) applied before the analysis (x right, y down)
    use: bool = True
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def wave(self):
        return self.amplitude * np.exp(1j * self.phase)


def from_wave(wave, hkl, pixel_nm, name="", source=""):
    w = np.asarray(wave, complex)
    return DFReflection(tuple(hkl), np.abs(w), np.angle(w), float(pixel_nm), name or str(tuple(hkl)), source=source)


def _pad_to(a, shape, fill):
    ny, nx = a.shape
    Y, X = shape
    oy, ox = (Y - ny) // 2, (X - nx) // 2
    out = np.full(shape, fill, a.dtype)
    out[oy:oy + ny, ox:ox + nx] = a
    return out


def common_frame(refls):
    """Amplitudes and phases of all used reflections, centred in a common frame (as createDFstack.s:
    smaller images are centred, amplitude 0 / mean phase outside) and shifted by ``shift_px``."""
    from scipy.ndimage import shift as nd_shift
    use = [r for r in refls if r.use]
    if not use:
        raise ValueError("no reflection selected")
    Y = max(r.amplitude.shape[0] for r in use)
    X = max(r.amplitude.shape[1] for r in use)
    A, P = [], []
    for r in use:
        a = _pad_to(np.asarray(r.amplitude, float), (Y, X), 0.0)
        ph = np.asarray(r.phase, float)
        p = _pad_to(ph, (Y, X), float(np.mean(ph)))
        dx, dy = r.shift_px
        if dx or dy:
            if float(dx).is_integer() and float(dy).is_integer():
                a = np.roll(a, (int(dy), int(dx)), (0, 1))
                p = np.roll(p, (int(dy), int(dx)), (0, 1))
            else:                                   # sub-pixel: shift the complex wave
                w = nd_shift(a * np.cos(p), (dy, dx), order=1, mode="nearest") + \
                    1j * nd_shift(a * np.sin(p), (dy, dx), order=1, mode="nearest")
                a, p = np.abs(w), np.angle(w)
        A.append(a)
        P.append(p)
    return use, np.array(A), np.array(P)


def phase_gradients(phase, pixel_nm):
    """Wrap-free central differences of a phase map: (d/dx, d/dy) in rad/nm, y upwards."""
    e = np.exp(1j * np.asarray(phase, float))
    gx = np.angle(np.roll(e, -1, 1) * np.conj(np.roll(e, 1, 1))) / (2 * pixel_nm)
    gy = -np.angle(np.roll(e, -1, 0) * np.conj(np.roll(e, 1, 0))) / (2 * pixel_nm)
    gx[:, 0] = gx[:, 1]
    gx[:, -1] = gx[:, -2]
    gy[0] = gy[1]
    gy[-1] = gy[-2]
    return gx, gy


def resolution_mask(shape, pixel_nm, resolution_nm, smooth=0.0):
    """Round band limit |q| <= 1/resolution with a cosine edge from (1 - smooth)/resolution."""
    ny, nx = shape
    qy = np.fft.fftfreq(ny, pixel_nm)[:, None]
    qx = np.fft.fftfreq(nx, pixel_nm)[None, :]
    q = np.hypot(qx, qy)
    hi = 1.0 / resolution_nm
    lo = (1.0 - smooth) * hi
    m = (q <= hi).astype(float)
    if smooth > 0:
        edge = (q > lo) & (q <= hi)
        m[edge] = 0.5 * (1 + np.cos(np.pi * (q[edge] - lo) / (hi - lo)))
    return m


def _band_limit(a, mask):
    a = np.nan_to_num(a)
    return np.real(np.fft.ifft2(np.fft.fft2(a) * mask))


def _box_slices(ref_box, shape):
    if ref_box is None:
        ny, nx = shape
        return slice(ny // 4, 3 * ny // 4), slice(nx // 4, 3 * nx // 4)
    t, l, b, r = (int(v) for v in ref_box)
    return slice(max(t, 0), b), slice(max(l, 0), r)


def strain_from_phases(refls, g2d=None, lattice=None, zone=(0, 0, 1), x_hkl=(1, 0, 0), rotation_deg=0.0,
                       ref_box=None, weight="intensity", reference_correction=True, resolution_nm=None,
                       smooth=0.0, det_threshold=1e-3):
    """Strain maps from N dark-field reflections (DFHoloAllBeamsGPA.s, 'compute strain maps').

    refls  : list of DFReflection (only ``use`` ones are taken)
    g2d    : (N_used, 2) reflection vectors in 1/nm (image frame); computed from
             ``lattice``/``zone``/``x_hkl``/``rotation_deg`` and the hkl of each reflection if None
    ref_box: (top, left, bottom, right) of the reference region in pixels (default central half)
    weight : 'intensity' (A^2, as in the DM script), 'amplitude' or 'uniform'
    det_threshold : pixels where det(sum w g g^T) < threshold * max are set to NaN
    Returns dict with e_xx, e_yy, e_xy, rotation (rad), D (2, 2, ny, nx displacement gradient
    du_i/dx_d), weight_sum, g (N, 2), hkl, ref_box, and for a single reflection e_gg.
    """
    use, A, P = common_frame(refls)
    pix = use[0].pixel_nm
    if g2d is None:
        if lattice is None:
            raise ValueError("give either g2d or the lattice")
        g2d = g_vectors_2d([r.hkl for r in use], lattice, zone, x_hkl, rotation_deg)
    g2d = np.asarray(g2d, float).reshape(len(use), 2)
    ny, nx = A.shape[1:]
    sy, sx = _box_slices(ref_box, (ny, nx))
    W = {"intensity": A ** 2, "amplitude": A, "uniform": np.ones_like(A)}[weight]
    XX = np.zeros((2, 2, ny, nx))
    XY = np.zeros((2, 2, ny, nx))                    # [component i, derivative d]
    grads = []
    for j in range(len(use)):
        gx, gy = phase_gradients(P[j], pix)
        if reference_correction:
            wr = W[j][sy, sx]
            s = wr.sum() + 1e-30
            gx = gx - (gx[sy, sx] * wr).sum() / s
            gy = gy - (gy[sy, sx] * wr).sum() / s
        grads.append((gx, gy))
        g = g2d[j]
        for a in range(2):
            for b in range(2):
                XX[a, b] += W[j] * g[a] * g[b]
            XY[a, 0] += W[j] * g[a] * gx
            XY[a, 1] += W[j] * g[a] * gy
    out = {"g": g2d, "hkl": [r.hkl for r in use], "names": [r.name for r in use], "weight_sum": W.sum(0),
           "ref_box": (sy.start, sx.start, sy.stop, sx.stop), "pixel_nm": pix}
    mask = resolution_mask((ny, nx), pix, resolution_nm, smooth) if resolution_nm else None
    if len(use) == 1 or np.linalg.matrix_rank(g2d) < 2:
        # strain along g only (weighted mean over collinear reflections)
        num = sum(W[j] * (g2d[j] @ np.array(grads[j])[:, :, :].reshape(2, -1)).reshape(ny, nx) /
                  (g2d[j] @ g2d[j]) for j in range(len(use)))
        egg = -num / (2 * np.pi * W.sum(0) + 1e-30)
        if mask is not None:
            egg = _band_limit(egg, mask)
        out["e_gg"] = egg
        out["g_direction"] = g2d[0] / np.linalg.norm(g2d[0])
        return out
    det = XX[0, 0] * XX[1, 1] - XX[0, 1] * XX[1, 0]
    bad = det < det_threshold * np.nanmax(det)
    det = np.where(bad, np.nan, det)
    inv = np.array([[XX[1, 1], -XX[0, 1]], [-XX[1, 0], XX[0, 0]]]) / det
    D = -np.einsum("ab...,bd...->ad...", inv, XY) / (2 * np.pi)
    if mask is not None:
        D = np.array([[_band_limit(D[a, b], mask) for b in range(2)] for a in range(2)])
        D[:, :, bad] = np.nan
    out.update(D=D, e_xx=D[0, 0], e_yy=D[1, 1], e_xy=0.5 * (D[0, 1] + D[1, 0]),
               rotation=0.5 * (D[1, 0] - D[0, 1]), e_yx_raw=D[1, 0], e_xy_raw=D[0, 1], valid=~bad)
    return out


def displacement_from_phases(refls, g2d=None, lattice=None, zone=(0, 0, 1), x_hkl=(1, 0, 0), rotation_deg=0.0,
                             ref_box=None, weight="amplitude", resolution_nm=None, smooth=0.0, unwrap=True):
    """Displacement field u (nm) from N reflections (DFHoloAllBeamsGPA_computeDisplacement.s).

    The phases are unwrapped (skimage) unless ``unwrap=False``; a plane fitted in the reference
    region is subtracted (amplitude weighted), the phases are band limited and combined by
    weighted least squares: u = -(1/2 pi) (sum w g g^T)^-1 sum w g phi.
    """
    use, A, P = common_frame(refls)
    pix = use[0].pixel_nm
    if g2d is None:
        g2d = g_vectors_2d([r.hkl for r in use], lattice, zone, x_hkl, rotation_deg)
    g2d = np.asarray(g2d, float).reshape(len(use), 2)
    ny, nx = A.shape[1:]
    sy, sx = _box_slices(ref_box, (ny, nx))
    W = {"intensity": A ** 2, "amplitude": A, "uniform": np.ones_like(A)}[weight]
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    mask = resolution_mask((ny, nx), pix, resolution_nm, smooth) if resolution_nm else None
    XX = np.zeros((2, 2, ny, nx))
    XY = np.zeros((2, ny, nx))
    for j in range(len(use)):
        ph = P[j]
        if unwrap:
            from skimage.restoration import unwrap_phase
            ph = unwrap_phase(ph)
        wr = W[j][sy, sx].ravel()
        M = np.c_[np.ones(wr.size), xx[sy, sx].ravel(), yy[sy, sx].ravel()]
        coef = np.linalg.lstsq(M * wr[:, None], ph[sy, sx].ravel() * wr, rcond=None)[0]
        ph = ph - (coef[0] + coef[1] * xx + coef[2] * yy)
        if mask is not None:
            ph = _band_limit(ph, mask)
        g = g2d[j]
        for a in range(2):
            for b in range(2):
                XX[a, b] += W[j] * g[a] * g[b]
            XY[a] += W[j] * g[a] * ph
    det = XX[0, 0] * XX[1, 1] - XX[0, 1] * XX[1, 0]
    det = np.where(det < 1e-3 * np.nanmax(det), np.nan, det)
    inv = np.array([[XX[1, 1], -XX[0, 1]], [-XX[1, 0], XX[0, 0]]]) / det
    u = -np.einsum("ab...,b...->a...", inv, XY) / (2 * np.pi)
    return {"u_x": u[0], "u_y": u[1], "g": g2d, "hkl": [r.hkl for r in use],
            "ref_box": (sy.start, sx.start, sy.stop, sx.stop), "pixel_nm": pix}


def auto_shift(refls, ref_index=0, max_shift_px=50):
    """Integer shifts (dx, dy) that align the amplitudes of all reflections to ``ref_index``
    (phase correlation of the amplitudes; the DM script aligns them by hand)."""
    from skimage.registration import phase_cross_correlation
    use, A, _ = common_frame([DFReflection(r.hkl, r.amplitude, r.phase, r.pixel_nm, use=r.use) for r in refls])
    out = []
    ref = A[ref_index] - A[ref_index].mean()
    for a in A:
        sh = phase_cross_correlation(ref, a - a.mean(), upsample_factor=1, normalization=None)[0]
        sh = np.clip(sh, -max_shift_px, max_shift_px)
        out.append((float(sh[1]), float(sh[0])))
    return out


def overlay_rgb(a1, a2):
    """Red/green overlay of two images (as showAlignedPhases of the DM script)."""
    def nrm(a):
        a = np.nan_to_num(np.asarray(a, float))
        lo, hi = np.percentile(a, [1, 99.5])
        return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)
    return np.dstack([nrm(a2), nrm(a1), np.zeros(a1.shape)])


# ----------------------------------------------------------------------------- file input
def _read_any_2d(path):
    """Amplitude, phase or complex wave from .dm3/.dm4/.img/.tif/.npy -> (array, pixel_nm, tags)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".img":
        from .frwr3_io import read_img
        a, h = read_img(path)
        return a, (h["dx"] / 10.0 if h["dx"] > 0 else None), {}
    if ext in (".dm3", ".dm4"):
        from .dmreader import read_dm
        ims, _ = read_dm(path)
        im = max(ims, key=lambda i: i["data"].size)                 # skip the thumbnail
        d = np.asarray(im["data"])
        pix = None
        if im["units"] and im["units"][-1] in ("nm", "µm", "um", "Å", "A"):
            f = {"nm": 1.0, "µm": 1e3, "um": 1e3, "Å": 0.1, "A": 0.1}[im["units"][-1]]
            pix = float(im["scales"][-1]) * f
        return d, pix, im.get("tags") or {}
    if ext == ".npy":
        return np.load(path), None, {}
    from .io_formats import load_series
    st, md, _ = load_series(path)
    return st[0] if len(st) == 1 else st, md.get("pixel_size_nm"), {}


def load_reflections(paths, pixel_nm=None, skipped=None):
    """DF data sets from files.

    * complex wave files (e.g. FRWR3 ``avgWave.img``): one reflection each;
    * amplitude/phase pairs: ``<name>amplitude.<ext>`` with ``<name>phase.<ext>`` next to it
      (as createDFstack.s expects), e.g. ``[200]_amplitude.dm3`` / ``[200]_phase.dm3``;
    * DM stacks made by createDFstack.s (``amplitude stack`` / ``phase stack`` with the
      ``DFHoloGPA`` tags: hkl and shifts of every frame, lattice, zone, reference region);
      give the amplitude stack, the phase stack is found by replacing 'amplitude' by 'phase'.
    The reflection (hkl) is taken from the tags or from the file name (``[200]``, ``(1 1 0)``,
    ``_220_``); otherwise it must be set afterwards.
    Returns (list of DFReflection, settings dict from DFHoloGPA tags or {}).  Files that are
    neither complex nor have a phase partner raise ValueError, or are appended to ``skipped``
    (a list) if one is given.
    """
    out, settings = [], {}
    done = set()
    for p in paths:
        if p in done:
            continue
        name = os.path.basename(p)
        low = name.lower()
        partner = None
        if "phase" in low and "amplitude" not in low:
            cand = _swap(p, "phase", "amplitude")
            if cand and os.path.exists(cand):
                if cand in paths or cand in done:
                    continue                                       # handled with its amplitude file
                p, name, low = cand, os.path.basename(cand), os.path.basename(cand).lower()
        if "amplitude" in low or "amp" in low:
            partner = _swap(p, "amplitude", "phase") or _swap(p, "amp", "phase")
            if partner and not os.path.exists(partner):
                partner = None
        a, pix, tags = _read_any_2d(p)
        pix = pixel_nm or pix
        done.add(p)
        if partner:
            ph, _, _ = _read_any_2d(partner)
            done.add(partner)
            a, ph = np.asarray(a, float), np.asarray(ph, float)
            gtag = tags.get("DFHoloGPA") if isinstance(tags, dict) else None
            if a.ndim == 3:                                        # DM DF stack
                gt = gtag or {}
                fr = gt.get("Frames", {})
                for j in range(a.shape[0]):
                    f = fr.get(str(j), {})
                    hkl = (int(f.get("h", 0)), int(f.get("k", 0)), int(f.get("l", 0)))
                    out.append(DFReflection(hkl, a[j], ph[j], pix or 1.0, name=f"{name}[{j}] {hkl}",
                                            shift_px=(float(f.get("shiftX", 0)), float(f.get("shiftY", 0))),
                                            source=p))
                settings = _settings_from_tags(gt)
            else:
                hkl = parse_hkl(name) or (0, 0, 0)
                out.append(DFReflection(hkl, a, ph, pix or 1.0, name=os.path.splitext(name)[0], source=p))
        else:
            a = np.asarray(a)
            if not np.iscomplexobj(a):
                msg = f"{name}: not a complex wave and no matching '...phase' file found"
                if skipped is None:
                    raise ValueError(msg)
                skipped.append(name)
                continue
            stem = os.path.splitext(name)[0]
            out.append(from_wave(a, parse_hkl(stem) or parse_hkl(os.path.basename(os.path.dirname(p))) or (0, 0, 0),
                                 pix or 1.0, stem, p))
    return out, settings


def _swap(path, old, new):
    d, n = os.path.split(path)
    i = n.lower().rfind(old)
    if i < 0:
        return None
    return os.path.join(d, n[:i] + new + n[i + len(old):])


def _settings_from_tags(gt):
    if not gt:
        return {}
    s = {}
    try:
        s["lattice"] = tuple(float(gt[k]) for k in ("ax", "by", "cz", "alpha", "beta", "gamma"))
    except (KeyError, TypeError, ValueError):
        pass
    for k, key in (("zone", "zone"), ("x_hkl", "X")):
        if isinstance(gt.get(key), dict):
            s[k] = tuple(int(gt[key].get(c, 0)) for c in "hkl")
    sel = gt.get("Selection")
    if isinstance(sel, dict):
        s["ref_box"] = tuple(int(sel.get(c, 0)) for c in ("top", "left", "bottom", "right"))
    gen = gt.get("General")
    if isinstance(gen, dict):
        if gen.get("applyResolutionLimit", 1):
            s["resolution_nm"] = float(gen.get("resolution", 0) or 0) or None
        s["smooth"] = float(gen.get("smooth", 0) or 0)
        s["symmetric"] = bool(gen.get("display symmetric", 1))
    return s
