# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2008-2026 Christoph T. Koch
"""File-format helpers for interoperability with the C++ FRWR3 program (C. T. Koch).

* ``write_cdf`` / ``read_cdf``: the netCDF-3 focal-series container read by ``frwr3``
  (dimensions ``width``, ``height``, ``depth``; variable ``data``; ``Microscope.*``
  global attributes).
* ``read_img`` / ``write_img``: the binary ``.img`` format written by ``frwr3``
  (``amplitude.img``, ``phase.img``, ``avgWave.img``, ...).
* ``read_conf`` / ``write_conf``: the ``key: value`` control files of ``frwr3``
  (``%`` starts a comment).  See :mod:`inlineholo.frwr3` for the YAML equivalent.

Units in the files follow FRWR3 (nm for lengths, mm for Cs, mrad for angles, deg for
the astigmatism angle).  The functions return/accept :class:`ImagingParameters`
(nm, rad) where appropriate.
"""
from __future__ import annotations

import struct

import numpy as np

from .params import ImagingParameters


# --------------------------------------------------------------------------- netCDF
def write_cdf(path, stack, params: ImagingParameters, roi=None, aperture_mrad=None,
              aperture_edge_mrad=0.1, normalized=True, defocus_offset_nm=None,
              defocus_step_nm=None):
    """Write a focal series in the netCDF layout expected by ``frwr3``.

    stack : (n, ny, nx) float array (whole frames).
    roi   : (top, left, bottom, right) region to reconstruct (default: whole frame).
    """
    from scipy.io import netcdf_file
    stack = np.asarray(stack, np.float32)
    n, ny, nx = stack.shape
    lam = params.wavelength_nm
    df = np.asarray(params.defoci_nm, float)
    ic = int(np.argmin(np.abs(df)))
    step = defocus_step_nm if defocus_step_nm is not None else (
        float(np.median(np.diff(df))) if n > 1 else 0.0)
    off = defocus_offset_nm if defocus_offset_nm is not None else float(df[ic])
    if aperture_mrad is None:
        aperture_mrad = 1e3 * (params.aperture_rad if params.aperture_rad else
                               1.5 * np.sqrt(2) * lam / (2 * params.pixel_size_nm))
    ab = params.aberrations
    a, b = ab.get("c_1_2_a", 0.0), ab.get("c_1_2_b", 0.0)
    ast = float(np.hypot(a, b))
    # frwr3 uses fa = def + A cos(2(phi - phiA)) with phiA = -AstPhi_deg * pi/180
    phiA = 0.5 * np.arctan2(b, a)
    f = netcdf_file(str(path), "w", version=1)
    f.createDimension("width", nx)
    f.createDimension("height", ny)
    f.createDimension("depth", n)
    vx = f.createVariable("width", "f4", ("width",))
    vy = f.createVariable("height", "f4", ("height",))
    vz = f.createVariable("depth", "f4", ("depth",))
    vx[:] = np.arange(nx) * params.pixel_size_nm
    vy[:] = np.arange(ny) * params.pixel_size_nm
    vz[:] = df
    vx.scale = np.float32(params.pixel_size_nm)
    vy.scale = np.float32(params.pixel_size_nm)
    vz.scale = np.float32(step)
    vz.origin = np.float32(0.0)
    d = f.createVariable("data", "f4", ("depth", "height", "width"))
    d[:] = stack
    roi = (0, 0, ny, nx) if roi is None else roi
    attrs = {
        "Microscope.Voltage_V": params.voltage_V,
        "Microscope.Binning": 1.0,
        "Microscope.alpha_mrad": 1e3 * np.sqrt(2) * params.convergence_rad,
        "Microscope.Delta_nm": params.focal_spread_nm,
        "Microscope.Cs_mm": ab.get("c_3_0", 0.0) * 1e-6,
        "Microscope.AstMag_nm": ast,
        "Microscope.AstPhi_deg": -np.rad2deg(phiA),
        "Microscope.ObjApert_mrad": aperture_mrad,
        "Microscope.ObjApertEdge_mrad": aperture_edge_mrad,
        "Microscope.ObjApertOffsetX_mrad": 0.0,
        "Microscope.ObjApertOffsetY_mrad": 0.0,
        "Microscope.DefocusStep_nm": step,
        "Microscope.DefocusOffset_nm": off,
        "Microscope.DefocusScale": 1.0,
    }
    for k, v in attrs.items():
        setattr(f, k, np.float32(v))
    setattr(f, "Microscope.ROI", np.asarray(roi, np.float32))
    setattr(f, "DMtags.FRWR.normalized", "1" if normalized else "0")
    setattr(f, "DMtags.FRWR.FocalSeries.Rot_MuRad_mm", "0")
    setattr(f, "DMtags.FRWR.FocalSeries.Zoomfactor_mm", "0")
    f.close()


def read_cdf(path):
    """Read an FRWR3 netCDF focal series -> (stack, params, roi, info)."""
    from scipy.io import netcdf_file
    with netcdf_file(str(path), "r", mmap=False) as f:
        data = np.array(f.variables["data"][:], float)
        g = {k.decode() if isinstance(k, bytes) else k: v for k, v in f._attributes.items()}
        dx = float(getattr(f.variables["width"], "scale", 1.0))
        step = float(g.get("Microscope.DefocusStep_nm", getattr(f.variables["depth"], "scale", 0.0)))
        off = float(g.get("Microscope.DefocusOffset_nm", 0.0))
        n = data.shape[0]
        c = (n - 1) // 2
        df = off + (np.arange(n) - c) * step
        if "depth" in f.variables:
            z = np.array(f.variables["depth"][:], float)
            if np.ptp(z) > 0:
                df = z
        cs = float(g.get("Microscope.Cs_mm", 0.0)) * 1e6
        ast = float(g.get("Microscope.AstMag_nm", 0.0))
        phi = -np.deg2rad(float(g.get("Microscope.AstPhi_deg", 0.0)))
        alpha = float(g.get("Microscope.alpha_mrad", 0.0)) * 1e-3 / np.sqrt(2)
        ab = {"c_3_0": cs}
        if ast:
            ab["c_1_2_a"] = ast * np.cos(2 * phi)
            ab["c_1_2_b"] = ast * np.sin(2 * phi)
        roi = np.asarray(g.get("Microscope.ROI", [0, 0, 0, 0]), float)
        if np.all(roi == 0):
            roi = np.array([0, 0, data.shape[1], data.shape[2]], float)
        ap = float(g.get("Microscope.ObjApert_mrad", 0.0)) * 1e-3
        p = ImagingParameters(voltage_V=float(g.get("Microscope.Voltage_V", 300e3)),
                              pixel_size_nm=dx, defoci_nm=df, aberrations=ab,
                              focal_spread_nm=float(g.get("Microscope.Delta_nm", 0.0)),
                              convergence_rad=alpha, aperture_rad=ap if ap > 0 else None)
        info = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in g.items()}
    return data, p, tuple(int(v) for v in roi), info


# --------------------------------------------------------------------------- .img
def read_img(path):
    """Read an FRWR3/QSTEM ``.img`` file -> (array, header dict).  Complex if stored so."""
    with open(path, "rb") as fh:
        buf = fh.read()
    hsize, psize, csize, nx, ny, cflag, dsize, ver = struct.unpack_from("<8i", buf, 0)
    t, dx, dy = struct.unpack_from("<3d", buf, 32)
    off = hsize
    params = np.frombuffer(buf, "<f8", psize, off)
    off += 8 * psize
    comment = buf[off:off + csize].decode(errors="ignore")
    off += csize
    if cflag:
        dt = "<c8" if dsize == 8 else "<c16"
    else:
        dt = "<f4" if dsize == 4 else "<f8"
    a = np.frombuffer(buf, dt, nx * ny, off)
    # frwr3 writes images transposed (x runs slowest)
    a = a.reshape(nx, ny).T.copy()
    return a, dict(nx=nx, ny=ny, complex=bool(cflag), t=t, dx=dx, dy=dy, version=ver,
                   params=params, comment=comment)


def write_img(path, array, dx=1.0, dy=1.0, t=0.0, transpose=True):
    a = np.asarray(array)
    ny, nx = a.shape
    cflag = int(np.iscomplexobj(a))
    out = a.astype(np.complex64 if cflag else np.float32)
    dsize = 8 if cflag else 4
    hsize = 8 * 4 + 3 * 8
    hdr = struct.pack("<8i", hsize, 0, 0, nx, ny, cflag, dsize, 1) + struct.pack("<3d", t, dx, dy)
    body = (out.T if transpose else out).tobytes()
    with open(path, "wb") as fh:
        fh.write(hdr + body)


# --------------------------------------------------------------------------- .conf
def read_conf(path_or_text):
    """Parse an frwr3 control file into an ordered dict {key: [values...]}.

    Keys are kept exactly as written (without the trailing colon); values are parsed as
    numbers where possible.  Lines are only recognised if the key starts in column 0,
    as in frwr3's own parser.
    """
    try:
        with open(path_or_text) as fh:
            text = fh.read()
    except (OSError, TypeError, ValueError):
        text = str(path_or_text)
    out = {}
    for line in text.splitlines():
        line = line.split("%", 1)[0].rstrip()
        if not line or line[0].isspace() or ":" not in line:
            continue
        key, val = line.split(":", 1)
        vals = []
        for tok in val.split():
            try:
                v = float(tok)
                vals.append(int(v) if v.is_integer() and "." not in tok and "e" not in tok.lower() else v)
            except ValueError:
                vals.append(tok)
        out[key.strip()] = vals
    return out


def write_conf(path, cfg: dict):
    lines = ["% frwr3 control file written by inlineholo"]
    for k, v in cfg.items():
        if v is None:
            continue
        vv = v if isinstance(v, (list, tuple, np.ndarray)) else [v]
        lines.append(f"{k}: " + " ".join(str(x) for x in vv))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
