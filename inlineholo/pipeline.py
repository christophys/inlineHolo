# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Headless reconstruction of a focal series: load -> (FRWR alignment) -> reconstruction -> NXem.

The same steps as tabs 1, 4b, 5 and 6 of the app, without a GUI.  Used by the NOMAD action
(``inlineholo_nomad.actions``) and usable from scripts::

    from inlineholo.pipeline import run
    out = run("data/hr_focal_series_sim.nxs", "output", method="frwr", n_iter=50)

Input: an NXem/NeXus file, a DM3/DM4/TIFF/EMD/MRC/NPY stack, one frame of an indexed ``.img``
series (the whole series is read) or a folder of single images (ordered by the defocus in their
names, e.g. ``F-1000nm.dm3``).  Missing imaging parameters are taken from ``overrides``.
"""
from __future__ import annotations

import os
import time

import numpy as np

from . import nxem
from .io_formats import load_folder, load_series
from .params import ImagingParameters

ITERATIVE = ("gs", "iwfr", "frwr", "frwr_gf", "gpfrwr", "frwr_gd", "gpfrwr_gd", "gpfrwr_lsq", "mal", "ml_poisson")


def load_input(path):
    """(stack, metadata, params or None) of a file or a folder."""
    if os.path.isdir(path):
        stack, md = load_folder(path)
        return np.asarray(stack, float), md, None
    stack, md, p = load_series(path)
    return np.asarray(stack, float), md, p


def params_from_metadata(md, p=None, n_images=None, overrides=None) -> ImagingParameters:
    """Imaging parameters from the file (``p`` or metadata ``md``), overridden by ``overrides``.

    overrides (all optional): voltage_kV, pixel_size_nm, defoci_nm (list), defocus_start_nm and
    defocus_step_nm (linear series), cs_mm, focal_spread_nm, convergence_mrad, aperture_mrad.
    """
    o = {k: v for k, v in (overrides or {}).items() if v is not None}
    if p is None:
        ab = {"c_3_0": float(md["c_3_0"])} if "c_3_0" in md else {}
        n = int(n_images or len(md.get("defoci_nm", [])) or 1)
        df = list(md.get("defoci_nm", []))
        p = ImagingParameters(voltage_V=float(md.get("voltage_V", 300e3)),
                              pixel_size_nm=float(md.get("pixel_size_nm", 0.05)),
                              defoci_nm=np.array((df + [np.nan] * n)[:n], float), aberrations=ab)
    n = len(p.defoci_nm)
    if "voltage_kV" in o:
        p.voltage_V = float(o["voltage_kV"]) * 1e3
    if "pixel_size_nm" in o:
        p.pixel_size_nm = float(o["pixel_size_nm"])
    if "cs_mm" in o:
        p.aberrations = {**p.aberrations, "c_3_0": float(o["cs_mm"]) * 1e6}
    if "focal_spread_nm" in o:
        p.focal_spread_nm = float(o["focal_spread_nm"])
    if "convergence_mrad" in o:
        p.convergence_rad = float(o["convergence_mrad"]) * 1e-3
    if "aperture_mrad" in o:
        p.aperture_rad = float(o["aperture_mrad"]) * 1e-3 or None
    if o.get("defoci_nm"):
        df = np.asarray(o["defoci_nm"], float)
        if len(df) != n:
            raise ValueError(f"{len(df)} defocus values given for {n} images")
        p.defoci_nm = df
    elif "defocus_start_nm" in o and "defocus_step_nm" in o:
        p.defoci_nm = float(o["defocus_start_nm"]) + float(o["defocus_step_nm"]) * np.arange(n)
    if np.any(~np.isfinite(p.defoci_nm)):
        raise ValueError("the defocus of every image is needed: the file does not contain it, give "
                         "defocus_start_nm and defocus_step_nm (or defoci_nm)")
    return p


def run(path, out_dir, method="frwr", n_iter=50, align=False, binning=2, overrides=None, out_name=None,
        callback=None, metadata=None, write_series=True, validate=False):
    """Reconstruct the exit wave of the focal series at ``path`` and write it as NXem.

    callback(stage, fraction, message) is called with the progress (stage "load", "align",
    "reconstruct", "write").  Returns dict(files, method, n_images, shape, R_final, runtime_s,
    valid, problems).
    """
    from . import align as _align
    from .reconstruct import METHODS, Reconstruction, reconstruct

    def say(stage, frac, msg):
        if callback:
            callback(stage, float(frac), msg)

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; available: {', '.join(METHODS)}")
    t0 = time.time()
    say("load", 0.0, f"loading {os.path.basename(path.rstrip('/'))}")
    stack, md, p = load_input(path)
    p = params_from_metadata(md, p, len(stack), overrides)
    if metadata is None and path.lower().endswith((".nxs", ".h5", ".hdf5", ".nx5")):
        try:
            metadata = nxem.read_metadata(path)     # keep sample, instrument and user of the input file
        except Exception:  # noqa: BLE001
            metadata = None
    if len(p.defoci_nm) != len(stack):
        raise ValueError(f"{len(stack)} images but {len(p.defoci_nm)} defocus values")
    say("load", 1.0, f"{len(stack)} images of {stack.shape[1]}×{stack.shape[2]} px, "
                     f"C1 = {p.defoci_nm.min():.1f} … {p.defoci_nm.max():.1f} nm")
    opts = {}
    init_rec = None
    if align:
        say("align", 0.0, f"FRWR alignment (binning {binning})")

        def acb(*a):
            say("align", 0.5, "FRWR alignment: loop " + str(a[1] if len(a) > 1 else ""))

        stack, info = _align.frwr3_align(stack, p, binning=binning, callback=acb)
        p.defoci_nm = np.asarray(info["refined_defoci_nm"], float)
        if "wave_start" in info and method in ITERATIVE:
            opts["init"] = np.asarray(info["wave_start"], complex)
        if "wave" in info:
            init_rec = Reconstruction(info["wave"], "frwr3_alignment", {"binning_fit": int(binning)},
                                      [r[2] for r in info["result"].rvalues], time.time() - t0, None,
                                      {"distortions_full_resolution": np.asarray(info["distortions"]).tolist(),
                                       "drift_px_xy": np.asarray(info["shifts_px"]).tolist(),
                                       "reference_plane": int(info["reference_plane"])})
        say("align", 1.0, f"aligned: {stack.shape[1]}×{stack.shape[2]} px, R = {info['R_final']:.4f}")
    if method in ITERATIVE:
        opts["n_iter"] = int(n_iter)

        def rcb(it, *_):
            say("reconstruct", min((it + 1) / max(int(n_iter), 1), 1.0), f"{method}: iteration {it + 1} of {n_iter}")

        opts["callback"] = rcb
    say("reconstruct", 0.0, f"reconstruction with {method}")
    rec = reconstruct(stack, p, method, **opts)
    say("reconstruct", 1.0, f"{method} done in {rec.runtime_s:.1f} s")
    os.makedirs(out_dir, exist_ok=True)
    base = out_name or (os.path.splitext(os.path.basename(path.rstrip("/")))[0] + f"_{method}.nxs")
    if not base.endswith(".nxs"):
        base += ".nxs"
    out = os.path.join(out_dir, base)
    say("write", 0.0, f"writing {base}")
    if write_series:
        nxem.write_focal_series(out, stack.astype(np.float32), p, metadata=metadata, is_simulation=None,
                                description=f"focal series {os.path.basename(path.rstrip('/'))}, reconstructed "
                                            f"with inlineholo ({method})")
    if init_rec is not None:
        nxem.write_reconstruction(out, init_rec, p, label="FRWR initial reconstruction")
    nxem.write_reconstruction(out, rec, p, label=method)
    valid, problems = None, []
    if validate:
        valid, problems, _ = nxem.validate(out)
    hist = list(getattr(rec, "history", []) or [])
    res = dict(files=[out], method=method, n_images=int(len(stack)), shape=list(stack.shape[1:]),
               R_final=float(hist[-1]) if hist and np.isscalar(hist[-1]) else None,
               runtime_s=round(time.time() - t0, 1), valid=valid, problems=list(problems)[:20])
    say("write", 1.0, f"written {base}")
    return res
