# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Loading experimental focal series from common file formats and extracting metadata.

Supported: NXem/NeXus HDF5 (.nxs, .h5, .hdf5), TIFF stacks (ImageJ/OME/plain; also
several single TIFFs), NumPy (.npy/.npz), MRC (.mrc, .st via ``mrcfile``), Digital Micrograph
(.dm3/.dm4, built-in reader; the tags of the FRWR DigitalMicrograph plugins are converted to
imaging parameters) and, if RosettaSciIO is installed, Velox EMD (.emd), SER etc.

The metadata dict contains (when found) ``pixel_size_nm``, ``voltage_V``,
``defoci_nm`` (list), ``c_3_0`` (nm), ``c_1_2_a``/``c_1_2_b``, ``magnification``,
``exposure_s`` and ``raw`` (the original metadata tree, for display).
"""
from __future__ import annotations

import io
import os
import re
import tempfile

import numpy as np

_UNIT = {"nm": 1.0, "nanometer": 1.0, "um": 1e3, "µm": 1e3, "micron": 1e3, "micrometer": 1e3,
         "mm": 1e6, "m": 1e9, "a": 0.1, "å": 0.1, "angstrom": 0.1, "pm": 1e-3, "\\u00b5m": 1e3}


def _to_nm(value, unit):
    if unit is None:
        return float(value)
    u = str(unit).strip().lower()
    return float(value) * _UNIT.get(u, 1.0)


def _walk(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk(v, f"{prefix}/{k}" if prefix else str(k))
    else:
        yield prefix, d


def _search_metadata(tree):
    """Heuristic extraction of common microscope parameters from a nested dict."""
    out = {}
    for key, val in _walk(tree):
        k = key.lower()
        try:
            fval = float(val) if not isinstance(val, (list, tuple, np.ndarray)) else None
        except (TypeError, ValueError):
            fval = None
        if fval is None:
            continue
        if ("voltage" in k or k.endswith("ht") or "high_tension" in k or "beam_energy" in k) \
                and "voltage_V" not in out:
            out["voltage_V"] = fval * (1e3 if fval < 1e3 else 1.0)
        elif ("spherical" in k or k.endswith("/cs") or "cs(mm)" in k or k.endswith("cs_mm")) \
                and "c_3_0" not in out:
            out["c_3_0"] = fval * 1e6 if fval < 100 else fval      # mm -> nm
        elif "defocus" in k and "defocus" not in out:
            out["defocus"] = fval
        elif "magnification" in k and "magnification" not in out:
            out["magnification"] = fval
        elif ("exposure" in k or "integration_time" in k) and "exposure_s" not in out:
            out["exposure_s"] = fval
    return out


def _load_tiff(path):
    import tifffile
    with tifffile.TiffFile(path) as t:
        arr = t.asarray()
        meta = {}
        if t.imagej_metadata:
            meta["imagej"] = dict(t.imagej_metadata)
        page = t.pages[0]
        tags = {tg.name: tg.value for tg in page.tags.values()}
        meta["tags"] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in tags.items()}
        px = None
        unit = (meta.get("imagej", {}) or {}).get("unit")
        if "XResolution" in tags:
            num, den = tags["XResolution"] if isinstance(tags["XResolution"], tuple) else (tags["XResolution"], 1)
            if num:
                px = den / num
                if unit is None:
                    ru = tags.get("ResolutionUnit")
                    unit = {2: "inch", 3: "cm"}.get(int(ru) if ru is not None else 0)
                    if unit == "cm":
                        px, unit = px * 1e7, "nm"
                    elif unit == "inch":
                        px, unit = None, None    # DPI: not physical
        if t.ome_metadata:
            m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"(?:\s+PhysicalSizeXUnit="([^"]+)")?', t.ome_metadata)
            if m:
                px, unit = float(m.group(1)), m.group(2) or "µm"
        md = {}
        if px:
            md["pixel_size_nm"] = _to_nm(px, unit)
        desc = tags.get("ImageDescription", "")
        if isinstance(desc, str):
            for m in re.finditer(r"defocus\s*[=:]\s*([-+0-9.eE]+)\s*(nm|um|µm)?", desc, re.I):
                md.setdefault("defoci_nm", []).append(_to_nm(m.group(1), m.group(2) or "nm"))
        md.update({k: v for k, v in _search_metadata(meta).items() if k not in md})
        md["raw"] = meta
        return arr, md


def _load_rsciio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".dm3", ".dm4"):
        from rsciio.digitalmicrograph import file_reader
    elif ext == ".emd":
        from rsciio.emd import file_reader
    elif ext == ".ser":
        from rsciio.tia import file_reader
    elif ext in (".mrc",):
        from rsciio.mrc import file_reader
    else:
        raise ValueError(ext)
    items = file_reader(path)
    imgs, md = [], {}
    for it in items:
        d = np.asarray(it["data"])
        if d.ndim < 2:
            continue
        imgs.append(d if d.ndim == 3 else d[None])
        ax = it.get("axes", [])
        for a in ax:
            if a.get("navigate") is False or a.get("name", "").lower() in ("x", "width"):
                if "pixel_size_nm" not in md and a.get("scale"):
                    md["pixel_size_nm"] = _to_nm(a["scale"], a.get("units"))
        found = _search_metadata({"metadata": it.get("metadata", {}),
                                  "original_metadata": it.get("original_metadata", {})})
        if "defocus" in found:
            md.setdefault("defoci_nm", []).append(found.pop("defocus"))
        for k, v in found.items():
            md.setdefault(k, v)
        md.setdefault("raw", {})[f"item{len(imgs)}"] = it.get("metadata", {})
    return np.concatenate(imgs, 0), md


def _get_path(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _frwr_parameters(tags, n, pixel_nm):
    """ImagingParameters from the tags written by the FRWR DigitalMicrograph plugins.

    FRWR/FocalSeries: offsetFoc (defocus of the central frame, nm), DeltaFoc (defocus step, nm),
    stepsFoc (number of frames), focusScale, optional nonlinear exponent (Eq. 1 of the FRWR
    manual: df_n = offset + step sign(n - n0) |n - n0|^m); FRWR/Info: AstigMag (nm),
    AstigAngle (deg), alpha (mrad, 1/e semi-angle), Delta (nm), ObjectiveAperture (mrad);
    Microscope Info: Voltage (V), Cs (mm).
    """
    from .params import ImagingParameters
    fs = _get_path(tags, "FRWR", "FocalSeries") or {}
    info = _get_path(tags, "FRWR", "Info") or {}
    mi = tags.get("Microscope Info", {}) if isinstance(tags, dict) else {}
    if not fs and not info and not mi:
        return None, {}
    hints = {}
    md = {}
    if mi.get("Voltage") or info.get("HighVoltage"):
        md["voltage_V"] = float(mi.get("Voltage") or float(info["HighVoltage"]) * 1e3)
    if mi.get("Cs") is not None:
        md["c_3_0"] = float(mi["Cs"]) * 1e6
    if fs.get("DeltaFoc") is not None and fs.get("offsetFoc") is not None:
        m = None
        for k in ("nonLinearity", "nonlinearity", "NonLinearity", "nonLinearFocus", "power"):
            if k in fs:
                m = float(fs[k])
        m = m or 1.0
        n0 = (n - 1) // 2
        k = np.arange(n) - n0
        md["defoci_nm"] = list(float(fs["offsetFoc"]) + float(fs["DeltaFoc"]) * np.sign(k) * np.abs(k) ** m)
        md["defocus_series"] = {"offset": float(fs["offsetFoc"]), "step": float(fs["DeltaFoc"]), "m": m, "n0": n0}
        if fs.get("stepsFoc") and int(float(fs["stepsFoc"])) != n:
            hints["warning"] = f"stepsFoc = {fs['stepsFoc']} but the stack has {n} images"
    ab = {}
    if "c_3_0" in md:
        ab["c_3_0"] = md["c_3_0"]
    if info.get("AstigMag"):
        A = float(info["AstigMag"])
        phiA = -np.deg2rad(float(info.get("AstigAngle", 0.0)))    # same convention as the FRWR3 .cdf
        ab["c_1_2_a"] = A * np.cos(2 * phiA)
        ab["c_1_2_b"] = A * np.sin(2 * phiA)
    conv = float(info["alpha"]) * 1e-3 / np.sqrt(2) if info.get("alpha") is not None else 0.0
    ap = float(info["ObjectiveAperture"]) * 1e-3 if info.get("ObjectiveAperture") else None
    if ap is not None:
        md["objective_aperture_edge_mrad"] = float(info.get("ObjectiveApertureEdge", 0.0))
    for src, dst in (("Microscope", "microscope"), ("Specimen", "specimen"), ("Operator", "operator")):
        v = info.get(src) or mi.get(src) or _get_path(mi, "Info", src)
        if v:
            hints[dst] = str(v)
    if fs.get("Acquisition Time"):
        hints["acquisition_time"] = str(fs["Acquisition Time"])
    for k, dst in (("Zoomfactor_mm", "zoom_per_mm"), ("Rot_MuRad_mm", "rotation_per_mm")):
        if fs.get(k) is not None:
            hints[dst] = float(fs[k])
    if fs.get("expTime") is not None:
        md["exposure_s"] = float(fs["expTime"])
    if "defoci_nm" not in md or pixel_nm is None:
        return None, dict(md, nxem_hints=hints, aberrations=ab, convergence_rad=conv)
    p = ImagingParameters(voltage_V=md.get("voltage_V", 200e3), pixel_size_nm=pixel_nm,
                          defoci_nm=np.array(md["defoci_nm"]), aberrations=ab,
                          focal_spread_nm=float(info.get("Delta", 0.0) or 0.0), convergence_rad=conv,
                          aperture_rad=ap)
    md.update(pixel_size_nm=pixel_nm, nxem_hints=hints)
    return p, md


def _load_dm(path):
    """DM3/DM4 via the built-in reader (memory-mapped), RosettaSciIO as fallback."""
    from .dmreader import read_dm
    try:
        images, tags = read_dm(path)
    except Exception:
        arr, md = _load_rsciio(path)
        return arr, md, None
    if not images:
        raise ValueError("no image data found in the DM file")
    im = max(images, key=lambda i: i["data"].size)                 # skip thumbnails
    arr = np.asarray(im["data"])
    md = {"source_dtype": str(arr.dtype)}
    unit = (im["units"][-1] if im["units"] else "") or ""
    pix = None
    if unit.strip() and unit.strip().lower() in _UNIT:
        pix = _to_nm(im["scales"][-1], unit)
        md["pixel_size_nm"] = pix
    else:
        md["uncalibrated"] = True
    n = arr.shape[0] if arr.ndim == 3 else 1
    p, md2 = _frwr_parameters(im["tags"], n, pix)
    md.update(md2)
    found = _search_metadata(im["tags"])
    for k, v in found.items():
        if k == "defocus":
            continue
        md.setdefault(k, v)
    md["raw"] = {"ImageTags": im["tags"], "name": im.get("name", "")}
    return arr, md, p


_IMG_INDEX = re.compile(r"^(.*?)(\d+)\.img$", re.IGNORECASE)


def img_series_files(path):
    """All frames of an indexed .img series (``<prefix><number>.img``) in numeric order.

    For ``.../image_3.img`` this returns ``image_0.img, image_1.img, ...`` of the same folder
    (only files with exactly the same prefix, so ``imageRef_3.img`` belongs to another series).
    A file without a trailing number is returned alone.
    """
    folder, name = os.path.split(os.path.abspath(path))
    m = _IMG_INDEX.match(name)
    if not m:
        return [path]
    prefix = m.group(1)
    out = []
    for f in os.listdir(folder or "."):
        mm = _IMG_INDEX.match(f)
        if mm and mm.group(1) == prefix:
            out.append((int(mm.group(2)), os.path.join(folder, f)))
    return [f for _, f in sorted(out)]


def natural_key(name):
    """Sort key: numbers in file names compare numerically (image_2 < image_10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(str(name)))]


def _load_img(paths):
    """FRWR3/QSTEM .img frames -> (stack, md).  Pixel size from the header (dx, dy in Å);
    the header field t of FRWR3 frames holds the defocus in Å (used as a suggestion)."""
    from .frwr3_io import read_img
    frames, ts, hdr0, notes = [], [], None, []
    for p in paths:
        a, h = read_img(p)
        if h["complex"]:
            a = np.abs(a) ** 2
            notes.append("complex .img: intensity |psi|^2 used")
        frames.append(np.asarray(a, float))
        ts.append(float(h["t"]))
        hdr0 = hdr0 or h
    shp = frames[0].shape
    if any(f.shape != shp for f in frames):
        raise ValueError("the .img frames have different sizes")
    md = {"source_format": "img (FRWR3/QSTEM)", "files": [os.path.basename(p) for p in paths]}
    if hdr0["dx"] > 0:
        md["pixel_size_nm"] = hdr0["dx"] / 10.0
        if abs(hdr0["dy"] - hdr0["dx"]) > 1e-6 * hdr0["dx"]:
            md["pixel_size_y_nm"] = hdr0["dy"] / 10.0
    if len(paths) > 1 and len(set(np.round(ts, 3))) == len(ts):
        md["defoci_nm"] = [t / 10.0 for t in ts]
        md["defocus_source"] = "header field t of the .img files (assumed defocus in Å) — please check"
    if hdr0.get("comment"):
        md["comment"] = hdr0["comment"]
    if notes:
        md["note"] = notes[0]
    return np.stack(frames), md


def load_series(path_or_bytes, filename=None):
    """Load a focal series. Returns (stack[N, ny, nx], metadata dict, params_or_None).

    ``params`` is an ImagingParameters object only for NXem input.
    """
    tmp = None
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        suffix = os.path.splitext(filename or "upload.bin")[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(bytes(path_or_bytes))
        tmp.close()
        path = tmp.name
    else:
        path = str(path_or_bytes)
    ext = os.path.splitext(filename or path)[1].lower()
    try:
        if ext in (".nxs", ".h5", ".hdf5", ".nx5"):
            from .nxem import read_focal_series
            stack, params, gt, recs = read_focal_series(path)
            md = dict(pixel_size_nm=params.pixel_size_nm, voltage_V=params.voltage_V,
                      defoci_nm=list(params.defoci_nm), c_3_0=params.aberrations.get("c_3_0", 0.0))
            return stack, md, params
        if ext in (".tif", ".tiff"):
            arr, md = _load_tiff(path)
        elif ext == ".npy":
            arr, md = np.load(path), {}
        elif ext == ".npz":
            z = np.load(path)
            arr = z[z.files[0]]
            md = {k: z[k].tolist() for k in z.files[1:]}
        elif ext in (".mrc", ".st", ".mrcs"):
            import mrcfile
            with mrcfile.open(path, permissive=True) as m:
                arr = np.asarray(m.data)
                md = {"pixel_size_nm": float(m.voxel_size.x) * 0.1} if m.voxel_size.x else {}
        elif ext == ".img":
            files = img_series_files(path) if tmp is None else [path]
            arr, md = _load_img(files)
            if tmp is not None:
                md["files"] = [os.path.basename(filename or path)]
        elif ext in (".dm3", ".dm4"):
            arr, md, p = _load_dm(path)
            arr = np.array(arr, dtype=float)
            if arr.ndim == 2:
                arr = arr[None]
            if p is not None and len(p.defoci_nm) != len(arr):
                p = None
            return arr, md, p
        else:
            arr, md = _load_rsciio(path)
    finally:
        if tmp is not None:
            os.unlink(tmp.name)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):   # RGB -> gray
        arr = arr[..., :3].mean(-1)
    return arr, md, None


_DEFOCUS_IN_NAME = re.compile(r"(?<![A-Za-z0-9.])[A-Za-z]*?([+-]?\d+(?:\.\d+)?)\s*(nm|um|µm|mum)", re.IGNORECASE)


def defocus_from_name(name):
    """Defocus in nm from a file name such as 'F-1000nm.dm3', 'df_+2.5um.tif' (None if absent)."""
    m = _DEFOCUS_IN_NAME.search(os.path.splitext(os.path.basename(str(name)))[0])
    if not m:
        return None
    v = float(m.group(1))
    return v * (1e3 if m.group(2).lower() in ("um", "µm", "mum") else 1.0)


def load_folder(folder):
    """All single-image files of a folder as one focal series.

    Frames are ordered by the defocus in their names (``F-1000nm.dm3``) if every file has one
    (the defoci are then used), otherwise by name with numbers compared numerically; indexed
    .img series are read as such.  Returns (stack, md)."""
    exts = (".dm3", ".dm4", ".tif", ".tiff", ".img", ".npy", ".mrc")
    names = [f for f in os.listdir(folder) if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))]
    if not names:
        raise ValueError("no image files in the folder")
    dfs = [defocus_from_name(n) for n in names]
    if all(d is not None for d in dfs):
        order = sorted(range(len(names)), key=lambda i: dfs[i])
    else:
        order = sorted(range(len(names)), key=lambda i: natural_key(names[i]))
    stacks, mds = [], []
    for i in order:
        s, md, _ = load_series(os.path.join(folder, names[i])) if not names[i].lower().endswith(".img") else \
            (*_load_img([os.path.join(folder, names[i])]), None)
        if len(s) != 1:
            raise ValueError(f"{names[i]} contains {len(s)} images; load stacks individually")
        stacks.append(s)
        mds.append(md)
    shp = stacks[0].shape[1:]
    if any(s.shape[1:] != shp for s in stacks):
        raise ValueError("the images have different sizes")
    md = {k: v for k, v in mds[0].items() if k not in ("defoci_nm", "raw")}
    md["files"] = [names[i] for i in order]
    if all(d is not None for d in dfs):
        md["defoci_nm"] = [dfs[i] for i in order]
        md["defocus_source"] = "file names"
    else:
        df = [m.get("defoci_nm", [np.nan])[0] if m.get("defoci_nm") else np.nan for m in mds]
        if not all(np.isnan(df)):
            md["defoci_nm"] = df
    return np.concatenate(stacks, 0), md


def load_many(files):
    """Load several single-image files (list of (bytes, name)) into one stack.

    The files are ordered by name with numbers compared numerically (image_2 before image_10)."""
    stacks, mds = [], []
    dfs = [defocus_from_name(n) for _, n in files]
    if all(d is not None for d in dfs):
        files = [f for _, f in sorted(zip(dfs, files), key=lambda t: t[0])]
    else:
        files = sorted(files, key=lambda f: natural_key(f[1]))
    for content, name in files:
        s, md, _ = load_series(content, name)
        stacks.append(s)
        mds.append(md)
    shp = stacks[0].shape[1:]
    if any(s.shape[1:] != shp for s in stacks):
        raise ValueError("images have different sizes")
    md = dict(mds[0])
    df = []
    for m, s in zip(mds, stacks):
        d = m.get("defoci_nm") or ([m["defocus"]] if "defocus" in m else [np.nan])
        df += list(d)[: len(s)] + [np.nan] * max(0, len(s) - len(d))
    md["defoci_nm"] = df
    names_df = [defocus_from_name(n) for _, n in files]
    if all(d is not None for d in names_df) and all(np.isnan(v) for v in df):
        md["defoci_nm"] = names_df
        md["defocus_source"] = "file names"
    return np.concatenate(stacks, 0), md
