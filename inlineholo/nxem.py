# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Read / write focal series and reconstructions in the NeXus NXem format (HDF5).

The files follow the NXem application definition (NeXus definitions as shipped with
pynxtools / pynxtools-em) and pass ``pynx validate`` (``validate_nexus``): all fields that
NXem requires are written, and the fields that only the user can know (sample, instrument,
user, ...) are taken from a metadata dictionary (see :data:`METADATA_FIELDS`), which the
Voila app exposes as editable fields.  :func:`validate` runs the pynxtools validator.

Layout (group names ending in a number are instances of the NXDL ``...ID`` concepts)::

  /entry1                                   NXentry   definition = "NXem", start_time, end_time
    profiling/program1                      NXcs_profiling / NXprogram (inlineholo + version)
    user1                                   NXuser    name, affiliation, email, identifier_orcid
    sample1                                 NXsample  is_simulation, name, atom_types,
                                                      preparation_date, physical_form, ...
    measurement                             NXem_measurement
      instrument                            NXem_instrument   name, location, type = tem
        fabrication                         NXfabrication     vendor, model, serial_number
        programID                           NXprogram         acquisition software (optional)
        ebeam_column                        NXebeam_column
          electron_source                   NXsource  voltage [V], emitter_type, probe
          corrector_cs1                     NXcorrector_cs    applied (True only with a corrector)
            tableau1/c_3_0, c_1_2_a, ...    NXaberration  magnitude [nm] (global residual aberrations)
        optics                              NXem_optical_system
          semi_convergence_angle [rad]      (rms of the Gaussian illumination-tilt distribution)
        inline_holography                   NXcollection (focal_spread [nm], pixel size, MTF, ...)
      event1 ... eventN                     NXem_event_data   (one per focal-series image)
        image1                              NXimage
          image_2d                          NXdata   real (counts), axis_i/axis_j [nm]
        instrument                          NXem_instrument   (per-image metadata)
          optics/defocus [nm]               NXem_optical_system
          ebeam_column/corrector_cs1/tableau1/c_1_0, c_1_2_a, c_1_2_b  NXaberration
    simulation                              NXem_simulation  (optional ground truth)
      results/image1/image_2d               real, imag of the exit wave
    roi1                                    NXroi_process  (the analysed field of view)
      img                                   NXem_img
        image1 ... imageK                   NXimage  imaging_mode = inline_holography (custom)
          reconstruction                    NXprocess  program, version, date, sequence_index
            parameters                      NXparameters  method, label, options (JSON),
                                                          imaging parameters, refined values
            input                           NXnote  the focal series this result comes from
            convergence                     NXdata  cost / R-value per iteration (if any)
          image_2d                          NXdata  real, imag (complex exit wave) or real
                                                    (derived maps: phase, amplitude, background,
                                                    diffusion field, strain, ...)

The reader also accepts files where the whole series is stored as one ``stack_2d``
(``real`` of shape (N, ny, nx)) with ``optics/defocus`` as an array of length N.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os

import h5py
import numpy as np

from . import __version__
from .optics import ABERRATION_NAMES
from .params import ImagingParameters

IMAGING_MODE = "inline_holography"
IDENTIFIER_TYPES = ("ARK", "DOI", "Hdl", "IGSN", "ISNI", "ISSN", "ISSN-L", "ORCID", "PURL", "ROR", "URL", "URN")

# ------------------------------------------------------------------ user metadata
# key: (section, field) -> (label, default, required, help)
METADATA_FIELDS = {
    "experiment": {
        "identifier": ("Experiment identifier", "", False, "e.g. lab-book number or proposal ID"),
        "description": ("Experiment description", "", False, "free text"),
        "start_time": ("Acquisition start (ISO 8601)", "", True,
                       "date/time when the focal series was recorded, e.g. 2026-11-03T14:20:00+01:00"),
        "end_time": ("Acquisition end (ISO 8601)", "", False, "optional"),
        "doi": ("DOI of related publication", "", False, "optional, stored as NXcite"),
    },
    "user": {
        "name": ("Name", "", False, "person who recorded / reconstructed the data"),
        "affiliation": ("Affiliation", "", False, ""),
        "email": ("E-mail", "", False, ""),
        "orcid": ("ORCID", "", False, "e.g. 0000-0002-1825-0097"),
        "role": ("Role", "", False, "e.g. principal_investigator, operator"),
    },
    "sample": {
        "name": ("Sample name", "", False, ""),
        "atom_types": ("Elements (atom_types)", "", True, "comma-separated chemical symbols, e.g. Si, N"),
        "preparation_date": ("Preparation date (ISO 8601)", "", True, "e.g. 2026-10-30T00:00:00+01:00"),
        "physical_form": ("Physical form", "foil", False, "bulk | foil | thin_film | powder"),
        "identifier": ("Sample identifier", "", False, "e.g. sample ID in your ELN / LIMS"),
        "identifier_type": ("Identifier type", "local", False,
                            "ARK, DOI, Hdl, IGSN, ISNI, PURL, ROR, URL, URN or a local scheme (e.g. local)"),
        "thickness_nm": ("Thickness (nm)", "", False, "optional"),
        "description": ("Sample description", "", False, ""),
        "is_simulation": ("Simulated data", False, True, "True for simulated focal series"),
    },
    "instrument": {
        "name": ("Microscope name", "", False, "local name of the instrument"),
        "location": ("Location", "", False, "institute / lab"),
        "vendor": ("Vendor", "", True, "e.g. JEOL, Thermo Fisher Scientific, Hitachi, Zeiss"),
        "model": ("Model", "", True, "e.g. JEM-ARM200F"),
        "serial_number": ("Serial number", "", False, ""),
        "emitter_type": ("Electron source / emitter type", "", True,
                         "cold_field_emission | schottky | thermionic (LaB6, W)"),
        "cs_corrector": ("Cs corrector present", False, False, "True if the aberrations are corrector-measured"),
        "corrector_vendor": ("Corrector vendor", "", False, "e.g. CEOS"),
        "corrector_model": ("Corrector model", "", False, ""),
        "detector": ("Camera / detector", "", False, "e.g. Gatan OneView"),
        "software": ("Acquisition software", "", False, "e.g. DigitalMicrograph 3.5"),
    },
}


def default_metadata():
    return {s: {k: v[1] for k, v in fs.items()} for s, fs in METADATA_FIELDS.items()}


def missing_required(meta):
    """List of (section, key, label) of required metadata that is still empty."""
    out = []
    for s, fs in METADATA_FIELDS.items():
        for k, (label, _, req, _) in fs.items():
            if req and k != "is_simulation" and not str((meta or {}).get(s, {}).get(k, "")).strip():
                out.append((s, k, label))
    return out


def _merge_meta(meta, is_simulation=None):
    m = default_metadata()
    for s, d in (meta or {}).items():
        if s in m and isinstance(d, dict):
            m[s].update({k: v for k, v in d.items() if v is not None})
    if is_simulation is not None:
        m["sample"]["is_simulation"] = bool(is_simulation)
    return m


# ------------------------------------------------------------------ low-level helpers
def _grp(parent, name, nxclass):
    g = parent.require_group(name)
    g.attrs["NX_class"] = nxclass
    return g


def _ds(parent, name, data, units=None, **attrs):
    if name in parent:
        del parent[name]
    if isinstance(data, np.ndarray) and data.size > 4096 and data.dtype.kind in "fciub":
        d = parent.create_dataset(name, data=data, compression="gzip", compression_opts=4, shuffle=True)
    else:
        d = parent.create_dataset(name, data=data)
    if units:
        d.attrs["units"] = units
    for k, v in attrs.items():
        d.attrs[k] = v
    return d


def _image_nxdata(g, shape, pixel, signal_fields, title="", units=None):
    d = _grp(g, "image_2d", "NXdata")
    ny, nx = shape
    for name, (arr, long_name) in signal_fields.items():
        _ds(d, name, arr, units, long_name=long_name)
    d.attrs["signal"] = list(signal_fields)[0]
    d.attrs["axes"] = ["axis_j", "axis_i"]
    d.attrs["axis_j_indices"] = np.uint64(0)
    d.attrs["axis_i_indices"] = np.uint64(1)
    if title:
        _ds(d, "title", title)
    _ds(d, "axis_i", np.arange(nx) * pixel, "nm", long_name="x (nm)")
    _ds(d, "axis_j", np.arange(ny) * pixel, "nm", long_name="y (nm)")
    return d


def _write_aberrations(col, ab: dict, applied=None, meta=None):
    cs = _grp(col, "corrector_cs1", "NXcorrector_cs")
    if applied is not None:
        _ds(cs, "applied", bool(applied))
    if meta and meta["instrument"].get("cs_corrector") and (meta["instrument"].get("corrector_vendor")
                                                           or meta["instrument"].get("corrector_model")):
        fab = _grp(cs, "fabrication", "NXfabrication")
        _ds(fab, "vendor", str(meta["instrument"].get("corrector_vendor") or "unknown"))
        _ds(fab, "model", str(meta["instrument"].get("corrector_model") or "unknown"))
    tab = _grp(cs, "tableau1", "NXprocess")
    for k, v in ab.items():
        a = _grp(tab, k, "NXaberration")
        _ds(a, "magnitude", float(v), "nm")


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _iso(s, fallback=None):
    """Return an ISO-8601 date-time string (dates are completed to midnight UTC)."""
    s = str(s or "").strip()
    if not s:
        return fallback
    try:
        t = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            t = _dt.datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            return s
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return t.isoformat()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size < 10000 else f"array{o.shape}"
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _write_entry_header(f, meta, description):
    f.attrs["default"] = "entry1"
    f.attrs["creator"] = f"inlineholo {__version__}"
    f.attrs["NX_class"] = "NXroot"
    e = _grp(f, "entry1", "NXentry")
    _ds(e, "definition", "NXem")
    ex = meta["experiment"]
    _ds(e, "start_time", _iso(ex.get("start_time"), _now()))
    _ds(e, "end_time", _iso(ex.get("end_time"), _iso(ex.get("start_time"), _now())))
    if ex.get("identifier"):
        _ds(e, "identifier_experiment", str(ex["identifier"]))
    _ds(e, "experiment_description", str(ex.get("description") or description))
    if ex.get("doi"):
        c = _grp(e, "cite1", "NXcite")
        _ds(c, "doi", str(ex["doi"]))
    prof = _grp(e, "profiling", "NXcs_profiling")
    prog = _grp(prof, "program1", "NXprogram")
    _ds(prog, "program", "inlineholo", version=__version__)
    u = meta["user"]
    if any(str(u.get(k, "")).strip() for k in ("name", "affiliation", "email", "orcid")):
        ug = _grp(e, "user1", "NXuser")
        for k in ("name", "affiliation", "email", "role"):
            if str(u.get(k, "")).strip():
                _ds(ug, k, str(u[k]))
        if str(u.get("orcid", "")).strip():
            _ds(ug, "identifier_orcid", str(u["orcid"]).strip(), type="ORCID")
    s = meta["sample"]
    sg = _grp(e, "sample1", "NXsample")
    _ds(sg, "is_simulation", bool(s.get("is_simulation", False)))
    _ds(sg, "atom_types", ", ".join(a.strip() for a in str(s.get("atom_types") or "").replace(";", ",").split(",")
                                    if a.strip()) or "unknown")
    _ds(sg, "preparation_date", _iso(s.get("preparation_date"), _iso(ex.get("start_time"), _now())))
    for k in ("name", "description"):
        if str(s.get(k, "")).strip():
            _ds(sg, k, str(s[k]))
    if str(s.get("physical_form", "")).strip():
        _ds(sg, "physical_form", str(s["physical_form"]))
    if str(s.get("identifier", "")).strip():
        t = str(s.get("identifier_type") or "local").strip()
        known = {x.lower(): x for x in IDENTIFIER_TYPES}
        d = _ds(sg, "identifier_sample", str(s["identifier"]), type=known.get(t.lower(), t))
        if t.lower() not in known:
            d.attrs["type_custom"] = True
    if str(s.get("thickness_nm", "")).strip():
        try:
            _ds(sg, "thickness", float(s["thickness_nm"]), "nm")
        except ValueError:
            pass
    return e


# ------------------------------------------------------------------ writing
def write_focal_series(path, stack, params: ImagingParameters, ground_truth=None,
                       description="focal series for inline electron holography",
                       is_simulation=None, extra_metadata=None, mode="w", metadata=None, dtype=None):
    """Write a focal series (+ imaging parameters, optional ground truth) as an NXem file.

    metadata : dict with the sections of :data:`METADATA_FIELDS` (missing values get defaults;
        required values that are empty are filled with "unknown" so that the file is still
        valid, but should be supplied by the user).
    """
    stack = np.asarray(stack) if dtype is None else np.asarray(stack, dtype=dtype)
    meta = _merge_meta(metadata, is_simulation)
    if meta["sample"].get("is_simulation"):
        im_ = meta["instrument"]
        im_.setdefault("vendor", "")
        for k, v in (("vendor", "simulation"), ("model", "inlineholo.simulate"), ("emitter_type", "simulated")):
            if not str(im_.get(k, "")).strip():
                im_[k] = v
    with h5py.File(path, mode) as f:
        e = _write_entry_header(f, meta, description)
        im = meta["instrument"]
        m = _grp(e, "measurement", "NXem_measurement")
        ins = _grp(m, "instrument", "NXem_instrument")
        _ds(ins, "type", "tem")
        for k in ("name", "location"):
            if str(im.get(k, "")).strip():
                _ds(ins, k, str(im[k]))
        fab = _grp(ins, "fabrication", "NXfabrication")
        _ds(fab, "vendor", str(im.get("vendor") or "unknown"))
        _ds(fab, "model", str(im.get("model") or "unknown"))
        if str(im.get("serial_number", "")).strip():
            _ds(fab, "serial_number", str(im["serial_number"]))
        if str(im.get("software", "")).strip():
            pg = _grp(ins, "program1", "NXprogram")
            sw = str(im["software"]).strip()
            name, _, ver = sw.partition(" ")
            _ds(pg, "program", name, version=ver or "unknown")
        col = _grp(ins, "ebeam_column", "NXebeam_column")
        src = _grp(col, "electron_source", "NXsource")
        _ds(src, "voltage", float(params.voltage_V), "V")
        _ds(src, "emitter_type", str(im.get("emitter_type") or "unknown"))
        _ds(src, "probe", "electron")
        glob = {k: v for k, v in params.aberrations.items() if k != "c_1_0"}
        if glob:
            _write_aberrations(col, glob, applied=bool(im.get("cs_corrector")), meta=meta)
        if str(im.get("detector", "")).strip():
            det = _grp(ins, "detector1", "NXdetector")
            _ds(det, "name", str(im["detector"]))
        opt = _grp(ins, "optics", "NXem_optical_system")
        _ds(opt, "semi_convergence_angle", float(params.convergence_rad), "rad")
        par = _grp(ins, "inline_holography", "NXcollection")
        _ds(par, "semi_convergence_angle_definition",
            "rms (standard deviation per axis) of the Gaussian illumination-tilt distribution")
        _ds(par, "focal_spread", float(params.focal_spread_nm), "nm",
            description="standard deviation of the Gaussian defocus distribution")
        _ds(par, "pixel_size", float(params.pixel_size_nm), "nm")
        if params.aperture_rad is not None:
            _ds(par, "objective_aperture_semi_angle", float(params.aperture_rad), "rad")
        _ds(par, "detector_mtf", json.dumps(params.mtf or {}))
        _ds(par, "c_1_0_global", float(params.aberrations.get("c_1_0", 0.0)), "nm",
            description="defocus added to all images (global defocus offset)")
        _ds(par, "imaging_parameters", params.to_json())
        if extra_metadata:
            _ds(par, "extra_metadata", json.dumps(extra_metadata, default=_json_default))
        for i in range(params.n_images):
            ev = _grp(m, f"event{i + 1}", "NXem_event_data")
            img = _grp(ev, "image1", "NXimage")
            _image_nxdata(img, stack.shape[1:], params.pixel_size_nm, {"real": (stack[i], "intensity")},
                          f"focal series image {i + 1}, C1 = {params.defoci_nm[i]:.3f} nm")
            ins_i = _grp(ev, "instrument", "NXem_instrument")
            oi = _grp(ins_i, "optics", "NXem_optical_system")
            _ds(oi, "defocus", float(params.defoci_nm[i]), "nm")
            ci = _grp(ins_i, "ebeam_column", "NXebeam_column")
            _write_aberrations(ci, {"c_1_0": float(params.defoci_nm[i]),
                                    "c_1_2_a": float(params.astig_a_nm[i]),
                                    "c_1_2_b": float(params.astig_b_nm[i])})
        if ground_truth is not None:
            sim = _grp(e, "simulation", "NXem_simulation")
            sp = _grp(sim, "program1", "NXprogram")
            _ds(sp, "program", "inlineholo.simulate", version=__version__)
            res = _grp(sim, "results", "NXprocess")
            im_ = _grp(res, "image1", "NXimage")
            _image_nxdata(im_, ground_truth.shape, params.pixel_size_nm,
                          {"real": (ground_truth.real, "real part of the exit wave"),
                           "imag": (ground_truth.imag, "imaginary part of the exit wave")},
                          "ground-truth exit wave (object plane)")


def _next_image(f):
    e = f["entry1"]
    roi = _grp(e, "roi1", "NXroi_process")
    img = _grp(roi, "img", "NXem_img")
    k = 1
    while f"image{k}" in img:
        k += 1
    return img, k


def _process_group(im, method, label, params, options, runtime, seq, source_file=None, extra_params=None):
    pr = _grp(im, "reconstruction", "NXprocess")
    _ds(pr, "program", "inlineholo")
    _ds(pr, "version", __version__)
    _ds(pr, "date", _now())
    _ds(pr, "sequence_index", np.uint64(seq))
    pa = _grp(pr, "parameters", "NXparameters")
    _ds(pa, "method", str(method))
    _ds(pa, "label", str(label))
    _ds(pa, "options", json.dumps(options or {}, default=_json_default))
    if params is not None:
        _ds(pa, "imaging_parameters", params.to_json())
    _ds(pa, "runtime_seconds", float(runtime or 0.0))
    for k, v in (extra_params or {}).items():
        if isinstance(v, (dict, list, tuple)) or (isinstance(v, np.ndarray) and v.ndim > 1):
            _ds(pa, k, json.dumps(v, default=_json_default))
        elif isinstance(v, np.ndarray):
            _ds(pa, k, v)
        else:
            _ds(pa, k, v if isinstance(v, (int, float, bool, np.number)) else str(v))
    if source_file is not None:
        n = _grp(pr, "input", "NXnote")
        _ds(n, "type", "application/x-hdf5")
        _ds(n, "file_name", str(source_file))
        _ds(n, "description", "focal series: /entry1/measurement/event*/image1 of this file")
    return pr


def _mode(im):
    _ds(im, "imaging_mode", IMAGING_MODE, custom=True)


def write_map(path, array, params: ImagingParameters, title, kind, source=None, units=None, method="",
              options=None, long_name=None):
    """Append a real-valued map derived from a reconstruction (phase, background, strain, ...)."""
    array = np.asarray(array)
    with h5py.File(path, "a") as f:
        img, k = _next_image(f)
        im = _grp(img, f"image{k}", "NXimage")
        _mode(im)
        pa_extra = {"map_type": kind}
        if source:
            pa_extra["derived_from"] = source
        _process_group(im, method or kind, title, params, options, 0.0, k, extra_params=pa_extra)
        if array.ndim == 2:
            _image_nxdata(im, array.shape, params.pixel_size_nm, {"real": (array, long_name or kind)}, title, units)
        else:
            d = _grp(im, "image_1d", "NXdata")
            _ds(d, "real", array, units, long_name=long_name or kind)
            d.attrs["signal"] = "real"
            d.attrs["axes"] = ["axis_i"]
            d.attrs["axis_i_indices"] = np.uint64(0)
            _ds(d, "axis_i", np.arange(len(array)), None, long_name="index")
            _ds(d, "title", title)
    return f"/entry1/roi1/img/image{k}"


def write_reconstruction(path, rec, params: ImagingParameters, label=None, derived=True):
    """Append a reconstruction (``Reconstruction`` object or complex array) to an NXem file.

    Besides the complex exit wave, everything the reconstruction carries is stored: the cost /
    R-value history, the incoherent background, the arrays in ``rec.extra`` (e.g. the
    Fokker-Planck diffusion field) and, for FRWR3, the refined defoci, alpha, distortions and
    frame offsets.  With ``derived=True`` amplitude and phase maps are added as separate images.
    Returns the HDF5 path of the exit-wave image.
    """
    wave = rec.wave if hasattr(rec, "wave") else np.asarray(rec)
    method = getattr(rec, "method", "unknown")
    label = label or method
    options = dict(getattr(rec, "options", {}) or {})
    options.pop("callback", None)
    extra = getattr(rec, "extra", {}) or {}
    extra_params, maps = {}, []
    for k, v in extra.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            maps.append((k, v))
        elif hasattr(v, "defoci_nm") and hasattr(v, "rvalues"):        # FRWR3Result
            extra_params.update({
                "frwr3_refined_defoci": np.asarray(v.defoci_nm, float),
                "frwr3_alpha": np.asarray(v.alpha_rad, float),
                "frwr3_distortions": np.asarray(v.distortions, float),
                "frwr3_frame_offsets": np.asarray(v.frame_offsets, float),
                "frwr3_wave_shifts": np.asarray(v.wave_shifts, float),
                "frwr3_backend": str(v.backend),
                "frwr3_config": {kk: vv for kk, vv in (v.config or {}).items()},
            })
            if getattr(v, "rvalues", None):
                extra_params["frwr3_rvalues"] = [list(r) if isinstance(r, (list, tuple)) else r for r in v.rvalues]
        elif isinstance(v, (int, float, str, bool, list, tuple, dict)):
            extra_params[k] = v
    with h5py.File(path, "a") as f:
        if "entry1" not in f:
            raise ValueError("write the focal series first (write_focal_series)")
        img, k = _next_image(f)
        im = _grp(img, f"image{k}", "NXimage")
        _mode(im)
        pr = _process_group(im, method, label, params, options, getattr(rec, "runtime_s", 0.0), k,
                            source_file=os.path.basename(path), extra_params=extra_params)
        hist = list(getattr(rec, "history", []) or [])
        if hist:
            cv = _grp(pr, "convergence", "NXdata")
            _ds(cv, "cost", np.asarray(hist, float), None, long_name="cost / R-value")
            _ds(cv, "iteration", np.arange(len(hist)), None, long_name="iteration")
            cv.attrs["signal"] = "cost"
            cv.attrs["axes"] = ["iteration"]
            cv.attrs["iteration_indices"] = np.uint64(0)
        _image_nxdata(im, wave.shape, params.pixel_size_nm,
                      {"real": (wave.real, "real part of the exit wave"),
                       "imag": (wave.imag, "imaginary part of the exit wave")},
                      f"reconstructed exit wave ({label})")
    src = f"/entry1/roi1/img/image{k}"
    if derived:
        write_map(path, np.abs(wave), params, f"amplitude ({label})", "amplitude", src, method=method)
        write_map(path, np.angle(wave), params, f"phase ({label})", "phase", src, method=method,
                  long_name="phase (rad)")
    bg = getattr(rec, "background", None)
    if bg is not None and np.ndim(bg) == 2:
        write_map(path, bg, params, f"incoherent background ({label})", "incoherent_background", src,
                  method=method)
    elif bg is not None and np.ndim(bg) == 1:
        write_map(path, np.asarray(bg, float), params, f"incoherent background per image ({label})",
                  "incoherent_background", src, method=method)
    for name, arr in maps:
        write_map(path, arr, params, f"{name} ({label})", name, src, method=method)
    return src


def write_curve(path, x, curves: dict, params: ImagingParameters, title, kind, axis_name="q", axis_units=None,
                method="", options=None):
    """Append 1-D results (e.g. half-set FRC and its threshold) as an NXimage with image_1d."""
    with h5py.File(path, "a") as f:
        img, k = _next_image(f)
        im = _grp(img, f"image{k}", "NXimage")
        _mode(im)
        _process_group(im, method or kind, title, params, options, 0.0, k, extra_params={"map_type": kind})
        d = _grp(im, "image_1d", "NXdata")
        names = list(curves)
        _ds(d, "real", np.asarray(curves[names[0]], float), None, long_name=names[0])
        for n in names[1:]:
            _ds(d, n, np.asarray(curves[n], float), None, long_name=n)
        d.attrs["signal"] = "real"
        if len(names) > 1:
            d.attrs["auxiliary_signals"] = names[1:]
        d.attrs["axes"] = ["axis_i"]
        d.attrs["axis_i_indices"] = np.uint64(0)
        _ds(d, "axis_i", np.asarray(x, float), axis_units, long_name=f"{axis_name}" + (f" ({axis_units})" if axis_units else ""))
        _ds(d, "title", title)
    return f"/entry1/roi1/img/image{k}"


def write_strain(path, strain, params, g, sigma_px, source=None, label="DIH strain"):
    """Append a dark-field inline holography strain map (eps_gg, dimensionless)."""
    return write_map(path, strain, params, f"{label}: eps_gg along g = {tuple(g)} 1/nm", "strain_gg", source,
                     units=None, method="strain_1d", options={"g_inv_nm": list(g), "sigma_px": sigma_px},
                     long_name="strain eps_gg (relative to the median)")


# ------------------------------------------------------------------ validation
def validate(path, ignore_undocumented=False):
    """Validate a file with pynxtools (as ``pynx validate``).

    Returns (is_valid, problems, warnings_undocumented) where problems are the messages about
    missing required concepts, wrong types, enumerations, ...; None if pynxtools is missing.
    """
    try:
        from pynxtools.dataconverter.validation import validate_hdf_group_against
        from pynxtools.dataconverter import helpers  # noqa: F401
    except ImportError:
        return None, ["pynxtools is not installed (pip install pynxtools pynxtools-em)"], []
    msgs = []

    class H(logging.Handler):
        def emit(self, record):
            msgs.append((record.levelname, record.getMessage()))

    h = H()
    lg = logging.getLogger("pynxtools")
    lg.addHandler(h)
    old = lg.level
    lg.setLevel(logging.INFO)
    try:
        with h5py.File(path, "r") as f:
            ok = True
            for name, g in f.items():
                if isinstance(g, h5py.Group) and g.attrs.get("NX_class") in ("NXentry", b"NXentry"):
                    d = g.get("definition")
                    nxdl = d[()].decode() if d is not None and isinstance(d[()], bytes) else (d[()] if d is not None else None)
                    if nxdl:
                        ok &= bool(validate_hdf_group_against(nxdl, g, path, ignore_undocumented))
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    undoc = [m for lv, m in msgs if "has no documentation" in m]
    problems = [f"{lv}: {m}" for lv, m in msgs if "has no documentation" not in m and "is valid according" not in m
                and lv in ("WARNING", "ERROR", "CRITICAL")]
    return ok, problems, undoc


def file_checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ reading
def _read_tableau(col):
    ab = {}
    for cs_name, cs in col.items():
        if not isinstance(cs, h5py.Group) or cs.attrs.get("NX_class") != "NXcorrector_cs":
            continue
        for _, tab in cs.items():
            if not isinstance(tab, h5py.Group):
                continue
            for k, a in tab.items():
                if k in ABERRATION_NAMES and "magnitude" in a:
                    ab[k] = float(a["magnitude"][()]) * _unit_to_nm(a["magnitude"])
    return ab


def _unit_to_nm(ds):
    u = ds.attrs.get("units", "nm")
    if isinstance(u, bytes):
        u = u.decode()
    return {"nm": 1.0, "m": 1e9, "um": 1e3, "µm": 1e3, "mm": 1e6, "angstrom": 0.1, "A": 0.1,
            "pm": 1e-3}.get(u, 1.0)


def _get(g, path, default=None):
    try:
        v = g[path][()]
        return v.decode() if isinstance(v, bytes) else v
    except KeyError:
        return default


def read_metadata(path, entry="entry1"):
    """User metadata stored in a file (inverse of the metadata argument of write_focal_series)."""
    meta = default_metadata()
    with h5py.File(path, "r") as f:
        e = f[entry]
        ex = meta["experiment"]
        ex["start_time"] = _get(e, "start_time", "") or ""
        ex["end_time"] = _get(e, "end_time", "") or ""
        ex["identifier"] = _get(e, "identifier_experiment", "") or ""
        ex["description"] = _get(e, "experiment_description", "") or ""
        ex["doi"] = _get(e, "cite1/doi", "") or ""
        for k in ("name", "affiliation", "email", "role"):
            meta["user"][k] = _get(e, f"user1/{k}", "") or ""
        meta["user"]["orcid"] = _get(e, "user1/identifier_orcid", "") or ""
        s = meta["sample"]
        for k in ("name", "atom_types", "preparation_date", "physical_form", "description"):
            s[k] = _get(e, f"sample1/{k}", s[k]) or s[k]
        s["identifier"] = _get(e, "sample1/identifier_sample", "") or ""
        if "sample1/identifier_sample" in e:
            t = e["sample1/identifier_sample"].attrs.get("type", "local")
            s["identifier_type"] = t.decode() if isinstance(t, bytes) else t
        th = _get(e, "sample1/thickness", None)
        s["thickness_nm"] = "" if th is None else str(th)
        s["is_simulation"] = bool(_get(e, "sample1/is_simulation", False))
        ins = "measurement/instrument"
        im = meta["instrument"]
        for k in ("name", "location"):
            im[k] = _get(e, f"{ins}/{k}", "") or ""
        for k in ("vendor", "model", "serial_number"):
            im[k] = _get(e, f"{ins}/fabrication/{k}", "") or ""
        im["emitter_type"] = _get(e, f"{ins}/ebeam_column/electron_source/emitter_type", "") or ""
        im["cs_corrector"] = bool(_get(e, f"{ins}/ebeam_column/corrector_cs1/applied", False))
        im["detector"] = _get(e, f"{ins}/detector1/name", "") or ""
        if f"{ins}/program1/program" in e:
            p = e[f"{ins}/program1/program"]
            v = p.attrs.get("version", "")
            im["software"] = f"{_get(e, f'{ins}/program1/program')} {v.decode() if isinstance(v, bytes) else v}".strip()
    for s, d in meta.items():
        for k, v in d.items():
            if v in ("unknown",):
                d[k] = ""
    return meta


def read_focal_series(path, entry="entry1"):
    """Return (stack, params, ground_truth_or_None, reconstructions dict).

    The reconstructions dict maps the label of every stored complex exit wave to the wave
    (derived real-valued maps are returned by :func:`read_maps`).
    """
    with h5py.File(path, "r") as f:
        e = f[entry]
        m = e["measurement"]
        ins = m.get("instrument")
        voltage = float(_get(ins, "ebeam_column/electron_source/voltage", 300e3)) if ins else 300e3
        glob = _read_tableau(ins["ebeam_column"]) if ins and "ebeam_column" in ins else {}
        alpha = float(_get(ins, "optics/semi_convergence_angle", 0.0)) if ins else 0.0
        fs = float(_get(ins, "inline_holography/focal_spread", 0.0)) if ins else 0.0
        pix = _get(ins, "inline_holography/pixel_size", None) if ins else None
        mtf = json.loads(_get(ins, "inline_holography/detector_mtf", "{}") or "{}") if ins else {}
        off = _get(ins, "inline_holography/c_1_0_global", None) if ins else None
        if off is None and ins:
            off = _get(ins, "inline_holography/global_defocus_offset", 0.0)
        off = float(off or 0.0)
        aperture = _get(ins, "inline_holography/objective_aperture_semi_angle", None) if ins else None
        events = sorted([k for k in m if k.startswith("event")], key=lambda s: int(s[5:] or 0))
        imgs, df, aa, ab_ = [], [], [], []
        for k in events:
            ev = m[k]
            imname = [n for n in ev if isinstance(ev[n], h5py.Group) and ev[n].attrs.get("NX_class") == "NXimage"]
            for n in imname:
                im = ev[n]
                for dname in ("image_2d", "stack_2d"):
                    if dname in im:
                        arr = im[dname]["real"][()]
                        if pix is None and "axis_i" in im[dname]:
                            ax = im[dname]["axis_i"]
                            pix = float(ax[1] - ax[0]) * _unit_to_nm(ax)
                        if arr.ndim == 2:
                            arr = arr[None]
                        imgs.append(arr)
            ab = _read_tableau(ev["instrument"]["ebeam_column"]) if "instrument" in ev and "ebeam_column" in ev["instrument"] else {}
            d = None
            if "instrument" in ev and "optics" in ev["instrument"] and "defocus" in ev["instrument"]["optics"]:
                ds = ev["instrument"]["optics"]["defocus"]
                d = np.atleast_1d(ds[()]) * _unit_to_nm(ds)
            elif "c_1_0" in ab:
                d = np.atleast_1d(ab["c_1_0"])
            n_here = imgs[-1].shape[0] if imgs else 1
            df.extend(list(d) if d is not None else [np.nan] * n_here)
            aa.extend([ab.get("c_1_2_a", 0.0)] * n_here)
            ab_.extend([ab.get("c_1_2_b", 0.0)] * n_here)
        stack = np.concatenate(imgs, 0)
        glob["c_1_0"] = off
        params = ImagingParameters(voltage_V=voltage, pixel_size_nm=float(pix or 1.0),
                                   defoci_nm=np.array(df), aberrations=glob,
                                   astig_a_nm=np.array(aa), astig_b_nm=np.array(ab_),
                                   focal_spread_nm=fs, convergence_rad=alpha, mtf=mtf or None,
                                   aperture_rad=None if aperture is None else float(aperture))
        gt = None
        if "simulation" in e and "results/image1/image_2d" in e["simulation"]:
            d = e["simulation/results/image1/image_2d"]
            gt = d["real"][()] + 1j * d["imag"][()]
        recs = {}
        if "roi1/img" in e:
            for k in sorted(e["roi1/img"], key=lambda s: int(s[5:]) if s[5:].isdigit() else 0):
                im = e["roi1/img"][k]
                if "image_2d" in im and "imag" in im["image_2d"]:
                    lab = (_get(im, "reconstruction/parameters/label", None)
                           or _get(im, "reconstruction/label", k))
                    recs[lab] = im["image_2d/real"][()] + 1j * im["image_2d/imag"][()]
    return stack, params, gt, recs


def read_maps(path, entry="entry1"):
    """All real-valued maps and curves stored under roi1/img: {label: (map_type, array)}."""
    out = {}
    with h5py.File(path, "r") as f:
        e = f[entry]
        if "roi1/img" not in e:
            return out
        for k, im in e["roi1/img"].items():
            for dn in ("image_2d", "image_1d"):
                if dn in im and "imag" not in im[dn]:
                    lab = _get(im, "reconstruction/parameters/label", k)
                    kind = _get(im, "reconstruction/parameters/map_type", "")
                    out[lab] = (kind, im[f"{dn}/real"][()])
    return out
