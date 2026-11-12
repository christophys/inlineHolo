# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2008-2026 Christoph T. Koch
"""FRWR3 - Full-Resolution Wave Reconstruction, Python/GPU port of C. T. Koch's C++ code.

This module re-implements the C++ program ``frwr3`` (2019 source code) faithfully,
loop by loop, but keeps everything in memory (no intermediate files) and runs on
NumPy, CuPy or PyTorch (CPU/GPU, see :mod:`inlineholo.backend`).

Algorithm (Koch, Micron 2008; Ultramicroscopy 2014; Ozsoy-Keskinbora et al. 2022)
---------------------------------------------------------------------------------
1. *Trial initialisation* (``initWaveTrial``): the image closest to focus is the
   reference; neighbours are located in their (larger) frames by a chi^2 search, a
   2-image reconstruction (``init loops 2``) is followed by 3-image reconstructions
   (``init loops 3``) adding the plane on either side; the best start is kept.
2. *Progressive reconstruction* (``reconstructAll``): the set of planes grows by one on
   each side per stage; each stage runs ``intermediate loops`` (+ ``phase predict
   loops``) of the flux-preserving projection algorithm.  On "align" iterations
   (after 0, 3, 6, ... loops, at most every ``distortion frequency``) the defoci are
   refined, the simulated images are re-registered to the data (``max shift``), the
   experimental frames are re-extracted from the full images using simulated
   reference images (drift, rotation, magnification, anisotropic zoom, spiral and
   pincushion distortion as functions of defocus), and alpha may be fitted.
3. *Final loops* with optional gradient flipping (``apply GF``), TIE phase prediction
   (``phase predict threshold``), incoherent background, and optional *residue loops*.

Projection step (``waveFunction::project``): propagate the exit wave to each plane
(hard objective aperture, C1, two-fold astigmatism, Cs), form the intensity
|psi|^2 (+ incoherent background), convolve it with the spatial-coherence envelope
exp[-(alpha theta |fa + Cs theta^2| pi/lambda)^2] and the detector MTF
(flux-preserving partial coherence), compare with the data (R = sum|I-D|/sum D),
update the amplitudes ``|psi| <- |psi| + feedback (sqrt D - sqrt I)``, propagate back and
average.  The phase is accumulated (unwrapped) as phi <- phi + feedback * arg(<psi> e^-i phi).

Configuration
-------------
All parameters use the *same names as the FRWR3 control file* (``print level``,
``fit defocus``, ``GF beta``, ``zoomFactor_mm``, ...).  They can be passed as a YAML
file, an frwr3 ``.conf`` file or a dict; see :data:`DEFAULTS` and
``examples/frwr3_lowres2.yaml``.  Additional keys of this implementation are listed
in :data:`EXTENSIONS` (backend, precision, full aberration set, temporal coherence,
sub-pixel shifts, initial wave, ...).

Differences to the 2019 C++ code (all documented, switchable):
* the FFTW r2c/c2r plans are executed correctly and the compensating 1-pixel x-offset
  in the chi^2 search is therefore not applied (see ``frwr3_io`` / docs);
* ``apply MTF: flag wGauss wExp ratio`` (format described in the example .conf) is
  accepted in addition to ``MTF: flag aExp wExp aGauss wGauss`` (format parsed by
  the C++ code);
* ``fit reference points`` (disabled in the C++ code) is implemented as a smooth
  displacement field estimated by local chi^2 searches on an NwarpX x NwarpY grid.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backend import get_backend
from .optics import wavelength_nm, chi_polar
from .params import ImagingParameters

# ----------------------------------------------------------------------------- config
#: FRWR3 control-file keys with their C++ default values (units as in the .conf file)
DEFAULTS = {
    "mode": "FluxPreserving",           # General | HighRes | LowRes | FluxPreserving
    "print level": 1,
    "save level": 0,                    # >=2: keep wave snapshots in memory
    "feedback parameter": 1.0,
    "high NA mode": 0,
    "refractive Index": 1.0,
    # gradient flipping
    "apply GF": 0,
    "GF flipping frequency": 2,
    "GF beta": 0.95,
    "GF signal fraction": 0.8,
    "GF min signal fraction": 0.2,
    "GF threshold": 0.02,
    "GF length scale (nm)": 11.0,
    "GF order": 1,
    "GF adjust fraction": 1,
    "GF 3rd order fraction": 0.21,
    # wave size / phase prediction
    "padding": 0.0,
    "phase predict threshold": 0.0,
    "low intensity threshold": 1.0,
    "low resolution threshold": 0.02,
    # alignment
    "shift trials": 1,
    "max shift": 50,
    "max shift global": 0,
    "max defocus": 0.0,
    "start frame": -1,
    "stop frame": -1,
    "first frame": -1,
    "warping points X": 4,
    "warping points Y": 4,
    "fit distortions": 0,
    "fit rotation": 1,
    "fit magnification": 1,
    "fit intensity scale": 0,
    "keep central plane": 0,
    "fit defocus": 2,
    "enforce positive phase": 0,
    "allow phase wrapping": 0,
    "fit incoherent background": [0, 0.1, 0.98],
    "MTF": [0, 0.5576, 4.6954, 0.3953, 476.918],
    "apply MTF": None,                  # [flag, wGauss, wExp, ratio] (alternative format)
    "fit alpha": [0, 7],
    "fit reference points": [0, 0],
    "use TIE": 0,
    "TIE low resolution (nm)": 500.0,
    "TIE high resolution (nm)": 1.0,
    "TIE align loops": 2,
    "TIE re-align loops": 1,
    "TIE threshold": 1.0,
    "enforce equal defocus steps": 0,
    "defocus offset": None,
    "defocus step": None,
    "defocus scale": None,
    "bandpass filter low": 0.0,
    "bandpass filter high": 0.3,
    "exponent": 0.4,
    "chi2 weight": 1.0,
    "Nedge": 30,
    "Nedge for smoothing": 30,
    "intermediate loops": 9,
    "final loops": 200,
    "residue loops": 40,
    "init loops 2": 5,
    "init loops 3": 3,
    "phase predict loops": 0,
    "distortion loops": 1,
    "distortion magnitude": 12.0,
    "distortion frequency": 50,
    "smoothen phase": 0.0,
    "smoothen amplitude": 0.0,
    "alpha_mrad": None,
    "zoomFactor_mm": [0.0, 0.0, 0.0],
    "rot_rad_mm": [0.0, 0.0, 0.0],
    "zoomX_mm": [0.0, 0.0, 0.0],
    "zoomY_mm": [0.0, 0.0, 0.0],
    "spiral_mm": [0.0, 0.0, 0.0],
    "pincush_mm": [0.0, 0.0, 0.0],
    "zoomFactor_mm range": 0.0,
    "rot_rad_mm range": 0.0,
    "zoomX_mm range": 0.0,
    "zoomY_mm range": 0.0,
    "spiral_kpix_mm range": 0.0,
    "pincushion_kpix_mm range": 0.0,
    # per-image lists (one value per frame)
    "defocus": None,                    # nm
    "rotation": None,                   # rad
    "zoom": None,
    "zoomX": None,
    "zoomY": None,
    "spiral": None,
    "pincushion": None,
    "alpha": None,                      # mrad
}

#: keys of this implementation that do not exist in the C++ program
EXTENSIONS = {
    "backend": "auto",                  # auto | numpy | cupy | torch
    "precision": "auto",                # auto | float32 | float64
    "aberrations": "frwr3",             # frwr3 (C1, A1, C3) | full (all ImagingParameters aberrations)
    "focal spread envelope": 0,         # 1: damp the wave with the temporal-coherence envelope
    "subpixel shifts": 0,               # 1: refine wave shifts to 1/10 pixel (Fourier shift)
    "distortion refine": 0,             # 0: C++ distortion search; N > 0: N bracketed/halving refinements
    "start with all planes": 0,         # 1: skip the 2/3-plane start, reconstruct with all frames at once
    "initial wave": "flat",             # flat | <method name> | ndarray (passed via API)
    "data normalized": 1,               # 1: images already normalized (no zoom^2 intensity scaling)
    "objective aperture (mrad)": None,  # overrides params.aperture_rad
    "objective aperture edge (mrad)": 0.1,
    "objective aperture offset X (mrad)": 0.0,
    "objective aperture offset Y (mrad)": 0.0,
    "roi": None,                        # [top, left, bottom, right] in pixels of the full frames
    "keep history": 1,
}

_MTF_OFF = [0, 0.5576, 4.6954, 0.3953, 476.918]

#: one-line documentation of every key (used by :func:`default_yaml`)
KEY_DOC = {
    "mode": "FluxPreserving (FRWR, default) | General (no envelope during defocus refinement)",
    "print level": "0 silent, 1 R-values, 2 + per-image info, 3 debug",
    "save level": ">= 2 keeps wave snapshots (every 50 final loops) in memory; nothing is written to disk",
    "feedback parameter": "amplitude/phase feedback beta (1 = full projection)",
    "high NA mode": "1: exact propagator 2 df sqrt(n^2-theta^2) (light optics)",
    "refractive Index": "refractive index for high NA mode",
    "apply GF": "1: gradient flipping in the final loops (Ozsoy-Keskinbora 2022)",
    "GF flipping frequency": "flip every N final loops",
    "GF beta": "flipped gradients are multiplied by -beta",
    "GF signal fraction": "target fraction of (small) gradient components that are flipped",
    "GF min signal fraction": "lower limit when the fraction is adapted",
    "GF threshold": "initial threshold (fraction of the max gradient), adapted automatically",
    "GF length scale (nm)": "flipped phase used below 1/length (Gaussian weight), original above",
    "GF order": "1 gradient, 2 second derivatives, 3 mixed",
    "GF adjust fraction": "1: reduce the signal fraction when R increases",
    "GF 3rd order fraction": "fraction of flips done with order 3",
    "padding": "wave is (1+padding) x ROI; data constraint only inside the ROI",
    "phase predict threshold": "max phase update of the TIE phase prediction (0 = off)",
    "low intensity threshold": "intensity floor for the division in the TIE phase prediction",
    "low resolution threshold": "CTF value defining the low-frequency cut of the phase prediction",
    "shift trials": "number of candidate positions tried in the initialisation",
    "max shift": "search radius (px) for drift / wave-shift alignment (0 = no alignment)",
    "max shift global": "(read but unused by frwr3)",
    "max defocus": "planes with |df - df_ref| larger than this (nm) are excluded (0 = all)",
    "start frame": "first frame used (-1 = 0)",
    "stop frame": "last frame used (-1 = last)",
    "first frame": "reference frame (-1: frame closest to focus)",
    "warping points X": "grid of reference points (fit reference points)",
    "warping points Y": "grid of reference points (fit reference points)",
    "fit distortions": "1: fit rotation/zoom/zoomX/zoomY/spiral/pincushion of each frame",
    "fit rotation": "allow rotation fit",
    "fit magnification": "allow magnification fit",
    "fit intensity scale": "(read but unused by frwr3)",
    "keep central plane": "re-read the reference plane only when >= this many planes are used",
    "fit defocus": "0 never, 1 final loops, 2 after the 3-loop, 3 in the 3-loop, 4 after the first 2-loop, 5 always",
    "enforce positive phase": "1: negative phases are damped",
    "allow phase wrapping": "1: phase is not unwrapped/accumulated",
    "fit incoherent background": "[flag, feedback, damping]",
    "MTF": "[flag, aExp, wExp, aGauss, wGauss]: MTF = aG exp(-wG u^2) + aE exp(-wE u), u = q*dx",
    "apply MTF": "[flag, wGauss, wExp, ratio] (alternative format, see example .conf)",
    "fit alpha": "[flag, min. number of planes]",
    "fit reference points": "[flag, min. number of planes] (inlineholo: smooth displacement field)",
    "use TIE": "1: start from a TIE phase (doTIEGuess)",
    "TIE low resolution (nm)": "TIE low-frequency filter",
    "TIE high resolution (nm)": "TIE high-frequency filter",
    "TIE align loops": "crude TIE alignment loops",
    "TIE re-align loops": "TIE re-alignment loops",
    "TIE threshold": "intensity floor for the TIE division",
    "enforce equal defocus steps": "1: defoci = central + j*step (defocus scale)",
    "defocus offset": "nm (with 'defocus step' when no list is given)",
    "defocus step": "nm",
    "defocus scale": "1 linear, 2 quadratic, ... series",
    "bandpass filter low": "low-frequency damping of the chi^2 alignment (Nyquist = 0.5)",
    "bandpass filter high": "high-frequency damping of the chi^2 alignment",
    "exponent": "1 = chi^2; < 1 = exponent of the Fourier amplitude in the correlation",
    "chi2 weight": "1 chi^2, 0 cross correlation",
    "Nedge": "edge (px) excluded in wave-shift alignment and TIE edge blending",
    "Nedge for smoothing": "edge (px) blended to the mean before alignment",
    "intermediate loops": "loops per stage of the progressive reconstruction",
    "final loops": "loops with all planes",
    "residue loops": "loops with residue accumulation (C++ default 40)",
    "init loops 2": "loops of the 2-plane initialisation",
    "init loops 3": "loops of the 3-plane initialisation",
    "phase predict loops": "additional phase-prediction loops per stage",
    "distortion loops": "repetitions of the distortion fit",
    "distortion magnitude": "step (px at the frame edge) for the distortion fit",
    "distortion frequency": "re-read/re-align every N loops",
    "smoothen phase": "Gaussian smoothing (nm) of the phase at re-reads (0: aperture limit)",
    "smoothen amplitude": "Gaussian smoothing (nm) of the amplitude at re-reads (0: aperture limit)",
    "alpha_mrad": "illumination semi-angle (1/e) in mrad; default sqrt(2) x params.convergence_rad",
    "zoomFactor_mm": "zoom per mm defocus [linear, quadratic, cubic]",
    "rot_rad_mm": "rotation (rad) per mm defocus",
    "zoomX_mm": "anisotropic zoom per mm defocus",
    "zoomY_mm": "anisotropic zoom per mm defocus",
    "spiral_mm": "spiral distortion (rad/kpixel) per mm defocus",
    "pincush_mm": "pincushion (1/kpixel) per mm defocus",
    "zoomFactor_mm range": "allowed deviation of the fitted zoom from the polynomial (per mm)",
    "rot_rad_mm range": "allowed deviation of the fitted rotation",
    "zoomX_mm range": "allowed deviation",
    "zoomY_mm range": "allowed deviation",
    "spiral_kpix_mm range": "allowed deviation",
    "pincushion_kpix_mm range": "allowed deviation",
    "defocus": "list of defoci (nm), one per frame (default: params.defoci_nm)",
    "rotation": "list: rotation per frame (rad)",
    "zoom": "list: zoom per frame",
    "zoomX": "list per frame",
    "zoomY": "list per frame",
    "spiral": "list per frame",
    "pincushion": "list per frame",
    "alpha": "list: alpha per frame (mrad)",
    "backend": "[inlineholo] auto | numpy | cupy | torch (GPU if available)",
    "precision": "[inlineholo] auto | float32 | float64",
    "aberrations": "[inlineholo] frwr3 (C1, A1, C3 as in frwr3) | full (all aberrations of the parameters)",
    "focal spread envelope": "[inlineholo] 1: temporal-coherence envelope on the wave (frwr3 ignores Delta)",
    "subpixel shifts": "[inlineholo] 1: sub-pixel wave shifts (parabolic chi^2 peak + Fourier shift)",
    "distortion refine": "[inlineholo] 0: one 3-point parabola per distortion parameter (C++); N > 0: N "
                         "refinements with convexity check and halved step (robust for small distortions)",
    "start with all planes": "[inlineholo] 1: no 2-/3-plane initialization and no step-wise inclusion of planes; "
                             "all frames are used from the first loop (for pre-aligned series and/or a given "
                             "start wave, e.g. the result of the FRWR alignment)",
    "initial wave": "[inlineholo] flat | name of another inlineholo method used as start",
    "data normalized": "[inlineholo] 1: no zoom^2 intensity scaling when warping",
    "objective aperture (mrad)": "[inlineholo] objective aperture radius (default params.aperture_rad or none)",
    "objective aperture edge (mrad)": "[inlineholo] cosine edge of the aperture",
    "objective aperture offset X (mrad)": "[inlineholo]",
    "objective aperture offset Y (mrad)": "[inlineholo]",
    "roi": "[inlineholo] [top, left, bottom, right] in pixels of the full frames",
    "keep history": "[inlineholo] keep the log/R-values in the result",
}


def default_yaml(path=None):
    """YAML text listing every key with its default value and a comment."""
    import yaml
    lines = ["# FRWR3 / inlineholo configuration - all keys with defaults",
             "# keys are identical to the frwr3 control file; [inlineholo] marks extensions"]
    for k, v in list(DEFAULTS.items()) + list(EXTENSIONS.items()):
        vv = yaml.safe_dump(v, default_flow_style=True).strip()
        if vv.endswith("..."):
            vv = vv[:-3].strip()
        lines.append(f"{k}: {vv}" + (f"   # {KEY_DOC[k]}" if k in KEY_DOC else ""))
    text = "\n".join(lines) + "\n"
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def _vals(v):
    """Turn '1 0.1 0.9' / [1, .1, .9] / 3 into a list of numbers (strings kept)."""
    if v is None:
        return None
    if isinstance(v, str):
        out = []
        for tok in v.replace(",", " ").split():
            try:
                out.append(float(tok))
            except ValueError:
                out.append(tok)
        return out
    if isinstance(v, (list, tuple, np.ndarray)):
        return [float(x) if not isinstance(x, str) else x for x in v]
    return [v]


def load_config(cfg=None, raw_only=False, **overrides):
    """Merge a config (YAML path, .conf path, YAML/conf text, dict or None) with the defaults.

    Keys are the FRWR3 names; a trailing colon is accepted.  Returns a plain dict
    (``raw_only=True``: only the keys given explicitly).
    """
    raw = {}
    if cfg is None:
        pass
    elif isinstance(cfg, dict):
        raw = dict(cfg)
    else:
        text = None
        path = str(cfg)
        try:
            with open(path) as fh:
                text = fh.read()
        except (OSError, ValueError):
            text = path
            path = ""
        if path.endswith(".conf") or ("%" in text and ":" in text and not path.endswith((".yaml", ".yml"))):
            from .frwr3_io import read_conf
            raw = {k: (v if len(v) != 1 else v[0]) for k, v in read_conf(text).items()}
        else:
            import yaml
            raw = yaml.safe_load(text) or {}
            # allow a top-level 'frwr3:' section
            if isinstance(raw, dict) and "frwr3" in raw and isinstance(raw["frwr3"], dict):
                raw = raw["frwr3"]
    raw.update(overrides)
    out = {} if raw_only else dict(DEFAULTS)
    if not raw_only:
        out.update(EXTENSIONS)
    for k, v in raw.items():
        k2 = str(k).strip().rstrip(":").strip()
        out[k2] = v
    return out


def conf_to_yaml(conf_path, yaml_path=None):
    """Convert an frwr3 .conf control file into YAML (same key names, comments kept)."""
    lines = ["# FRWR3 configuration (converted from %s)" % conf_path,
             "# Same keys as the frwr3 control file; '#' starts a comment."]
    with open(conf_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            body, _, com = line.partition("%")
            com = com.strip().lstrip("%").strip()
            if not body.strip():
                if com:
                    lines.append("# " + com)
                else:
                    lines.append("")
                continue
            if body[0].isspace() or ":" not in body:
                lines.append("#   " + (body.strip() + " " + com).strip())
                continue
            k, v = body.split(":", 1)
            v = v.strip()
            toks = v.split()
            if len(toks) > 1:
                vv = "[" + ", ".join(toks) + "]"
            else:
                vv = v
            s = f'{k.strip()}: {vv}'
            if com:
                s += "   # " + com
            lines.append(s)
    text = "\n".join(lines) + "\n"
    if yaml_path:
        with open(yaml_path, "w") as fh:
            fh.write(text)
    return text


def write_yaml(cfg: dict, path, only_changed=True):
    import yaml
    d = {k: v for k, v in cfg.items()
         if not only_changed or (k in DEFAULTS and DEFAULTS[k] != v) or (k in EXTENSIONS and EXTENSIONS[k] != v)}
    with open(path, "w") as fh:
        yaml.safe_dump(d, fh, sort_keys=False, default_flow_style=None)


# ----------------------------------------------------------------------------- helpers
DIST_ROT, DIST_ZOOM, DIST_ZOOMX, DIST_ZOOMY, DIST_SPIRAL, DIST_PINCUSH = range(6)
DIST_NAMES = ("rotation", "zoom", "zoomX", "zoomY", "spiral", "pincushion")


def _find_minimum(chi2, val0, delta):
    """Parabola through 3 points at val0-delta, val0, val0+delta (bicubint.cpp)."""
    d2 = (chi2[2] - 2.0 * chi2[1] + chi2[0]) / delta if delta != 0 else 0.0
    d1 = 0.5 * (chi2[2] + chi2[1]) - 0.5 * (chi2[1] + chi2[0])
    val = val0 - d1 / d2 if abs(d2) > 1e-8 else val0
    val = min(max(val, val0 - delta), val0 + delta)
    return val


@dataclass
class Plane:
    """State of one focal plane (``imageData`` in the C++ code)."""
    index: int
    defocus_img: float                  # nm, defocus assigned when the plane was created (img->defocus)
    defocus0: float = 0.0               # nm, defocus of the reference plane at creation
    alpha: float = 0.0                  # rad (FRWR3 convention: 1/e semi-angle)
    distortions: np.ndarray = field(default_factory=lambda: np.zeros(6))
    frame_off: list = field(default_factory=lambda: [0, 0])   # x, y (top-left in full frame)
    offset: list = field(default_factory=lambda: [0, 0])
    wave_shift: list = field(default_factory=lambda: [0.0, 0.0])
    shift_xy: list = field(default_factory=lambda: [0, 0])
    shifts: list = field(default_factory=list)   # [(x, y, chi2), ...] best positions
    shift_min: float = 0.0
    data: object = None
    residue: object = None
    residue_off: list = field(default_factory=lambda: [0, 0])
    fit_rot: int = 0
    fit_mag: int = 0
    fit_ref_points: int = 0
    warp: object = None                 # optional displacement field (extension)
    shift_sub: list = None              # sub-pixel frame offset (x, y) from the chi^2 parabola


@dataclass
class FRWR3Result:
    wave: np.ndarray                    # complex exit wave on the ROI
    wave_padded: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray                   # unwrapped (accumulated) phase on the ROI
    background: np.ndarray | None
    defoci_nm: np.ndarray
    alpha_rad: np.ndarray               # FRWR3 alpha (1/e semi-angle)
    distortions: np.ndarray             # (n, 6)
    frame_offsets: np.ndarray           # (n, 2) x, y
    wave_shifts: np.ndarray             # (n, 2)
    rvalues: list
    log: list
    config: dict
    runtime_s: float
    backend: str
    snapshots: list = field(default_factory=list)
    frame_offsets_subpixel: np.ndarray | None = None   # (n, 2) x, y with sub-pixel precision
    aligned: np.ndarray | None = None        # (n, sizeY, sizeX) aligned, distortion-corrected data (= image_#.img)
    aligned_full: np.ndarray | None = None   # (n, NY, NX) full frames warped + shifted to the reference geometry
    aligned_valid: np.ndarray | None = None  # (NY, NX) bool, pixels covered by all aligned full frames
    reference_plane: int = -1
    stopped: bool = False
    pad_offset: tuple = (0, 0)               # (py, px) of the ROI inside wave_padded


# ----------------------------------------------------------------------------- engine
class FRWR3:
    """In-memory FRWR3 engine.

    Parameters
    ----------
    frames : (n, NY, NX) full focal-series images (the ROI is cut out of these; larger
        frames allow drift and distortion correction).
    params : ImagingParameters (voltage, pixel size, defoci, aberrations, alpha, aperture)
    config : YAML/conf path, text or dict with FRWR3 keys (see DEFAULTS, EXTENSIONS)
    """

    def __init__(self, frames, params: ImagingParameters, config=None, roi=None, callback=None,
                 init_wave=None, **overrides):
        self.cfg = c = load_config(config, **overrides)
        prec = c.get("precision", "auto")
        self.B = B = get_backend(c.get("backend", "auto"), None if prec in (None, "auto") else prec)
        self.p = params
        self.callback = callback
        self.work_done = 0
        self.init_wave_user = init_wave
        self.log = []
        self.rvalues = []
        self.snapshots = []
        self.print_level = int(c["print level"])
        frames = np.asarray(frames, float)
        if frames.ndim == 2:
            frames = frames[None]
        self.nz, self.NY, self.NX = frames.shape
        self.frames_np = frames
        self.frames = [B.asarray(f) for f in frames]
        self.dx = float(params.pixel_size_nm)
        self.lam = float(wavelength_nm(params.voltage_V))
        ab = dict(params.aberrations)
        self.Cs = float(ab.get("c_3_0", 0.0))
        a, b = ab.get("c_1_2_a", 0.0), ab.get("c_1_2_b", 0.0)
        self.ast_mag = float(np.hypot(a, b))
        self.ast_phi = float(0.5 * np.arctan2(b, a))
        self.full_ab = str(c.get("aberrations", "frwr3")).lower() == "full"
        self.focal_spread = float(params.focal_spread_nm or 0.0)
        # alpha: FRWR3 uses the 1/e semi-angle (= sqrt(2) x rms tilt per axis)
        if c.get("alpha_mrad") is not None:
            self.alpha0 = float(_vals(c["alpha_mrad"])[0]) * 1e-3
        else:
            self.alpha0 = float(np.sqrt(2) * (params.convergence_rad or 0.0))
        # objective aperture
        if c.get("objective aperture (mrad)") is not None:
            ap = float(c["objective aperture (mrad)"]) * 1e-3
        elif params.aperture_rad:
            ap = float(params.aperture_rad)
        else:
            ap = 2.0 * np.sqrt(2) * self.lam / (2 * self.dx)      # no aperture: beyond corners
        self.ap = ap
        self.ap_edge = float(c.get("objective aperture edge (mrad)", 0.1) or 0.0) * 1e-3
        self.ap_offx = float(c.get("objective aperture offset X (mrad)", 0.0) or 0.0) * 1e-3
        self.ap_offy = float(c.get("objective aperture offset Y (mrad)", 0.0) or 0.0) * 1e-3
        # ROI
        if roi is None:
            roi = c.get("roi")
        if roi is None:
            roi = (0, 0, self.NY, self.NX)
        self.roi = [int(v) for v in roi]               # top, left, bottom, right
        self.sizeY = self.roi[2] - self.roi[0]
        self.sizeX = self.roi[3] - self.roi[1]
        self._parse()
        self._init_wave_grid()

    # ------------------------------------------------------------------ parsing
    def _parse(self):
        c = self.cfg
        g = lambda k: c.get(k, DEFAULTS.get(k))
        f1 = lambda k: float(_vals(g(k))[0])
        i1 = lambda k: int(round(float(_vals(g(k))[0])))
        mode = str(g("mode")).lower()
        self.flux_pres = not mode.startswith("general")
        self.feedback = f1("feedback parameter")
        self.highNA = i1("high NA mode")
        self.n_refr = f1("refractive Index")
        self.applyGF = i1("apply GF")
        self.GF_freq = max(1, i1("GF flipping frequency"))
        self.GF_beta = f1("GF beta")
        self.GF_frac = f1("GF signal fraction")
        self.GF_minfrac = f1("GF min signal fraction")
        self.GF_thresh = f1("GF threshold")
        self.GF_length = f1("GF length scale (nm)")
        self.GF_order = i1("GF order")
        self.GF_adjust = i1("GF adjust fraction")
        self.GF_o3frac = f1("GF 3rd order fraction")
        self.padding = f1("padding")
        self.phaseThresh = f1("phase predict threshold")
        self.imgThresh = f1("low intensity threshold")
        self.threshCTFlow = f1("low resolution threshold")
        self.SHIFT_LEN = i1("shift trials")
        self.MAX_SHIFT = i1("max shift")
        md = f1("max defocus")
        self.maxDefocus = abs(md)
        self.startFrame = i1("start frame")
        self.stopFrame = i1("stop frame")
        self.firstPlane = i1("first frame")
        self.NwarpX, self.NwarpY = i1("warping points X"), i1("warping points Y")
        self.fitDistortion = i1("fit distortions")
        self.fitRotation = i1("fit rotation")
        self.fitMagnification = i1("fit magnification")
        self.keepCentralPlane = i1("keep central plane")
        self.allowDefFitting = i1("fit defocus")
        self.enforcePositivePhase = i1("enforce positive phase")
        self.allowPhaseWrapping = i1("allow phase wrapping")
        if self.allowPhaseWrapping:
            self.enforcePositivePhase = 0
        bg = _vals(g("fit incoherent background")) + [0.1, 0.98]
        self.fitIncIntensity, self.feedbackIncBkgrnd, self.dampIncInt = int(bg[0]), float(bg[1]), float(bg[2])
        # MTF
        self.applyMTF = 0
        self.MTFaExp, self.MTFwExp, self.MTFaGauss, self.MTFwGauss = 0.5576, 4.6954, 0.3953, 476.918
        m = c.get("MTF")
        if m is not None:
            mv = _vals(m) + [0.0] * 5
            self.applyMTF = int(mv[0])
            aE, wE, aG, wG = abs(mv[1]), abs(mv[2]), abs(mv[3]), abs(mv[4])
            n = aE + aG
            if n > 0:
                aE, aG = aE / n, aG / n
            self.MTFaExp, self.MTFwExp, self.MTFaGauss, self.MTFwGauss = aE, wE, aG, wG
        am = c.get("apply MTF")
        if am is not None and (m is None or _vals(m)[0] == 0):
            av = _vals(am) + [0.5, 0.5, 1.0]
            self.applyMTF = int(av[0])
            wG, wE, ratio = float(av[1]), float(av[2]), float(av[3])
            if ratio >= 1000:
                aG, aE = 1.0, 0.0
            elif ratio <= 1e-4:
                aG, aE = 0.0, 1.0
            else:
                aG, aE = 1.0 / (1 + ratio), ratio / (1 + ratio)
            # MTF = aG exp(-u^2/wG^2) + aE exp(-u/wE) with u = q*dx (Nyquist = 0.5)
            self.MTFaGauss, self.MTFwGauss = aG, (1.0 / wG ** 2 if wG > 0 else 0.0)
            self.MTFaExp, self.MTFwExp = aE, (1.0 / wE if wE > 0 else 0.0)
        if self.fitDistortion == 0:
            self.fitRotation = self.fitMagnification = 0
        fa = _vals(g("fit alpha")) + [7]
        self.fitAlpha, self.fitAlphaMinImages = int(fa[0]), int(fa[1])
        self.allowAlphaFitting = 2
        self.globalAlpha = 1
        fr = _vals(g("fit reference points")) + [0, 0]
        self.fitWarp, self.fitWarpMinImages = int(fr[0]), int(fr[1])
        self.fitReferencePointsFlag = 0
        self.useTIE = i1("use TIE")
        self.kTIELow = 1.0 / f1("TIE low resolution (nm)")
        self.kTIEHigh = 1.0 / f1("TIE high resolution (nm)")
        self.TIEloops = i1("TIE align loops")
        self.TIErealignLoops = i1("TIE re-align loops")
        self.TIEthresh = f1("TIE threshold")
        self.enforceEqualDefSteps = i1("enforce equal defocus steps")
        self.kCutoffLow = f1("bandpass filter low")
        self.kCutoffHigh = f1("bandpass filter high")
        self.expon = f1("exponent")
        self.chi2weight = f1("chi2 weight")
        self.Nedge = i1("Nedge")
        self.NedgeSmooth = i1("Nedge for smoothing")
        self.reconLoops = i1("intermediate loops")
        self.reconLoopsFinal = i1("final loops")
        self.reconLoopsResidue = i1("residue loops")
        self.initLoops2 = i1("init loops 2")
        self.initLoops3 = i1("init loops 3")
        self.phasePredictLoops = i1("phase predict loops")
        if self.phaseThresh == 0:
            self.phasePredictLoops = 0
        self.distortionLoops = i1("distortion loops")
        self.distortionMagnitude = f1("distortion magnitude")
        self.distortionRefine = int(c.get("distortion refine", 0) or 0)
        self.startAllPlanes = int(float(_vals(c.get("start with all planes", 0) or 0)[0]))
        self.distortionLoopFrequency = max(1, i1("distortion frequency"))
        self.phaseSmoothing = f1("smoothen phase")
        self.amplitudeSmoothing = f1("smoothen amplitude")
        self.dist_poly = np.zeros((6, 3))
        for j, k in enumerate(("rot_rad_mm", "zoomFactor_mm", "zoomX_mm", "zoomY_mm", "spiral_mm", "pincush_mm")):
            v = (_vals(g(k)) or [0.0]) + [0.0, 0.0, 0.0]
            self.dist_poly[j] = [float(x) for x in v[:3]]
        self.dist_range = np.zeros(6)
        for j, k in enumerate(("rot_rad_mm range", "zoomFactor_mm range", "zoomX_mm range", "zoomY_mm range",
                               "spiral_kpix_mm range", "pincushion_kpix_mm range")):
            self.dist_range[j] = float(_vals(g(k))[0])
        if self.fitRotation == 0:
            self.dist_range[DIST_ROT] = 0.0
        if self.fitMagnification == 0:
            self.dist_range[DIST_ZOOM] = 0.0
        # defocus values
        df = np.asarray(self.p.defoci_nm, float).copy()
        if len(df) != self.nz:
            raise ValueError("params.defoci_nm must have one value per frame")
        if g("defocus") is not None:
            df = np.asarray(_vals(g("defocus"))[:self.nz], float)
        dscale = g("defocus scale")
        if g("defocus step") is not None and g("defocus") is None:
            off = float(_vals(g("defocus offset"))[0]) if g("defocus offset") is not None else 0.0
            step = float(_vals(g("defocus step"))[0])
            sc = float(_vals(dscale)[0]) if dscale is not None else 1.0
            cp = (self.nz - 1) // 2
            j = np.arange(self.nz) - cp
            df = off + (np.sign(j) * np.abs(j) ** sc if sc > 1 else j) * step
        self.defocusScale = float(_vals(dscale)[0]) if dscale is not None else 1.0
        self.defocus = df
        self.vectors = {}
        for k in ("rotation", "zoom", "zoomX", "zoomY", "spiral", "pincushion", "alpha"):
            if g(k) is not None:
                self.vectors[k] = np.asarray(_vals(g(k))[:self.nz], float)
        self.normalized = int(c.get("data normalized", 1))
        self.subpixel = int(c.get("subpixel shifts", 0))
        self.temporal_env = int(c.get("focal spread envelope", 0))
        self.keep_history = int(c.get("keep history", 1))
        self.save_level = i1("save level")
        self.centralPlane = (self.nz - 1) // 2

    # ------------------------------------------------------------------ grids
    def _init_wave_grid(self):
        B = self.B
        pad = self.padding
        if pad <= 0:
            self.Sx, self.Sy = self.sizeX, self.sizeY
        else:
            self.Sx = int(np.floor(self.sizeX * (1.0 + pad)))
            self.Sy = int(np.floor(self.sizeY * (1.0 + pad)))
        self.px = (self.Sx - self.sizeX) // 2
        self.py = (self.Sy - self.sizeY) // 2
        lam, dx = self.lam, self.dx
        tx = lam * np.fft.fftfreq(self.Sx, dx)
        ty = lam * np.fft.fftfreq(self.Sy, dx)
        TX, TY = np.meshgrid(tx, ty)
        self.thX, self.thY = TX, TY
        self.th2_np = TX ** 2 + TY ** 2
        self.phi_np = np.arctan2(TY, TX)
        apx, apy = TX + self.ap_offx, TY + self.ap_offy
        ap2 = apx ** 2 + apy ** 2
        self.hard_ap = B.asarray((ap2 <= self.ap ** 2).astype(float))
        r0 = self.ap - self.ap_edge / 2
        rmax = r0 + self.ap_edge
        soft = np.zeros_like(ap2)
        inside = ap2 <= rmax ** 2
        soft[inside] = 1.0
        edge = inside & (ap2 > r0 ** 2)
        if self.ap_edge > 0:
            soft[edge] = 0.5 * (1 + np.cos(np.pi / self.ap_edge * (np.sqrt(ap2[edge]) - r0)))
        self.soft_ap = B.asarray(soft)
        # convAperture (C++ applies a hard cut at r0 + edge, the soft scale is unused)
        self.conv_ap_r = rmax
        # half-plane grid for intensity convolutions
        txr = lam * np.fft.rfftfreq(self.Sx, dx)
        # C++: ix < centerX ? ix : ix - sizeX -> last column negative for even sizes
        cx = (self.Sx + self.Sx % 2) // 2
        ixs = np.arange(len(txr))
        txr = lam / (dx * self.Sx) * np.where(ixs < cx, ixs, ixs - self.Sx)
        TXr, TYr = np.meshgrid(txr, ty)
        self.th2r = TXr ** 2 + TYr ** 2
        self.thr = np.sqrt(self.th2r)
        self.phir = np.arctan2(TYr, TXr)
        mtfScale = dx / lam
        MTFwG = mtfScale ** 2 * self.MTFwGauss
        MTFwE = mtfScale * self.MTFwExp
        if self.applyMTF:
            mtf = (self.MTFaGauss * np.exp(-MTFwG * self.th2r) if MTFwG > 0 else self.MTFaGauss)
            mtf = mtf + (self.MTFaExp * np.exp(-MTFwE * self.thr) if MTFwE > 0 else self.MTFaExp)
            self.mtf_r = np.asarray(mtf, float) * np.ones_like(self.th2r)
        else:
            self.mtf_r = None
        self._prop_cache = {}
        self._env_cache = {}
        # wave state
        S = (self.Sy, self.Sx)
        self.exitAmp = B.ones(S)
        self.exitPhase = B.zeros(S)
        self.exitWave = B.zeros(S, True)
        self.exitRecip = False
        self.avgWave = B.zeros(S, True)
        self.w = B.zeros(S, True)
        self.intensity = B.zeros(S)
        self.cut = None
        self.incInt = B.zeros(S)
        self.incIntDelta = B.zeros(S)
        self.imageCount = 0
        self.Rvalue = 0.0
        self.alignFlag = 0
        self.fitAlphaFlag = 0
        self.doPhasePrediction = 0
        self.finalLoop = 0
        self.defocusStep = 1.0
        self.wave_frame_off = [0, 0]
        # base polar coordinates of the full frames (distortions)
        x0, y0 = (self.NX - 1) / 2.0, (self.NY - 1) / 2.0
        X, Y = np.meshgrid(np.arange(self.NX) - x0, np.arange(self.NY) - y0)
        self._fr = np.sqrt(X ** 2 + Y ** 2)
        self._fphi = np.arctan2(Y, X)
        self._fx0, self._fy0 = x0, y0
        self._warp_cache = {}

    # ------------------------------------------------------------------ printing
    def _say(self, level, msg):
        if self.keep_history:
            self.log.append(msg)
        if self.print_level >= level:
            print(msg, flush=True)

    # ------------------------------------------------------------------ optics
    def _chi_code(self, defocus, half=False):
        """Return the propagation phase factor exp(i chi_code) = exp(-i chi) on the wave grid."""
        key = (round(float(defocus), 6),)
        if key in self._prop_cache:
            return self._prop_cache[key]
        lam = self.lam
        th2, phi = self.th2_np, self.phi_np
        pil = np.pi / lam
        if self.highNA:
            chi_c = 2.0 * defocus * pil * np.sqrt(np.maximum(self.n_refr ** 2 - th2, 0))
        elif self.full_ab:
            ab = dict(self.p.aberrations)
            ab["c_1_0"] = float(defocus) + float(ab.get("c_1_0", 0.0)) * 0
            chi_c = -chi_polar(np.sqrt(th2), phi, ab, lam)
        else:
            fa = defocus * pil
            if self.ast_mag != 0:
                fa = fa + self.ast_mag * pil * np.cos(2 * (phi - self.ast_phi))
            chi_c = -(fa + 0.5 * self.Cs * pil * th2) * th2
        P = np.exp(1j * chi_c)
        if self.temporal_env and self.focal_spread > 0:
            q2 = th2 / lam ** 2
            P = P * np.exp(-0.5 * (np.pi * lam * self.focal_spread * q2) ** 2)
        N = self.Sx * self.Sy
        ap = self.B.to_numpy(self.hard_ap)
        # forward factor (includes the N of the unnormalized C++ IFFT) and backward factor
        Pd = (self.B.asarray(P * ap * N, True), self.B.asarray(np.conj(P) * ap / N, True))
        if len(self._prop_cache) > 64:
            self._prop_cache.clear()
        self._prop_cache[key] = Pd
        return Pd

    def _envelope(self, defocus, alpha):
        key = (round(float(defocus), 6), float(alpha))
        if key in self._env_cache:
            return self._env_cache[key]
        lam = self.lam
        pil = np.pi / lam
        th2, th, phi = self.th2r, self.thr, self.phir
        if alpha != 0:
            if self.full_ab:
                from .optics import chi_gradient_theta
                ab = dict(self.p.aberrations)
                ab["c_1_0"] = float(defocus)
                gx, gy = chi_gradient_theta(th, phi, ab, lam)
                env = np.exp(-(alpha ** 2) * (gx ** 2 + gy ** 2) / 4.0)
            else:
                fa = defocus * pil
                if self.ast_mag != 0:
                    fa = fa + self.ast_mag * pil * np.cos(2 * (phi - self.ast_phi))
                e = alpha * th * np.abs(fa + 0.5 * self.Cs * pil * th2)
                env = np.exp(-e ** 2)
        else:
            env = np.ones_like(th2)
        if self.mtf_r is not None:
            env = env * self.mtf_r
        env = np.array(env, float)
        env[0, -1] = 0.0                  # C++: dg[sizeXpadC-1] = 0
        E = self.B.asarray(env)
        if len(self._env_cache) > 256:
            self._env_cache.clear()
        self._env_cache[key] = E
        return E

    # ------------------------------------------------------------------ wave bookkeeping
    def copy_amp_phase_to_exit(self):
        B = self.B
        self.exitWave = self.exitAmp * B.exp(1j * self.exitPhase) if self.B.name != "torch" else \
            self.exitAmp * (B.cos(self.exitPhase) + 1j * B.sin(self.exitPhase))
        self.exitRecip = False

    def fft_exit(self):
        if not self.exitRecip:
            self.exitWave = self.B.fft2(self.exitWave) / (self.Sx * self.Sy)
            self.exitRecip = True

    def reset_avg(self):
        self.avgWave = self.B.zeros((self.Sy, self.Sx), True)
        self.Rvalue = 0.0

    def propagate(self, defocus):
        self.fft_exit()
        Pf, _ = self._chi_code(defocus)
        self.w = self.B.ifft2(self.exitWave * Pf)

    def back_propagate(self, defocus):
        _, Pb = self._chi_code(defocus)
        self.w = self.B.fft2(self.w) * Pb                   # stays in reciprocal space

    def get_intensity(self):
        I = self.w.real ** 2 + self.w.imag ** 2
        if self.fitIncIntensity:
            I = I + self.incInt
        self.intensity = I

    def convolute_espat(self, img: Plane, defocus):
        if self.fitAlphaFlag and img is not None:
            self.fit_alpha(img, defocus)
        alpha = img.alpha
        if alpha == 0 and not self.applyMTF:
            return
        E = self._envelope(defocus, alpha)
        B = self.B
        I = B.irfft2(B.rfft2(self.intensity) * E, (self.Sy, self.Sx))
        self.intensity = B.maximum(I, 0.0)

    def simulate_image(self, img: Plane, defocus):
        self.propagate(defocus)
        self.get_intensity()
        self.convolute_espat(img, defocus)
        self.sim_defocus = defocus
        self.wave_frame_off = list(img.frame_off)

    def get_cut(self, arr):
        if self.padding <= 0:
            return arr
        return arr[self.py:self.py + self.sizeY, self.px:self.px + self.sizeX]

    # ------------------------------------------------------------------ chi^2 search
    def _kgrid(self, ny, nx):
        iy = np.arange(ny)
        ky = np.where(iy < ny // 2, iy, ny - iy) / ny
        kx = np.arange(nx // 2 + 1) / nx
        return ky[:, None] ** 2 + kx[None, :] ** 2

    def _chi2_map(self, big, small_int, mask_shape, filt, expon, chi2weight, sumSmall):
        B = self.B
        ny, nx = big.shape
        N = nx * ny
        S = B.zeros((ny, nx))
        sy, sx = small_int.shape
        S[:sy, :sx] = small_int
        FS = B.rfft2(S)
        FB = B.rfft2(big)
        scale = 1.0 / (sumSmall * N)
        if chi2weight == 1:
            M = B.zeros((ny, nx))
            M[:mask_shape[0], :mask_shape[1]] = 1.0
            FM = B.rfft2(M)
            FB2 = B.rfft2(big * big)
            C = scale * (B.conj(FM) * FB2 - 2.0 * B.conj(FS) * FB)
        else:
            C = -scale * B.conj(FS) * FB
        C[:, -1] = 0
        if expon != 1:
            mag = B.abs(C)
            C = C * B.where(mag > 0, B.maximum(mag, 1e-300) ** (expon - 1.0), 0.0)
        if filt is not None:
            C = C * filt
        C[0, 0] = C[0, 0] + 1.0
        return B.irfft2(C, (ny, nx)) * N

    def find_small_in_big_whole(self, big, small, expon, chi2weight, damp, kLow, Nedge, maxShift):
        """findSmallInBigWholeImage (equal-size images); returns (sx, sy, chi2min)."""
        B = self.B
        ny, nx = big.shape
        m = Nedge + maxShift
        if maxShift <= 0:
            # C++: empty search window -> sx2 = sy2 = 0
            return float(-m), float(-m), 3.4e34
        inner = small[m:ny - m, m:nx - m]
        sumS = B.item(B.sum(inner * inner))
        if sumS <= 0:
            return 0, 0, 0.0
        kLow2 = kLow * 0.5 * (nx + ny)
        kLow22 = kLow2 * kLow2 / 4.0
        filt = None
        if damp != 0 or kLow2 > 0:
            k2 = self._kgrid(ny, nx)
            mod = np.exp(-(damp * damp) * k2)
            lo = k2 < kLow22
            if kLow22 > 0:
                mod[lo] *= (1.0 - np.exp(-k2[lo] / kLow22))
            filt = B.asarray(mod)
        chi2 = self._chi2_map(big, inner, (ny - 2 * m, nx - 2 * m), filt, expon, chi2weight, sumS)
        win = B.to_numpy(chi2[Nedge:Nedge + 2 * maxShift, Nedge:Nedge + 2 * maxShift])
        k = int(np.argmin(win))
        sy2, sx2 = divmod(k, win.shape[1])
        minval = float(win.flat[k]) + sumS
        sx, sy = float(sx2 + Nedge - m), float(sy2 + Nedge - m)
        if self.subpixel:
            sx, sy = self._subpixel(win, sx2, sy2, sx, sy)
        return sx, sy, minval

    @staticmethod
    def _subpixel(win, ix, iy, sx, sy):
        def para(a, b, c):
            d = a - 2 * b + c
            return 0.0 if d <= 0 else 0.5 * (a - c) / d
        if 0 < ix < win.shape[1] - 1:
            sx += para(win[iy, ix - 1], win[iy, ix], win[iy, ix + 1])
        if 0 < iy < win.shape[0] - 1:
            sy += para(win[iy - 1, ix], win[iy, ix], win[iy + 1, ix])
        return round(sx * 10) / 10, round(sy * 10) / 10

    def find_small_in_big(self, big, small, expon, chi2weight, damp, kLow, PosX, PosY, Nedge, NS, shiftLen):
        """findSmallInBig: search the (smaller) reference ``small`` in ``big`` around (PosX, PosY)."""
        B = self.B
        NY, NX = big.shape
        sy, sx = small.shape
        if Nedge <= 0:
            return [(float(PosX), float(PosY), 0.0)] * max(1, shiftLen)
        kLow2 = kLow * 0.5 * (NX + NY)
        kLow22 = kLow2 * kLow2
        if NX - 2 * Nedge <= sx:
            Nedge = (NX - sx) // 2
        if NY - 2 * Nedge <= sy:
            Nedge = (NY - sy) // 2
        if sx - 2 * NS + PosX >= NX:
            PosX = NX - 1 - (sx - 2 * NS)
        if sy - 2 * NS + PosY >= NY:
            PosY = NY - 1 - (sy - 2 * NS)
        PosX, PosY = max(PosX, 1), max(PosY, 1)
        inner = small[NS:sy - NS, NS:sx - NS]
        sumS = B.item(B.sum(inner * inner))
        if sumS <= 0:
            return [(float(PosX), float(PosY), 0.0)] * max(1, shiftLen)
        filt = None
        if damp != 0:
            k2 = self._kgrid(NY, NX)
            mod = np.exp(-(damp * damp) * k2)
            lo = k2 < kLow22
            if kLow2 > 0:
                mod[lo] *= 0.5 * (1 - np.cos(np.pi * np.sqrt(k2[lo]) / kLow2))
            filt = B.asarray(mod)
        chi2 = self._chi2_map(big, inner, (sy - 2 * NS, sx - 2 * NS), filt, expon, chi2weight, sumS)
        chi2 = B.to_numpy(chi2)
        ys = np.arange(PosY - Nedge, PosY + Nedge + 1)
        xs = np.arange(PosX - Nedge, PosX + Nedge + 1)
        ys = ys[(ys >= 0) & (ys < NY - (sy - 2 * NS))]
        xs = xs[(xs >= 0) & (xs < NX - (sx - 2 * NS))]
        iy0, ix0 = max(PosY - Nedge, 0), max(PosX - Nedge, 0)
        ladder = [(float(ix0), float(iy0), 1e20 + float(chi2[iy0, ix0]))]
        cur = ladder[0][2]
        if len(ys) and len(xs):
            W = chi2[np.ix_(ys, xs)].ravel()
            run = np.minimum.accumulate(np.concatenate([[cur], W]))
            new = np.nonzero(W < run[:-1])[0]
            for k in new:
                iy, ix = divmod(k, len(xs))
                ladder.append((float(xs[ix]), float(ys[iy]), float(W[k])))
        ladder = ladder[::-1][:max(1, shiftLen)]
        # sub-pixel position of the best minimum (3-point parabolas of the chi^2 map); stored for
        # the aligned output (the reconstruction itself uses integer shifts, as in the C++ code)
        bx, by = int(ladder[0][0]), int(ladder[0][1])
        sub = [0.0, 0.0]
        for ax, (c0, cm, cp) in enumerate(((chi2[by, bx], chi2[by, bx - 1] if bx > 0 else None,
                                            chi2[by, bx + 1] if bx + 1 < chi2.shape[1] else None),
                                           (chi2[by, bx], chi2[by - 1, bx] if by > 0 else None,
                                            chi2[by + 1, bx] if by + 1 < chi2.shape[0] else None))):
            if cm is not None and cp is not None:
                den = cm - 2 * c0 + cp
                if den > 0:
                    sub[ax] = float(np.clip(0.5 * (cm - cp) / den, -0.5, 0.5))
        self._last_sub = sub
        while len(ladder) < shiftLen:
            ladder.append((0.0, 0.0, 0.0))
        return ladder

    # ------------------------------------------------------------------ image utilities
    def smoothen_edge2(self, img, Nedge):
        """smoothenImageEdge2: clamp >= 0 and blend the edges to the global mean."""
        B = self.B
        img = B.maximum(img, 0.0)
        if Nedge <= 0:
            return img
        img = B.copy(img)
        avg = B.mean(img)
        ny, nx = img.shape
        Ne = min(Nedge, nx, ny)
        s = 0.5 * (1.0 + np.cos(np.pi * np.arange(Ne) / max(Nedge - 1, 1)))
        sx = B.asarray(s)[None, :]
        img[:, :Ne] = sx * avg + (1 - sx) * img[:, :Ne]
        sr = B.asarray(s[::-1].copy())[None, :]
        img[:, nx - Ne:] = sr * avg + (1 - sr) * img[:, nx - Ne:]
        sy_ = B.asarray(s)[:, None]
        img[:Ne, :] = sy_ * avg + (1 - sy_) * img[:Ne, :]
        sry = B.asarray(s[::-1].copy())[:, None]
        img[ny - Ne:, :] = sry * avg + (1 - sry) * img[ny - Ne:, :]
        return img

    def _smoothen_image(self, img: Plane):
        """imageData::smoothenImage (only without padding)."""
        if self.padding > 0:
            return
        img.data = self.smoothen_edge2(img.data, self.NedgeSmooth)
        if img.residue is not None:
            B = self.B
            d = img.data
            m = d > img.residue
            img.residue = B.where(m, img.residue, d)
            img.data = B.where(m, d - img.residue, 0.0)

    def conv_gaussian(self, arr, kc):
        """imageData::convGaussian: multiply the spectrum by exp(-k^2/kc^2) (k in 1/nm)."""
        B = self.B
        ny, nx = arr.shape
        ky = np.fft.fftfreq(ny, self.dx)
        kx = np.fft.rfftfreq(nx, self.dx)
        k2 = ky[:, None] ** 2 + kx[None, :] ** 2
        F = B.asarray(np.exp(-k2 / kc ** 2))
        return B.irfft2(B.rfft2(arr) * F, (ny, nx))

    def conv_aperture(self, arr):
        """imageData::convAperture (hard cut at aperture radius + edge/2)."""
        B = self.B
        ny, nx = arr.shape
        th_y = self.lam * np.abs(np.fft.fftfreq(ny, self.dx))
        th_x = self.lam * np.fft.rfftfreq(nx, self.dx)
        ax = th_x[None, :] + self.ap_offx
        ay = th_y[:, None] + self.ap_offy
        m = (ax ** 2 + ay ** 2) <= self.conv_ap_r ** 2
        m = m & (np.abs(ax) <= self.conv_ap_r) & (np.abs(ay) <= self.conv_ap_r)
        return B.irfft2(B.rfft2(arr) * B.asarray(m.astype(float)), (ny, nx))

    def inv_laplace(self, arr, kLow, kHigh, Nedge):
        """ImgAlignTools::invLaplace with edge blending and [1-exp(-k^2/kL^2)] exp(-k^2/kH^2) / (-4 pi^2 k^2)."""
        B = self.B
        ny, nx = arr.shape
        a = B.copy(arr)
        if Nedge > 0:
            s = 0.5 * (1 + np.cos(np.pi * np.arange(Nedge) / Nedge))
            v1 = 0.25 * (a[:, 0] + a[:, 1] + a[:, nx - 2] + a[:, nx - 1])
            S = B.asarray(s)[None, :]
            a[:, :Nedge] = S * v1[:, None] + (1 - S) * a[:, :Nedge]
            Sr = B.asarray(s[::-1].copy())[None, :]
            a[:, nx - Nedge:] = Sr * v1[:, None] + (1 - Sr) * a[:, nx - Nedge:]
            v2 = 0.25 * (a[0, :] + a[1, :] + a[ny - 2, :] + a[ny - 1, :])
            S = B.asarray(s)[:, None]
            a[:Nedge, :] = S * v2[None, :] + (1 - S) * a[:Nedge, :]
            Sr = B.asarray(s[::-1].copy())[:, None]
            a[ny - Nedge:, :] = Sr * v2[None, :] + (1 - Sr) * a[ny - Nedge:, :]
        ky = np.fft.fftfreq(ny, self.dx)
        kx = np.fft.rfftfreq(nx, self.dx)
        k2 = ky[:, None] ** 2 + kx[None, :] ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            mod = np.exp(-k2 / kHigh ** 2) * (1 - np.exp(-k2 / kLow ** 2)) / (-4 * np.pi ** 2 * k2)
        mod[0, 0] = 0.0
        return B.irfft2(B.rfft2(a) * B.asarray(mod), (ny, nx))

    # ------------------------------------------------------------------ distortions
    def apply_distortions(self, frame_index, dist, img: Plane | None = None):
        """imageData::applyDistortions: warp the full frame (bilinear; intensity x zoom^2)."""
        key = (frame_index, tuple(np.round(dist, 12)), None if img is None or img.warp is None else id(img.warp))
        if key in self._warp_cache:
            return self._warp_cache[key]
        whole = self.frames[frame_index]
        if np.all(dist == 0) and (img is None or img.warp is None):
            ny, nx = self.NY, self.NX
            out = self.B.copy(whole)
            out[:, nx - 1] = 0.0
            out[ny - 1, :] = 0.0
            if len(self._warp_cache) > 4 * self.nz:
                self._warp_cache.clear()
            self._warp_cache[key] = out
            return out
        zoom, rot = dist[DIST_ZOOM], dist[DIST_ROT]
        phiZ = np.arctan2(dist[DIST_ZOOMY], dist[DIST_ZOOMX])
        astZ = np.hypot(dist[DIST_ZOOMY], dist[DIST_ZOOMX])
        spiral, pincush = dist[DIST_SPIRAL] / 1024.0, dist[DIST_PINCUSH] / 1024.0
        r, phi = self._fr, self._fphi
        zf = 1.0 + zoom + astZ * np.cos(2.0 * (phi - phiZ)) + pincush * r
        ph = phi + rot + spiral * r
        rr = r * zf
        x2 = self._fx0 + rr * np.cos(ph)
        y2 = self._fy0 + rr * np.sin(ph)
        if img is not None and img.warp is not None:
            wx, wy = img.warp
            x2 = x2 + wx
            y2 = y2 + wy
        scale = np.ones_like(zf) if self.normalized else zf * zf
        out = self._bilinear(whole, x2, y2, scale)
        if len(self._warp_cache) > 4 * self.nz:
            self._warp_cache.clear()
        self._warp_cache[key] = out
        return out

    def _bilinear(self, img, x2, y2, scale):
        B = self.B
        ny, nx = self.NY, self.NX
        ix = np.floor(x2).astype(np.int64)
        iy = np.floor(y2).astype(np.int64)
        fx = x2 - ix
        fy = y2 - iy
        valid = (ix >= 0) & (ix < nx - 1) & (iy >= 0) & (iy < ny - 1)
        ixc = np.clip(ix, 0, nx - 2)
        iyc = np.clip(iy, 0, ny - 2)
        i00 = B.asint((iyc * nx + ixc).ravel())
        flat = img.reshape(-1)
        v00 = flat[i00]
        v10 = flat[i00 + 1]
        v01 = flat[i00 + nx]
        v11 = flat[i00 + nx + 1]
        FX, FY = B.asarray(fx.ravel()), B.asarray(fy.ravel())
        val = ((1 - FX) * (1 - FY) * v00 + FX * (1 - FY) * v10 + FX * FY * v11 + (1 - FX) * FY * v01)
        val = val * B.asarray((scale * valid).ravel())
        return val.reshape((ny, nx))

    def fit_distortions(self, img: Plane, ref_cut):
        """imageData::fitDistortions (3-point parabola per distortion, range-constrained)."""
        d = img.distortions
        delta = self.distortionMagnitude / max(self.sizeX, self.sizeY)
        dd = np.array([delta] * 6)
        if img.index != self.firstPlane and (img.fit_rot or img.fit_mag):
            dzmm = (img.defocus_img - img.defocus0) * 1e-6
            def chi_of(val, j):
                d[j] = val
                warped = self.apply_distortions(img.index, d, img)
                s = self.find_small_in_big(warped, ref_cut, self.expon, 1.0, img._damp, self.kCutoffLow,
                                           self.wave_frame_off[0] + self.NedgeSmooth,
                                           self.wave_frame_off[1] + self.NedgeSmooth,
                                           self.MAX_SHIFT, self.NedgeSmooth, self.SHIFT_LEN)
                return s[0][2]

            for _ in range(self.distortionLoops):
                for j in range(6):
                    if self.dist_range[j] > 0:
                        d0 = d[j]
                        if self.distortionRefine <= 0:
                            # C++: one 3-point parabola with the fixed step (no convexity check)
                            chi = [chi_of(d0 + t * dd[j], j) for t in (-1, 0, 1)]
                            d[j] = _find_minimum(chi, d0, dd[j])
                        else:
                            # extension: bracketed parabola, step halved in each refinement
                            step = dd[j]
                            c0 = chi_of(d0, j)
                            for _r in range(self.distortionRefine):
                                cm, cp = chi_of(d0 - step, j), chi_of(d0 + step, j)
                                den = cm - 2 * c0 + cp
                                if den > 0:
                                    x = float(np.clip(0.5 * (cm - cp) / den, -1.0, 1.0)) * step
                                else:
                                    x = (-step, 0.0, step)[int(np.argmin([cm, c0, cp]))]
                                if x != 0.0:
                                    c_new = chi_of(d0 + x, j)
                                    best = min((c_new, d0 + x), (c0, d0), (cm, d0 - step), (cp, d0 + step))
                                    c0, d0 = best
                                step *= 0.5
                            d[j] = d0
                        ref = dzmm * self.dist_poly[j, 0]
                        lim = abs(self.dist_range[j] * dzmm)
                        if abs(d[j] - ref) > lim and d[j] != 0:
                            d[j] = ref + (d[j] - ref) * lim / abs(d[j] - ref)
        img.distortions = d
        return self.apply_distortions(img.index, d, img)

    def _fit_reference_points(self, img: Plane, warped, ref_cut):
        """Extension: smooth displacement field from local chi^2 searches (NwarpX x NwarpY)."""
        from scipy.ndimage import zoom as ndzoom
        B = self.B
        nx_, ny_ = max(self.NwarpX, 2), max(self.NwarpY, 2)
        fx, fy = img.frame_off
        ref = B.to_numpy(ref_cut)
        W = B.to_numpy(warped)
        h, w = ref.shape
        bh, bw = max(h // ny_, 16), max(w // nx_, 16)
        dX = np.zeros((ny_, nx_))
        dY = np.zeros((ny_, nx_))
        from skimage.registration import phase_cross_correlation
        for j in range(ny_):
            for i in range(nx_):
                cy = int((j + 0.5) * h / ny_)
                cx = int((i + 0.5) * w / nx_)
                y0, x0 = max(cy - bh // 2, 0), max(cx - bw // 2, 0)
                a = ref[y0:y0 + bh, x0:x0 + bw]
                Y0, X0 = fy + y0, fx + x0
                if Y0 < 0 or X0 < 0 or Y0 + bh > W.shape[0] or X0 + bw > W.shape[1]:
                    continue
                b = W[Y0:Y0 + bh, X0:X0 + bw]
                try:
                    s = phase_cross_correlation(a, b, upsample_factor=10, normalization=None)[0]
                except Exception:
                    s = (0.0, 0.0)
                dY[j, i], dX[j, i] = -s[0], -s[1]
        dX -= np.median(dX)
        dY -= np.median(dY)
        lim = max(2.0, self.MAX_SHIFT / 4)
        dX, dY = np.clip(dX, -lim, lim), np.clip(dY, -lim, lim)
        full_x = ndzoom(dX, (self.NY / ny_, self.NX / nx_), order=3, mode="nearest")[:self.NY, :self.NX]
        full_y = ndzoom(dY, (self.NY / ny_, self.NX / nx_), order=3, mode="nearest")[:self.NY, :self.NX]
        prev = img.warp
        if prev is not None:
            full_x, full_y = prev[0] + full_x, prev[1] + full_y
        img.warp = (full_x, full_y)
        return self.apply_distortions(img.index, img.distortions, img)

    # ------------------------------------------------------------------ reading planes
    def _crop(self, arr, x, y):
        """Cut a sizeY x sizeX window at top-left (x, y), zero-filled outside."""
        B = self.B
        out = B.zeros((self.sizeY, self.sizeX))
        NY, NX = arr.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(NX, x + self.sizeX), min(NY, y + self.sizeY)
        if x1 > x0 and y1 > y0:
            out[y0 - y:y1 - y, x0 - x:x1 - x] = arr[y0:y1, x0:x1]
        return out

    def _new_plane(self, index):
        img = Plane(index=index, defocus_img=float(self.defocus[index]),
                    defocus0=float(self.defocus[self.firstPlane]), alpha=self.alpha0)
        img.fit_rot = self.fitRotation
        img.fit_mag = self.fitMagnification
        dz = self.defocus[index] - self.defocus[self.firstPlane]
        mm = 1e-6 * dz
        d = np.zeros(6)
        for j in range(6):
            c1, c2, c3 = self.dist_poly[j]
            d[j] = mm * (c1 + mm * (c2 + mm * c3))
        for j, k in enumerate(DIST_NAMES):
            if k in self.vectors:
                d[j] = self.vectors[k][index]
        if "alpha" in self.vectors:
            img.alpha = 1e-3 * self.vectors["alpha"][index]
        if self.fitDistortion and index != self.firstPlane and d[DIST_ZOOM] == 0:
            i2 = index + 1 if index < self.firstPlane else index - 1
            p2 = self.planes.get(i2)
            if p2 is not None:
                if abs(self.defocus[i2]) > 0:
                    d = p2.distortions * self.defocus[index] / self.defocus[i2]
                else:
                    d = p2.distortions.copy()
            i3 = i2 - (index - i2)
            p3 = self.planes.get(i3)
            if p3 is not None and p2 is not None:
                d = 2.0 * p2.distortions - p3.distortions
        img.distortions = np.array(d, float)
        return img

    def read_plane(self, index, ref=None):
        """frwrData::readPlane.  ref: None | 'wave' (simulated image) | Plane (experimental)."""
        if index < 0 or index >= self.nz or index < self.startFrame or index > self.stopFrame:
            return -1
        img = self.planes.get(index)
        if img is None:
            img = self._new_plane(index)
            self.planes[index] = img
        indexPrev = self.firstPlane
        if index != self.firstPlane:
            indexPrev = index - 1 if index > self.firstPlane else index + 1
        if ref is None:
            img.wave_shift = [0.0, 0.0]
            x = self.roi[1] + img.offset[0]
            y = self.roi[0] + img.offset[1]
            img.frame_off = [x, y]
            img.data = self._crop(self.frames[index], x, y)
            if self.padding == 0:
                self._smoothen_image(img)
            return 0
        img.wave_shift = [0.0, 0.0]
        img.frame_off = [self.roi[1], self.roi[0]]
        if indexPrev != self.firstPlane and indexPrev in self.planes:
            img.frame_off = list(self.planes[indexPrev].frame_off)
        img.fit_ref_points = self.fitReferencePointsFlag
        if ref == "wave":
            ref_cut = self.get_cut(self.intensity)
        else:
            ref_cut = ref.data
        self._extract_subregion(img, ref_cut)
        if self.padding == 0:
            self._smoothen_image(img)
        return 0

    def _extract_subregion(self, img: Plane, ref_cut):
        B = self.B
        if ref_cut.shape == (self.NY, self.NX):
            # same size: no search (C++ keeps the previous integer shift)
            warped = self.apply_distortions(img.index, img.distortions, img)
            sx, sy = img.shift_xy
            img.shifts = [(float(sx), float(sy), 0.0)] * max(1, self.SHIFT_LEN)
            img.frame_off = [sx, sy]
        else:
            img._damp = 1.097 / self.kCutoffHigh
            if self.fitDistortion and (img.fit_rot or img.fit_mag):
                warped = self.fit_distortions(img, ref_cut)
            else:
                warped = self.apply_distortions(img.index, img.distortions, img)
            if img.fit_ref_points:
                warped = self._fit_reference_points(img, warped, ref_cut)
            sh = self.find_small_in_big(warped, ref_cut, self.expon, self.chi2weight, img._damp,
                                        self.kCutoffLow, img.frame_off[0] + self.NedgeSmooth,
                                        img.frame_off[1] + self.NedgeSmooth, self.MAX_SHIFT,
                                        self.NedgeSmooth, self.SHIFT_LEN)
            img.shifts = [(s[0] - self.NedgeSmooth, s[1] - self.NedgeSmooth, s[2]) for s in sh]
            sx, sy = int(img.shifts[0][0]), int(img.shifts[0][1])
            sub = getattr(self, "_last_sub", [0.0, 0.0])
            img.shift_sub = [sx + sub[0], sy + sub[1]]
            img.frame_off = [sx, sy]
            img.shift_xy = [sx, sy]
            img.shift_min = img.shifts[0][2]
        if img.residue is not None and img.residue_off != img.frame_off:
            dX = img.frame_off[0] - img.residue_off[0]
            dY = img.frame_off[1] - img.residue_off[1]
            res = B.to_numpy(img.residue)
            new = np.zeros_like(res)
            h, w = res.shape
            ys, xs = slice(max(0, dY), min(h, h + dY)), slice(max(0, dX), min(w, w + dX))
            ys2, xs2 = slice(max(0, dY) - dY, min(h, h + dY) - dY), slice(max(0, dX) - dX, min(w, w + dX) - dX)
            new[ys, xs] = res[ys2, xs2]
            img.residue = B.asarray(new)
            img.residue_off = list(img.frame_off)
        img.data = self._crop(warped, sx, sy)
        img.wave_shift = [0.0, 0.0]

    # ------------------------------------------------------------------ amplitude constraints
    def _overlap(self, img: Plane):
        """Slices of the wave grid (inside the ROI) and of the data that overlap for the wave shift."""
        sx, sy = int(round(img.wave_shift[0])), int(round(img.wave_shift[1]))
        X, Y = self.sizeX, self.sizeY
        x0, x1 = max(0, -sx), min(X, X - sx)
        y0, y1 = max(0, -sy), min(Y, Y - sy)
        wsl = (slice(self.py + y0, self.py + y1), slice(self.px + x0, self.px + x1))
        dsl = (slice(y0 + sy, y1 + sy), slice(x0 + sx, x1 + sx))
        # R-value edge exclusion (ix >= e, ix < N-e, iy > e, iy < N-e)
        e = self.MAX_SHIFT
        xx = np.arange(x0, x1)
        yy = np.arange(y0, y1)
        rm = ((yy > e) & (yy < Y - e))[:, None] & ((xx >= e) & (xx < X - e))[None, :]
        return wsl, dsl, rm

    def _data_for(self, img: Plane, dsl):
        D = img.data[dsl]
        if self.subpixel and (img.wave_shift[0] % 1 or img.wave_shift[1] % 1):
            pass
        return D

    def adjust_amplitudes(self, img: Plane, phase_pred=False, residue=False, enforce=False):
        B = self.B
        wsl, dsl, rm = self._overlap(img)
        D = self._data_for(img, dsl)
        I = self.intensity[wsl]
        w = self.w[wsl]
        rmask = B.asarray(rm.astype(float))
        pos = D > 0
        a = B.abs(w)
        if enforce:
            fac = B.where(I <= 0, 1.0, B.sqrt(B.maximum(D, 0.0) / B.where(I <= 0, 1.0, I)))
            fac = B.where(pos, fac, 1.0)
            num = B.sum(B.where(pos, B.abs(I - D), 0.0) * rmask)
            den = B.sum(B.where(pos, D, 0.0) * rmask)
            self.w[wsl] = w * fac
            den = B.item(den)
            return B.item(num) / den if den > 0 else 1.0
        if residue:
            return self._adjust_residue(img, wsl, dsl, rm)
        sqD = B.sqrt(B.maximum(D, 0.0))
        sqI = B.sqrt(B.maximum(I, 0.0))
        small = a < 1e-8
        a_safe = B.maximum(a, 1e-8)
        fac = (a + self.feedback * (sqD - sqI)) / a_safe
        fac = B.where(small, sqD * 1e8, fac)
        fac = B.where(pos, fac, 1.0)
        if self.fitIncIntensity:
            self.incIntDelta[wsl] = self.incIntDelta[wsl] + B.where(pos & ~small, D - I, 0.0)
        if phase_pred:
            dIdz = B.zeros((self.Sy, self.Sx))
            dz = img.defocus_img - img.defocus0
            dIdz[wsl] = (I - D) / dz
        self.w[wsl] = w * fac
        num = B.sum(B.where(pos, B.abs(I - D), 0.0) * rmask)
        den = B.item(B.sum(B.where(pos, D, 0.0) * rmask))
        R = B.item(num) / den if den > 0 else 1.0
        if phase_pred:
            dphi = self.predict_phase_shift(img, dIdz)
            self.w = self.w * (B.cos(dphi) + 1j * B.sin(dphi))
        return R

    def _adjust_residue(self, img, wsl, dsl, rm):
        """waveFunction::adjustAmplitudesResidue (modifies the data and residue of the plane)."""
        B = self.B
        if img.residue is None:
            img.residue = B.zeros((self.sizeY, self.sizeX))
            img.residue_off = list(img.frame_off)
        D = B.copy(img.data[dsl])
        Rs = B.copy(img.residue[dsl])
        I = self.intensity[wsl]
        w = self.w[wsl]
        rmask = B.asarray(rm.astype(float))
        pos = D > 0
        a = B.abs(w)
        small = a < 1e-8
        a_safe = B.where(small, 1.0, a)
        diff = I - D
        oldRes = Rs
        # case diff >= 0
        c1 = (diff >= 0) & (Rs > diff)
        D1 = B.where(c1, D + diff, B.where(diff >= 0, D + Rs, D))
        R1 = B.where(c1, Rs - diff, B.where(diff >= 0, 0.0, Rs))
        fac_pos = B.where(c1, B.sqrt(B.maximum(D1, 0.0)) / a_safe,
                          (a + self.feedback * (B.sqrt(B.maximum(D1, 0.0)) - B.sqrt(B.maximum(I, 0.0)))) / a_safe)
        # case diff < 0
        Rn = Rs - 0.05 * diff
        Dn = D + 0.05 * diff
        neg = Dn < 0
        Rn = B.where(neg, Rn + Dn, Rn)
        Dn = B.where(neg, 0.0, Dn)
        fac_neg = (a + B.sqrt(Dn) - B.sqrt(B.maximum(I, 0.0))) / a_safe
        Dnew = B.where(diff >= 0, D1, Dn)
        Rnew = B.where(diff >= 0, R1, Rn)
        fac = B.where(diff >= 0, fac_pos, 1.0)        # C++ does not apply fac_neg (bug kept)
        fac = B.where(small, B.sqrt(B.maximum(D, 0.0)) * 1e8, fac)
        fac = B.where(pos, fac, 1.0)
        Dnew = B.where(pos & ~small, Dnew, D)
        Rnew = B.where(pos & ~small, Rnew, Rs)
        self.w[wsl] = w * fac
        img.data[dsl] = Dnew
        img.residue[dsl] = Rnew
        num = B.where(pos & small, D - I, B.where(pos, B.abs(I - (Dnew + oldRes)), 0.0))
        den = B.item(B.sum(B.where(pos, Dnew, 0.0) * rmask))
        _ = fac_neg
        return B.item(B.sum(num * rmask)) / den if den > 0 else 1.0

    def predict_phase_shift(self, img: Plane, dIdz):
        """waveFunction::predictPhaseShift: TIE applied to the intensity residual."""
        B = self.B
        lam = self.lam
        dz = abs(img.defocus_img - img.defocus0)
        kLow = np.sqrt(self.threshCTFlow / (np.pi * lam * dz))
        kHigh = np.sqrt(3.0 / (np.pi * lam * dz))
        img0 = self.exitAmp ** 2
        img0 = self.conv_gaussian(img0, self.kTIEHigh_wave)
        d = self.inv_laplace(dIdz, kLow, kHigh, 0)
        gy = (B.roll(d, -1, 0) - B.roll(d, 1, 0)) / (2 * self.dx)
        gx = (B.roll(d, -1, 1) - B.roll(d, 1, 1)) / (2 * self.dx)
        den = B.where(img0 > self.imgThresh, img0, self.imgThresh)
        gx, gy = gx / den, gy / den
        scale = 2 * np.pi / lam
        div = scale * ((B.roll(gy, -1, 0) - B.roll(gy, 1, 0)) / (2 * self.dx) +
                       (B.roll(gx, -1, 1) - B.roll(gx, 1, 1)) / (2 * self.dx))
        ph = self.inv_laplace(div, kLow, kHigh, self.Nedge)
        h = 0.5 * self.phaseThresh
        ph = B.where(ph > h, h * (1.0 + (1.0 - B.exp((h - ph) / h))), ph)
        ph = B.where(ph < -h, -h * (1.0 + (1.0 - B.exp((ph + h) / h))), ph)
        return ph

    # ------------------------------------------------------------------ R-factor, fits
    def get_rfactor(self, img: Plane, shift, int_cut):
        B = self.B
        sx, sy = int(round(shift[0])), int(round(shift[1]))
        X, Y = self.sizeX, self.sizeY
        x0, x1 = max(0, -sx), min(X, X - sx)
        y0, y1 = max(0, -sy), min(Y, Y - sy)
        I = int_cut[y0:y1, x0:x1]
        D = img.data[y0 + sy:y1 + sy, x0 + sx:x1 + sx]
        e = self.MAX_SHIFT
        xx, yy = np.arange(x0, x1), np.arange(y0, y1)
        rm = B.asarray((((yy > e) & (yy < Y - e))[:, None] & ((xx >= e) & (xx < X - e))[None, :]).astype(float))
        pos = D > 0
        num = B.sum(B.where(pos, B.abs(I - D), 0.0) * rm)
        den = B.item(B.sum(B.where(pos, D, 0.0) * rm))
        return B.item(num) / den if den > 0 else 1.0

    def _cut_for_compare(self):
        if self.padding == 0:
            return self.intensity
        return self.smoothen_edge2(self.get_cut(self.intensity), self.NedgeSmooth)

    def fit_alpha(self, img: Plane, defocus):
        """waveFunction::fitConvergence2Intensity."""
        a0 = img.alpha
        dA = 0.15 * a0
        damp = 0.8325546 / self.kCutoffHigh
        I0 = self.intensity
        alphas = [a0 - dA, a0, a0 + dA] if a0 - dA >= 0 else [0.0, 0.1 * dA, 0.2 * dA]
        chi = []
        for a in alphas:
            img.alpha = a
            self.fitAlphaFlag = 0
            self.convolute_espat(img, defocus)
            cut = self._cut_for_compare()
            s = self.find_small_in_big_whole(img.data, cut, self.expon, self.chi2weight, damp,
                                             self.kCutoffLow, self.Nedge, self.MAX_SHIFT)
            chi.append(self.get_rfactor(img, s, cut))
            self.intensity = I0
        img.alpha = a0
        img.alpha = _find_minimum(chi, a0, dA)
        self._say(3, f"{img.index}: alpha {a0*1e3:.5f} -> {img.alpha*1e3:.5f} mrad")

    def refine_defocus(self, img: Plane, defocus):
        """waveFunction::refineDefocus (3-point parabola of the R-factor)."""
        dD = abs(0.2 * self.defocusStep)
        damp = 0.8325546 / self.kCutoffHigh
        R = []
        for j in range(3):
            d = defocus + (j - 1) * dD
            self.propagate(d)
            self.get_intensity()
            if self.flux_pres:
                self.fitAlphaFlag = 0
                self.convolute_espat(img, d)
            if self.MAX_SHIFT > 0:
                cut = self._cut_for_compare()
                s = self.find_small_in_big_whole(img.data, cut, self.expon, self.chi2weight, damp,
                                                 self.kCutoffLow, self.Nedge, self.MAX_SHIFT)
            else:
                cut = self._cut_for_compare()
                s = (0, 0, 0)
            R.append(self.get_rfactor(img, s, cut))
        new = _find_minimum(R, defocus, dD)
        self._say(3, f"{img.index}: defocus {defocus:.2f} -> {new:.2f} nm")
        return new

    # ------------------------------------------------------------------ projection
    def project(self, img: Plane, defocus):
        """waveFunction::project."""
        B = self.B
        self.simulate_image(img, defocus)
        if self.alignFlag:
            damp = 0.8325546 / self.kCutoffHigh
            if self.MAX_SHIFT > 0:
                cut = self.intensity if self.padding == 0 else \
                    self.smoothen_edge2(self.get_cut(self.intensity), self.NedgeSmooth)
                s = self.find_small_in_big_whole(img.data, cut, self.expon, self.chi2weight, damp,
                                                 self.kCutoffLow, self.Nedge, self.MAX_SHIFT)
            else:
                s = (0, 0, 0)
            if img.wave_shift != [s[0], s[1]]:
                img.wave_shift = [s[0], s[1]]
                self._say(3, f"Image {img.index}: wave shift = ({s[0]:.1f}, {s[1]:.1f})")
        if self.subpixel and (img.wave_shift[0] % 1 or img.wave_shift[1] % 1):
            R = self._adjust_subpixel(img)
        elif img.alpha == 0:
            if self.phaseThresh == 0:
                R = self.adjust_amplitudes(img, enforce=True)
            else:
                R = self.adjust_amplitudes(img, phase_pred=abs(img.defocus_img - img.defocus0) >= 0.1)
        else:
            if self.finalLoop == 2:
                R = self.adjust_amplitudes(img, residue=True)
            elif self.doPhasePrediction and abs(img.defocus_img - img.defocus0) >= 0.1:
                R = self.adjust_amplitudes(img, phase_pred=True)
            else:
                R = self.adjust_amplitudes(img)
        self.Rvalue += R
        self.back_propagate(defocus)
        self.avgWave = self.avgWave + self.w
        self.imageCount += 1

    def _adjust_subpixel(self, img):
        """Extension: fractional wave shifts via Fourier interpolation of the data."""
        B = self.B
        sx, sy = img.wave_shift
        ix, iy = int(np.floor(sx)), int(np.floor(sy))
        fx, fy = sx - ix, sy - iy
        D = img.data
        ky = np.fft.fftfreq(self.sizeY)[:, None]
        kx = np.fft.rfftfreq(self.sizeX)[None, :]
        ph = B.asarray(np.exp(2j * np.pi * (kx * fx + ky * fy)), True)
        Ds = B.irfft2(B.rfft2(D) * ph, (self.sizeY, self.sizeX))
        old_d, old_s = img.data, img.wave_shift
        img.data = Ds * (D > 0)
        img.wave_shift = [float(ix), float(iy)]
        R = self.adjust_amplitudes(img)
        img.data, img.wave_shift = old_d, old_s
        return R

    def assign_avg_to_exit(self):
        """waveFunction::assignAvg2ExitFace."""
        B = self.B
        if self.imageCount <= 0:
            return
        self.exitWave = self.avgWave * self.soft_ap / self.imageCount
        avg = B.ifft2(self.exitWave) * (self.Sx * self.Sy)
        self.avgWave = avg
        if self.allowPhaseWrapping == 0:
            ce, se = B.cos(self.exitPhase), B.sin(self.exitPhase)
            re = avg.real * ce + avg.imag * se
            im = avg.imag * ce - avg.real * se
            self.exitPhase = self.exitPhase + self.feedback * B.arctan2(im, re)
            self.exitAmp = B.abs(avg)
            if self.enforcePositivePhase:
                self.exitPhase = B.where(self.exitPhase < 0, self.exitPhase * 0.98 * self.feedback, self.exitPhase)
        else:
            self.exitPhase = B.arctan2(avg.imag, avg.real)
            self.exitAmp = B.abs(avg)
        if self.fitIncIntensity:
            self.incInt = B.maximum((self.incInt + self.feedbackIncBkgrnd * self.incIntDelta / self.imageCount)
                                    * self.dampIncInt, 0.0)
            self.incIntDelta = B.zeros((self.Sy, self.Sx))
        self.imageCount = 0
        if self.allowPhaseWrapping == 0 and self.enforcePositivePhase:
            self.copy_amp_phase_to_exit()
            self.fft_exit()

    # ------------------------------------------------------------------ gradient flipping
    def do_gradient_flip(self, order):
        """waveFunction::doGradientFlip (GF_METHOD 1, flipping inside the ROI + NedgeFlip)."""
        B = self.B
        Sx, Sy = self.Sx, self.Sy
        N = Sx * Sy
        NedgeFlip = 30
        NedgeFlip = int(min(NedgeFlip, np.floor(0.5 * self.px), np.floor(0.5 * self.py)))
        lowFreq = self.lam / self.GF_length            # theta units
        wy = np.arange(Sy)
        wy = np.where(wy < Sy // 2, wy, wy - Sy).astype(float)
        wx = np.arange(Sx)
        wx = np.where(wx < Sx // 2, wx, wx - Sx).astype(float)
        WY, WX = np.meshgrid(wy, wx, indexing="ij")
        re = -Sy * np.sin(WY * 2 * np.pi / Sy)
        im = Sx * np.sin(WX * 2 * np.pi / Sx)
        if order == 2:
            re, im = re * re, im * im
        elif order == 3:
            re, im = re * re - im * im, -2.0 * re * im
        Dfw = re + 1j * im
        re2 = -Sy * np.sin(WY * 2 * np.pi / Sy)
        im2 = -Sx * np.sin(WX * 2 * np.pi / Sx)
        if order == 2:
            re2, im2 = re2 * re2, -im2 * im2
        elif order == 3:
            re2, im2 = re2 * re2 - im2 * im2, -2.0 * re2 * im2
        Dbw = re2 + 1j * im2
        mod2 = np.maximum(re2 ** 2 + im2 ** 2, 1e-10)
        k2 = (WY * self.lam / (self.dx * Sy)) ** 2 + (WX * self.lam / (self.dx * Sx)) ** 2
        h = np.exp(-k2 / lowFreq ** 2)
        PH = B.fft2(self.exitPhase + 0j) / N
        G = B.ifft2(PH * B.asarray(Dfw, True)) * N
        gr, gi = G.real, G.imag
        X0, Y0 = self.px, self.py
        reg_max = (slice(Y0 + 3, Y0 + self.sizeY - 3), slice(X0 + 3, X0 + self.sizeX - 3))
        mx = max(B.item(B.max(B.abs(gr[reg_max]))), B.item(B.max(B.abs(gi[reg_max]))))
        reg = (slice(Y0 - NedgeFlip, Y0 + self.sizeY + NedgeFlip), slice(X0 - NedgeFlip, X0 + self.sizeX + NedgeFlip))
        a_r, a_i = gr[reg], gi[reg]
        thr = self.GF_thresh
        fr_ = B.abs(a_r) < thr * mx
        fi_ = B.abs(a_i) < thr * mx
        count = float(int(B.item(B.sum(fr_ * 1.0)) + B.item(B.sum(fi_ * 1.0))) // 2)
        gr = B.copy(gr)
        gi = B.copy(gi)
        gr[reg] = B.where(fr_, -self.GF_beta * a_r, a_r)
        gi[reg] = B.where(fi_, -self.GF_beta * a_i, a_i)
        npix = (self.sizeX + 2 * NedgeFlip) * (self.sizeY + 2 * NedgeFlip)
        target = self.GF_frac * npix
        new_t = thr
        if count > 1.02 * target or count < 0.98 * target:
            new_t = thr * target / max(count, 1.0)
        if new_t != thr:
            # C++ counts on the already flipped gradients
            c2 = float(int(B.item(B.sum((B.abs(gr[reg]) < new_t * mx) * 1.0)) +
                           B.item(B.sum((B.abs(gi[reg]) < new_t * mx) * 1.0))) // 2)
            if c2 != count:
                thr = thr + 0.2 * (target - count) * (new_t - thr) / (c2 - count)
            else:
                thr = new_t
            thr = min(max(thr, 1e-3), 0.9)
        self._say(2, f"GF: flipped {100*count/npix:.1f} % (target {100*self.GF_frac:.1f} %), order {order}, "
                     f"threshold {self.GF_thresh:.4f} -> {thr:.4f}")
        self.GF_thresh = thr
        G2 = B.fft2(gr + 1j * gi) / N
        F = G2 * B.asarray(Dbw / mod2, True)
        Hd = B.asarray(h)
        Wc = Hd * F + (1 - Hd) * PH
        self.exitPhase = B.ifft2(Wc).real * N

    # ------------------------------------------------------------------ initialisation
    def init_exit_wave(self):
        B = self.B
        S = (self.Sy, self.Sx)
        self.exitAmp = B.ones(S)
        self.exitPhase = B.zeros(S)
        self.copy_amp_phase_to_exit()

    def _pad_amplitude_from(self, img: Plane):
        """Padded start amplitude: data inside, cosine-blended edge values outside, Gaussian smoothed."""
        B = self.B
        d = B.to_numpy(img.data)
        Sx, Sy, px, py = self.Sx, self.Sy, self.px, self.py
        X, Y = self.sizeX, self.sizeY
        amp = np.full((Sy, Sx), np.sqrt(max(d.mean(), 0)))
        amp[py:py + Y, px:px + X] = np.sqrt(np.maximum(d, 0))
        if px > 0:
            vr = np.sqrt(np.maximum(0.5 * (d[:, X - 1] + d[:, X - 2]), 0))
            vl = np.sqrt(np.maximum(0.5 * (d[:, 0] + d[:, 1]), 0))
            ixr = np.arange(X + px, Sx)
            wr = 0.5 * (1 + np.cos(np.pi * (ixr - X - px) / px))
            amp[py:py + Y, X + px:] = (1 - wr)[None, :] * amp[py:py + Y, X + px:] + wr[None, :] * vr[:, None]
            ixl = np.arange(px)
            wl = 0.5 * (1 + np.cos(np.pi * (px - ixl) / px))
            amp[py:py + Y, :px] = (1 - wl)[None, :] * amp[py:py + Y, :px] + wl[None, :] * vl[:, None]
        if py > 0:
            vt = 0.5 * (amp[py, :] + amp[py + 1, :])
            iyt = np.arange(py)
            wt = 0.5 * (1 + np.cos(np.pi * (py - iyt) / py))
            amp[:py, :] = (1 - wt)[:, None] * amp[:py, :] + wt[:, None] * vt[None, :]
            vb = 0.5 * (amp[Sy - 1 - py, :] + amp[Sy - 2 - py, :])
            iyb = np.arange(Y + py, Sy)
            wb = 0.5 * (1 + np.cos(np.pi * (iyb - Y - py) / py))
            amp[Y + py:, :] = (1 - wb)[:, None] * amp[Y + py:, :] + wb[:, None] * vb[None, :]
        A = B.asarray(amp)
        A = self.conv_gaussian(A, 0.2 * X / (self.dx * Sx))
        self.exitAmp = A
        self.exitPhase = B.zeros((Sy, Sx))
        self.copy_amp_phase_to_exit()

    def _loop(self, planes, n, label, align_every=None):
        """Simple multi-plane projection loop used by the initialisation."""
        for j in range(n):
            self.reset_avg()
            for i in planes:
                self.project(self.planes[i], self.defocus[i])
            self.assign_avg_to_exit()
            R = self.Rvalue / len(planes)
            self._say(1, f"{label} {j} {tuple(planes)}: R-value={R:.6g}")
            self.rvalues.append((label, j, R))
        return R

    def init_from_two(self, i0, i1):
        """initWaveFrom2Images (defocus fit only if 'fit defocus' > 4)."""
        self.doPhasePrediction = 0
        delta = 1.5
        R = 1.0
        for j in range(self.initLoops2):
            self.alignFlag = 0
            self.reset_avg()
            self.project(self.planes[i0], self.defocus[i0])
            self.project(self.planes[i1], self.defocus[i1])
            self.assign_avg_to_exit()
            R = self.Rvalue / 2
            self._record("2-loop", j, R, f"planes ({i0}, {i1})", n_img=2)
            if self.allowDefFitting > 4 and j % 2 == 0:
                p = self.planes[i1]
                new = self.refine_defocus(p, self.defocus[i1])
                dd = new - self.defocus[i1]
                self.defocus[i1] = (1 - delta) * self.defocus[i1] + delta * new
                if dd < 1.0:
                    break
        return R

    def include_third(self, i0, i1, i2):
        self.simulate_image(self.planes[i0], self.defocus_of(i2))
        self.read_plane(i2, "wave")
        self.fitAlphaFlag = 0
        self.alignFlag = 0
        self.doPhasePrediction = 0
        R = 1.0
        for j in range(self.initLoops3):
            self.reset_avg()
            for i in (i0, i1, i2):
                self.project(self.planes[i], self.defocus[i])
            self.assign_avg_to_exit()
            R = self.Rvalue / 3
            self._record("3-loop", j, R, f"planes ({i0}, {i1}, {i2})", n_img=3)
        return R

    def defocus_of(self, i):
        return float(self.defocus[i])

    def init_wave_trial(self):
        """frwrData::initWaveTrial."""
        nz = self.nz
        if self.stopFrame < 0:
            self.stopFrame = nz - 1
        if self.startFrame < 0:
            self.startFrame = 0
        df = self.defocus
        if self.firstPlane == -1:
            if nz < 3 or self.stopFrame - self.startFrame < 2:
                self.firstPlane = self.startFrame if abs(df[self.startFrame]) < abs(df[self.stopFrame]) else self.stopFrame
            else:
                cand = range(1, nz - 1)
                self.firstPlane = min(cand, key=lambda j: (abs(df[j]), j))
                if self.startFrame > self.firstPlane - 1:
                    self.firstPlane = self.startFrame + 1
                if self.stopFrame >= 0 and self.firstPlane > self.stopFrame - 1:
                    self.firstPlane = self.stopFrame - 1
        fp = self.firstPlane
        self._say(1, f"Wave trial initialization: reference image {fp}")
        self.read_plane(fp, None)
        if fp > self.startFrame:
            self.read_plane(fp - 1, self.planes[fp])
        if fp < self.stopFrame:
            self.read_plane(fp + 1, self.planes[fp])
        self.defocusStep = abs(df[fp + 1] - df[fp]) if fp < self.stopFrame else abs(df[fp] - df[fp - 1])
        # candidate shifts (best chi^2 first)
        trials = []
        jm = jp = 0
        lo = self.planes.get(fp - 1) if fp > self.startFrame else None
        hi = self.planes.get(fp + 1) if fp < self.stopFrame else None
        for _ in range(max(self.SHIFT_LEN, 0)):
            if lo is not None and hi is not None:
                if lo.shifts[min(jm, len(lo.shifts) - 1)][2] < hi.shifts[min(jp, len(hi.shifts) - 1)][2]:
                    trials.append((lo.shifts[min(jm, len(lo.shifts) - 1)], fp - 1)); jm += 1
                else:
                    trials.append((hi.shifts[min(jp, len(hi.shifts) - 1)], fp + 1)); jp += 1
            elif lo is not None:
                trials.append((lo.shifts[min(jm, len(lo.shifts) - 1)], fp - 1)); jm += 1
            else:
                trials.append((hi.shifts[min(jp, len(hi.shifts) - 1)], fp + 1)); jp += 1
        Rmin = 1e10
        best = None
        best_wave = None
        fdf = self.fitDistortion
        # C++ assigns feedback / background fitting to the wave only after the trials
        keep = (self.feedback, self.fitIncIntensity, self.feedbackIncBkgrnd, self.dampIncInt)
        self.feedback, self.fitIncIntensity, self.feedbackIncBkgrnd, self.dampIncInt = 1.0, 0, 0.1, 1.0
        for j, (sh, plane) in enumerate(trials):
            p = self.planes[plane]
            p.offset = [int(sh[0] - self.roi[1]), int(sh[1] - self.roi[0])]
            if j == 0:
                self.fitDistortion = 0
            self.read_plane(plane, None)
            self.fitDistortion = fdf
            self.init_exit_wave()
            if self.padding > 0:
                self._pad_amplitude_from(self.planes[fp])
            if self.init_wave_user is not None:
                self._set_user_wave(self.init_wave_user)
            if self.useTIE:
                i2 = (fp - 1 if plane > fp else fp + 1) if (self.startFrame < fp < self.stopFrame) else -1
                R = self.tie_guess(fp, plane, i2)
                if R < Rmin:
                    Rmin = R
                    i2b = i2 if i2 >= 0 else plane
                    best = (plane, list(p.frame_off), i2b, list(self.planes[i2b].frame_off))
                    best_wave = (self.B.copy(self.exitWave), self.exitRecip)
                continue
            R2 = self.init_from_two(fp, plane)
            if self.stopFrame - self.startFrame < 2:
                if R2 < Rmin:
                    Rmin = R2
                    best = (plane, list(p.frame_off), plane, list(p.frame_off))
                    best_wave = (self.B.copy(self.exitWave), self.exitRecip)
                continue
            saved = (self.B.copy(self.exitWave), self.exitRecip)
            cands = [i2 for i2 in ((fp - 1 if plane > fp else fp + 1), (plane + 1 if plane > fp else plane - 1))]
            first = True
            for i2 in cands:
                if not first:      # C++: restore the 2-plane wave before the second candidate
                    self.exitWave, self.exitRecip = self.B.copy(saved[0]), saved[1]
                self.exitPhase = self.B.zeros((self.Sy, self.Sx))
                first = False
                if self.startFrame <= i2 <= self.stopFrame:
                    R3 = self.include_third(fp, plane, i2)
                    if R3 < Rmin:
                        Rmin = R3
                        best = (plane, list(p.frame_off), i2, list(self.planes[i2].frame_off))
                        best_wave = (self.B.copy(self.exitWave), self.exitRecip)
        if best is None:
            best = (trials[0][1], list(self.planes[trials[0][1]].frame_off),
                    trials[0][1], list(self.planes[trials[0][1]].frame_off))
        for k in (0, 2):
            pl = best[k]
            P = self.planes[pl]
            P.offset = [int(best[k + 1][0] - self.roi[1]), int(best[k + 1][1] - self.roi[0])]
            self.read_plane(pl, None)
        self.feedback, self.fitIncIntensity, self.feedbackIncBkgrnd, self.dampIncInt = keep
        if best_wave is not None:
            # C++ copies bestWave into exitWave (used for reading the next planes) while
            # exitAmp/exitPhase keep the state of the last trial
            self.exitWave, self.exitRecip = best_wave
        a, b = best[0], best[2]
        self.indexLow = min(a, b, fp)
        self.indexHigh = max(a, b, fp)
        self._say(1, f"Initial planes {fp}, {best[0]}, {best[2]}; R={Rmin:.6g}; continue with planes "
                     f"{self.indexLow} .. {self.indexHigh}")
        self._init_done = True
        return Rmin

    def _set_user_wave(self, w0):
        """Extension: start from a given complex wave (ROI or padded size)."""
        B = self.B
        w0 = np.asarray(w0, complex)
        if w0.shape == (self.Sy, self.Sx):
            full = w0
        elif w0.shape == (self.sizeY, self.sizeX):
            # padding region: mirrored wave (continuous amplitude and phase at the ROI edge)
            pads = ((self.py, self.Sy - self.sizeY - self.py), (self.px, self.Sx - self.sizeX - self.px))
            if all(a < n and b_ < n for (a, b_), n in zip(pads, w0.shape)):
                full = np.pad(w0, pads, mode="symmetric")
            else:
                full = np.full((self.Sy, self.Sx), np.mean(np.abs(w0)), complex)
                full[self.py:self.py + self.sizeY, self.px:self.px + self.sizeX] = w0
        else:
            raise ValueError(f"start wave {w0.shape} does not match the ROI {(self.sizeY, self.sizeX)} or the "
                             f"padded wave {(self.Sy, self.Sx)}")
        self.exitAmp = B.asarray(np.abs(full))
        ph = np.unwrap(np.unwrap(np.angle(full), axis=0), axis=1)
        self.exitPhase = B.asarray(ph)
        self.copy_amp_phase_to_exit()

    # ------------------------------------------------------------------ main loops
    def _smooth_exit(self):
        if self.phaseSmoothing > 0:
            self.exitPhase = self.conv_gaussian(self.exitPhase, 1.0 / self.phaseSmoothing)
        else:
            self.exitPhase = self.conv_aperture(self.exitPhase)
        if self.amplitudeSmoothing > 0:
            self.exitAmp = self.conv_gaussian(self.exitAmp, 1.0 / self.amplitudeSmoothing)
        else:
            self.exitAmp = self.conv_aperture(self.exitAmp)
        self.copy_amp_phase_to_exit()
        self.avgWave = self.B.copy(self.exitWave)
        self.fft_exit()

    def _reread(self, idx, label="re-read"):
        img = self.planes[idx]
        self.simulate_image(img, self.defocus[idx])
        self.read_plane(idx, "wave")

    def _fit_all_defoci(self, lo, hi):
        for jp in range(lo, hi + 1):
            if jp != self.firstPlane:
                self.defocus[jp] = self.refine_defocus(self.planes[jp], self.defocus[jp])

    def _global_alpha(self, lo, hi, power):
        num = den = 0.0
        d0 = self.defocus[self.firstPlane]
        for jp in range(lo, hi + 1):
            rd = abs(self.defocus[jp] - d0) ** power
            num += self.planes[jp].alpha * rd
            den += rd
        if den > 0:
            a = num / den
            for jp in range(lo, hi + 1):
                self.planes[jp].alpha = a

    def _record(self, label, j, R, extra="", n_img=None):
        self.work_done += n_img if n_img is not None else max(1, getattr(self, "imageCountFinal", 1) or 1)
        self.rvalues.append((label, j, R))
        self._say(1, f"{label} {j}: align={self.alignFlag}, R-value={R:.6g} {extra}")
        if self.callback is not None:
            try:
                self.callback(label, j, R, self)
            except TypeError:
                self.callback(j, None, R)

    def reconstruct_all(self):
        """frwrData::reconstructAll."""
        B = self.B
        self.finalLoop = 0
        firstTime = True
        nz = self.nz
        fp = self.firstPlane
        imageCount = 0
        while True:
            lo_old, hi_old = self.indexLow, self.indexHigh
            lo, hi = self.indexLow, self.indexHigh
            readNew = 0
            if lo not in self.planes or (firstTime and not self.startAllPlanes):
                if hi - lo > 1:
                    self.simulate_image(self.planes[lo + 1], self.defocus_of(lo))
                    self.read_plane(lo, "wave")
                else:
                    self.read_plane(lo, self.planes[lo + 1])
                self.planes[lo].alpha = self.planes[lo + 1].alpha if (lo + 1) in self.planes else self.alpha0
                readNew = 1
            if hi not in self.planes or (firstTime and not self.startAllPlanes):
                if hi - lo > 1:
                    self.simulate_image(self.planes[hi - 1], self.defocus_of(hi))
                    self.read_plane(hi, "wave")
                else:
                    self.read_plane(hi, self.planes[hi - 1])
                if (hi - 1) in self.planes:
                    self.planes[hi].alpha = self.planes[hi - 1].alpha
                readNew = 1
            if lo > 0 or hi < nz - 1:
                readNew = 1
            if readNew == 0 and not firstTime:
                break
            firstTime = False
            imageCount = 1 + hi - lo
            if self.fitWarp and imageCount >= self.fitWarpMinImages:
                self.fitReferencePointsFlag = 1
            self.copy_amp_phase_to_exit()
            alignStep, jAlign, Rprev = 0, 0, 2.0
            for j in range(self.reconLoops + self.phasePredictLoops):
                self.doPhasePrediction = int(j >= self.reconLoops)
                self.alignFlag = 0
                self.fitAlphaFlag = 0
                if jAlign == alignStep:
                    if self.allowDefFitting > 1:
                        self._fit_all_defoci(lo, hi)
                    self.alignFlag = 1
                    alignStep = min(alignStep + 3, self.distortionLoopFrequency)
                    jAlign = 0
                jAlign += 1
                self.reset_avg()
                for jp in range(lo, hi + 1):
                    if self.alignFlag == 1 and self.allowAlphaFitting > 1 and \
                            imageCount >= self.fitAlphaMinImages and self.fitAlpha:
                        self.fitAlphaFlag = 1
                    self.project(self.planes[jp], self.defocus[jp])
                if imageCount >= self.fitAlphaMinImages and self.fitAlpha and self.globalAlpha and \
                        self.alignFlag == 1 and self.allowAlphaFitting > 0:
                    self._global_alpha(lo, hi, 2)
                self.assign_avg_to_exit()
                R = self.Rvalue / imageCount
                self._record(f"{imageCount}-Loop", j, R, "(P)" if self.doPhasePrediction else "", n_img=imageCount)
                if self.alignFlag:
                    self._smooth_exit()
                    for jp in range(lo, hi + 1):
                        if self.keepCentralPlane <= imageCount or jp != fp:
                            P = self.planes[jp]
                            if P.fit_mag or P.wave_shift[0] != 0 or P.wave_shift[1] != 0:
                                P.fit_ref_points = self.fitReferencePointsFlag
                                self._reread(jp)
                if jAlign > 1 and 0 < Rprev - R < 1e-6:
                    jAlign = alignStep
                Rprev = R
            for jp in range(lo, hi + 1):
                if self.keepCentralPlane <= imageCount or jp != fp:
                    self._reread(jp)
            # mean defocus step
            cp = self.centralPlane
            dsum = 0.0
            for jp in range(lo, hi + 1):
                if jp != cp:
                    jd = np.sign(jp - cp) * abs(jp - cp) ** self.defocusScale
                    dsum += (self.defocus[jp] - self.defocus[cp]) / jd
            if imageCount > 1:
                self.defocusStep = dsum / (imageCount - 1)
            for jp in range(lo, hi + 1):
                self._say(2, f"Image {jp}: defocus {self.defocus[jp]:.1f} nm, "
                             f"wave shift {self.planes[jp].wave_shift}")
            if self.enforceEqualDefSteps:
                d_cp = self.defocus[cp]
                for jp in range(nz):
                    jd = np.sign(jp - cp) * abs(jp - cp) ** self.defocusScale
                    self.defocus[jp] = d_cp + jd * self.defocusStep
            lo = max(lo - 1, 0)
            hi = min(hi + 1, nz - 1)
            d0 = self.defocus_of(fp)
            if self.maxDefocus > 0:
                if abs(self.defocus_of(lo) - d0) > self.maxDefocus and abs(self.defocus_of(hi) - d0) > self.maxDefocus:
                    readNew = 0
                if abs(self.defocus_of(lo) - d0) > self.maxDefocus:
                    lo += 1
                if abs(self.defocus_of(hi) - d0) > self.maxDefocus:
                    hi -= 1
            if self.startFrame >= 0 and lo < self.startFrame:
                lo += 1
            if self.stopFrame >= 0 and hi > self.stopFrame:
                hi -= 1
            self.indexLow, self.indexHigh = lo, hi
            if lo == lo_old and hi == hi_old:
                readNew = 0
            if readNew == 0:
                break
        self.imageCountFinal = 1 + self.indexHigh - self.indexLow
        self.copy_amp_phase_to_exit()
        self._final_loops(imageCount)
        self._residue_loops(imageCount)

    def _final_loops(self, imageCount):
        lo, hi = self.indexLow, self.indexHigh
        alignStep = self.distortionLoopFrequency
        self.finalLoop = 1
        self.doPhasePrediction = int(self.phaseThresh > 0)
        RvaluePrev = 0.0
        GF_thrPrev = 1.0
        GF_frac_max = self.GF_frac
        for j in range(self.reconLoopsFinal):
            self.alignFlag = 0
            self.fitAlphaFlag = 0
            if j % alignStep == 0:
                if self.allowDefFitting > 0:
                    self._fit_all_defoci(lo, hi)
                self.alignFlag = 1
            if self.applyGF and j % self.GF_freq == 0:
                lim = np.floor(10.0 * self.GF_freq * self.GF_o3frac + 0.5) - 1
                order = self.GF_order if (j % (10 * self.GF_freq)) > lim else 3
                self.do_gradient_flip(order)
                self.copy_amp_phase_to_exit()
            self.reset_avg()
            for jp in range(lo, hi + 1):
                if self.alignFlag == 1 and self.allowAlphaFitting > 0 and \
                        imageCount >= self.fitAlphaMinImages and self.fitAlpha:
                    self.fitAlphaFlag = 1
                self.project(self.planes[jp], self.defocus[jp])
            if imageCount >= self.fitAlphaMinImages and self.fitAlpha and self.globalAlpha and \
                    self.alignFlag == 1 and self.allowAlphaFitting > 0:
                self._global_alpha(lo, hi, 1)
            self.assign_avg_to_exit()
            R = self.Rvalue / imageCount
            self._record("Final Loop", j, R, "(P)" if self.doPhasePrediction else "", n_img=imageCount)
            if j < 10:
                RvaluePrev = R
            if self.applyGF and (j + 1) % self.GF_freq == 0 and self.GF_adjust and self.alignFlag == 0:
                if j > self.GF_freq:
                    if R > RvaluePrev and self.GF_frac > self.GF_minfrac and self.GF_thresh <= GF_thrPrev:
                        self.GF_frac *= 0.95
                    self.GF_frac = max(self.GF_frac, self.GF_minfrac)
                    if R < 0.98 * RvaluePrev:
                        self.GF_frac = min(self.GF_frac * 1.04, GF_frac_max)
                RvaluePrev = R if (R < RvaluePrev and self.GF_thresh <= GF_thrPrev) else RvaluePrev
                GF_thrPrev = self.GF_thresh
            if self.save_level >= 2 and ((j + 1) % 50 == 0 or j == self.reconLoopsFinal - 1):
                self.snapshots.append((j + 1, R, self.B.to_numpy(self.exitAmp * B_exp(self.B, self.exitPhase))))
            if (j + 1) % self.distortionLoopFrequency == 0:
                if self.phaseSmoothing > 0:
                    self.exitPhase = self.conv_gaussian(self.exitPhase, 1.0 / self.phaseSmoothing)
                if self.amplitudeSmoothing > 0:
                    self.exitAmp = self.conv_gaussian(self.exitAmp, 1.0 / self.amplitudeSmoothing)
                if self.phaseSmoothing > 0 or self.amplitudeSmoothing > 0:
                    self.copy_amp_phase_to_exit()
                    self.fft_exit()
                for jp in range(lo, hi + 1):
                    self._reread(jp)

    def _residue_loops(self, imageCount):
        B = self.B
        lo, hi = self.indexLow, self.indexHigh
        alignStep = self.distortionLoopFrequency
        for j in range(self.reconLoopsResidue):
            self.finalLoop = 2
            self.alignFlag = 0
            self.fitAlphaFlag = 0
            if j % alignStep == 0:
                if self.allowDefFitting > 0:
                    self._fit_all_defoci(lo, hi)
                self.alignFlag = 1
                if self.allowAlphaFitting > 0:
                    self.fitAlphaFlag = 1
            self.reset_avg()
            for jp in range(lo, hi + 1):
                self.project(self.planes[jp], self.defocus[jp])
            res = [self.planes[jp].residue for jp in range(lo, hi + 1) if self.planes[jp].residue is not None]
            if res:
                avg = sum(res) / imageCount
                for jp in range(lo, hi + 1):
                    r = self.planes[jp].residue
                    if r is not None:
                        avg = B.where(avg > r, 0.8 * r + 0.2 * avg, avg)
                avg = B.maximum(avg, 0.0)
                for jp in range(lo, hi + 1):
                    self.planes[jp].residue = B.copy(avg)
            self.assign_avg_to_exit()
            R = self.Rvalue / imageCount
            self._record("Residue Loop", j, R, n_img=imageCount)

    # ------------------------------------------------------------------ run
    def run(self):
        t0 = time.time()
        self.planes = {}
        self._say(1, f"FRWR3 (inlineholo) on {self.B.name}/{self.B.device}: {self.nz} frames "
                     f"{self.NY}x{self.NX}, ROI {self.roi}, wave {self.Sy}x{self.Sx}")
        if self.firstPlane == -1 and self.nz >= 5:
            pass
        self.kTIEHigh_wave = self.kTIEHigh if self.useTIE else 250.0     # wave.kTIEHigh (25/A)
        if self.useTIE:
            self._say(1, "use TIE: starting guess from the TIE (see inlineholo.reconstruct.tie)")
        from .control import StopRequested
        self.stopped = False
        self.work_done = 0
        try:
            if self.startAllPlanes:
                self.init_all_planes()
            else:
                self.init_wave_trial()
            self.reconstruct_all()
        except StopRequested:
            self.stopped = True
            self._say(1, "stopped by the user: returning the current wave")
        res = self.result(time.time() - t0)
        res.stopped = self.stopped
        return res

    def _frame_range(self):
        lo = max(self.startFrame, 0)
        hi = self.stopFrame if 0 <= self.stopFrame < self.nz else self.nz - 1
        ref = self.firstPlane if self.firstPlane >= 0 else lo + int(np.argmin(np.abs(self.defocus[lo:hi + 1])))
        return lo, hi, ref

    def planned_work(self):
        """Estimated (number of loops, sum over loops of the number of images in the loop).

        The time of a loop is roughly proportional to the number of images it projects, so
        progress = work_done / planned work (work_done is updated by every recorded loop)."""
        g = lambda k: float(_vals(self.cfg.get(k, DEFAULTS.get(k)))[0])
        lo, hi, ref = self._frame_range()
        n_all = hi - lo + 1
        inter = g("intermediate loops") + (g("phase predict loops") if self.phaseThresh > 0 else 0)
        loops, work = 0.0, 0.0
        if self.startAllPlanes or n_all < 3:
            counts = [n_all]
        else:
            trials = max(int(self.SHIFT_LEN), 1)
            n3 = 2 if n_all >= 3 else 0
            loops += trials * (g("init loops 2") + n3 * g("init loops 3"))
            work += trials * (2 * g("init loops 2") + 3 * n3 * g("init loops 3"))
            a, b = max(ref - 1, lo), min(ref + 1, hi)
            counts = []
            while True:
                counts.append(b - a + 1)
                a2, b2 = max(a - 1, lo), min(b + 1, hi)
                if (a2, b2) == (a, b):
                    break
                a, b = a2, b2
        loops += inter * len(counts)
        work += inter * sum(counts)
        tail = g("final loops") + g("residue loops")
        loops += tail
        work += tail * counts[-1]
        return int(loops), float(work)

    def planned_schedule(self, from_current=False):
        """Planned loops in execution order as (key, number of images) for time estimates.

        key identifies loops of the same kind: ('2-loop', j), ('3-loop', j), ('inter', j) for the
        j-th loop of every intermediate stage (the stages repeat the same pattern of alignment,
        re-reading and plain projection loops), ('final', 'align' | 'proj') and ('residue', 'proj').
        With ``from_current`` (after the initialization) only the loops still to come are listed,
        using the planes actually selected by the initialization (indexLow .. indexHigh).
        """
        g = lambda k: int(float(_vals(self.cfg.get(k, DEFAULTS.get(k)))[0]))
        lo, hi, ref = self._frame_range()
        n_all = hi - lo + 1
        inter = g("intermediate loops") + (g("phase predict loops") if self.phaseThresh > 0 else 0)
        sched = []
        if from_current and getattr(self, "_init_done", False):
            a, b = self.indexLow, self.indexHigh
            counts = []
            while True:
                counts.append(b - a + 1)
                a2, b2 = max(a - 1, lo), min(b + 1, hi)
                if (a2, b2) == (a, b):
                    break
                a, b = a2, b2
        elif self.startAllPlanes or n_all < 3:
            counts = [n_all]
        else:
            if not self.useTIE:
                trials = max(int(self.SHIFT_LEN), 1)
                for _ in range(trials):
                    sched += [(("2-loop", j), 2) for j in range(g("init loops 2"))]
                    for _ in range(2):
                        sched += [(("3-loop", j), 3) for j in range(g("init loops 3"))]
            a, b = max(ref - 1, lo), min(ref + 1, hi)
            counts = []
            while True:
                counts.append(b - a + 1)
                a2, b2 = max(a - 1, lo), min(b + 1, hi)
                if (a2, b2) == (a, b):
                    break
                a, b = a2, b2
        for n in counts:
            sched += [(("inter", j), n) for j in range(inter)]
        freq = max(1, self.distortionLoopFrequency)
        sched += [(("final", "align" if j % freq == 0 else "proj"), counts[-1]) for j in range(g("final loops"))]
        sched += [(("residue", "proj"), counts[-1]) for _ in range(g("residue loops"))]
        return sched

    def init_all_planes(self):
        """Extension 'start with all planes': read all frames at once (pre-aligned series).

        Without a start wave the frames are taken at their positions in the stack (the alignment
        is refined during the loops); with a start wave (init_wave) every frame is aligned to the
        image simulated from that wave, going outwards from the reference frame."""
        lo, hi, fp = self._frame_range()
        self.startFrame, self.stopFrame, self.firstPlane = lo, hi, fp
        self._say(1, f"Start with all planes {lo} .. {hi}, reference image {fp}")
        self.read_plane(fp, None)
        self.init_exit_wave()
        if self.padding > 0:
            self._pad_amplitude_from(self.planes[fp])
        user = self.init_wave_user is not None
        if user:
            self._set_user_wave(self.init_wave_user)
        order = sorted((i for i in range(lo, hi + 1) if i != fp), key=lambda i: (abs(i - fp), i))
        for i in order:
            if user:
                self.simulate_image(self.planes[fp], self.defocus_of(i))
                self.read_plane(i, "wave")
            else:
                self.read_plane(i, None)
        if hi > lo:
            self.defocusStep = abs(self.defocus[min(fp + 1, hi)] - self.defocus[max(fp - 1, lo)]) / \
                max(min(fp + 1, hi) - max(fp - 1, lo), 1)
        self.indexLow, self.indexHigh = lo, hi
        self._init_done = True
        return 1.0

    def planned_loops(self):
        """Estimated total number of loops (for progress information)."""
        g = lambda k: float(_vals(self.cfg.get(k, DEFAULTS.get(k)))[0])
        ref = self.firstPlane if self.firstPlane >= 0 else int(np.argmin(np.abs(self.defocus)))
        a, b = ref - max(self.startFrame, 0), (min(self.stopFrame, self.nz - 1) if self.stopFrame >= 0
                                               else self.nz - 1) - ref
        stages = max(a, b) + 1
        return int(g("init loops 2") + 2 * g("init loops 3") + stages * g("intermediate loops")
                   + g("final loops") + g("residue loops"))

    # ------------------------------------------------------------------ TIE start (use TIE: 1)
    def deconvolute_espat(self, img: Plane):
        """imageData::deconvoluteEspat: divide the image spectrum by the spatial envelope (2-plane TIE)."""
        if img.alpha == 0 or img.defocus_img == 0:
            return
        B = self.B
        lam, pil = self.lam, np.pi / self.lam
        ny, nx = self.sizeY, self.sizeX
        ty = lam * np.fft.fftfreq(ny, self.dx)
        cx = (nx + nx % 2) // 2
        ixs = np.arange(nx // 2 + 1)
        tx = lam / (self.dx * nx) * np.where(ixs < cx, ixs, ixs - nx)
        TX, TY = np.meshgrid(tx, ty)
        th2 = TX ** 2 + TY ** 2
        th = np.sqrt(th2)
        thH = self.kTIEHigh * lam
        fa = img.defocus_img * pil
        if self.ast_mag != 0:
            fa = fa + self.ast_mag * pil * np.cos(2 * (np.arctan2(TY, TX) - self.ast_phi))
        e = img.alpha * th * np.abs(fa + 0.5 * self.Cs * pil * th2)
        env = np.where(th < 1.5 * thH, np.exp(np.minimum(e ** 2 - (th / thH) ** 2, 80.0)), 0.0)
        env[0, -1] = 0.0
        img.data = B.maximum(B.irfft2(B.rfft2(img.data) * B.asarray(env), (ny, nx)), 0.0)

    def init_wave_tie(self, img: Plane, dIdz):
        """waveFunction::initWaveTIE (TIE phase, then back-propagation by the plane's defocus)."""
        B = self.B
        I0 = img.data
        img0 = self.conv_gaussian(I0, self.kTIEHigh)
        d = self.inv_laplace(dIdz, self.kTIELow, self.kTIEHigh, self.Nedge)
        gy = (B.roll(d, -1, 0) - B.roll(d, 1, 0)) / (2 * self.dx)
        gx = (B.roll(d, -1, 1) - B.roll(d, 1, 1)) / (2 * self.dx)
        den = B.where(img0 > self.TIEthresh, img0, self.TIEthresh)
        gx, gy = gx / den, gy / den
        div = (-2 * np.pi / self.lam) * ((B.roll(gy, -1, 0) - B.roll(gy, 1, 0)) / (2 * self.dx) +
                                         (B.roll(gx, -1, 1) - B.roll(gx, 1, 1)) / (2 * self.dx))
        ph = self.inv_laplace(div, self.kTIELow, self.kTIEHigh, self.Nedge)
        S = (self.Sy, self.Sx)
        amp = B.full(S, float(np.sqrt(max(B.item(B.mean(B.maximum(I0, 0.0))), 0.0))))
        phase = B.zeros(S)
        sl = (slice(self.py, self.py + self.sizeY), slice(self.px, self.px + self.sizeX))
        amp[sl] = B.sqrt(B.maximum(I0, 0.0))
        phase[sl] = ph
        self.exitAmp, self.exitPhase = amp, phase
        self.copy_amp_phase_to_exit()
        # correct for the defocus offset of the reference plane
        self.w = self.exitWave
        self.back_propagate(img.defocus_img)
        w = B.ifft2(self.w) * (self.Sx * self.Sy)
        ce, se = B.cos(self.exitPhase), B.sin(self.exitPhase)
        re = w.real * ce + w.imag * se
        im = w.imag * ce - w.real * se
        self.exitAmp = B.abs(w)
        self.exitPhase = self.exitPhase + B.arctan2(im, re)
        self.exitWave = w
        self.exitRecip = False

    def tie_guess(self, i0, i1, i2):
        """frwrData::doTIEGuess -> R-value of the TIE-started 2- or 3-plane reconstruction."""
        B = self.B
        P0, P1 = self.planes[i0], self.planes[i1]
        R = 1.0
        if i2 >= 0:
            fdf = self.fitDistortion
            for jl in range(max(self.TIEloops, 1)):
                if jl == 0:
                    self.fitDistortion = 0
                tmp = Plane(index=i1, defocus_img=P1.defocus_img)
                tmp.data = 2 * P0.data - P1.data
                tmp.frame_off = list(P1.frame_off)
                self.read_plane(i2, tmp)
                P2 = self.planes[i2]
                keep = B.copy(P2.data)
                tmp2 = Plane(index=i2, defocus_img=P2.defocus_img)
                tmp2.data = 2 * P0.data - P2.data
                tmp2.frame_off = list(P2.frame_off)
                self.read_plane(i1, tmp2)
                P2.data = keep
                self.fitDistortion = fdf
            P1, P2 = self.planes[i1], self.planes[i2]
            for _ in range(max(self.TIErealignLoops, 1)):
                dIdz = (P1.data - P2.data) / (self.defocus[i1] - self.defocus[i2])
                self.init_wave_tie(P0, dIdz)
                self.fitAlphaFlag = self.alignFlag = 0
                for j in range(self.initLoops3):
                    self.reset_avg()
                    for i in (i0, i1, i2):
                        self.project(self.planes[i], self.defocus[i])
                    self.assign_avg_to_exit()
                    R = self.Rvalue / 3
                    self._say(1, f"3-Loop(TIE) {j} ({i0}, {i1}, {i2}): R-value={R:.6g}")
                    self.rvalues.append(("3-loop(TIE)", j, R))
                self.simulate_image(P2, self.defocus[i2])
                self.read_plane(i2, "wave")
                self.simulate_image(P1, self.defocus[i1])
                self.read_plane(i1, "wave")
        else:
            self.deconvolute_espat(P1)
            self.deconvolute_espat(P0)
            dIdz = (P1.data - P0.data) / (self.defocus[i1] - self.defocus[i0])
            self.init_wave_tie(P0, dIdz)
            self.fitAlphaFlag = self.alignFlag = 0
            for j in range(self.initLoops2):
                self.reset_avg()
                for i in (i0, i1):
                    self.project(self.planes[i], self.defocus[i])
                self.assign_avg_to_exit()
                R = self.Rvalue / 2
                self._say(1, f"2-Loop(TIE) {j} ({i0}, {i1}): R-value={R:.6g}")
                self.rvalues.append(("2-loop(TIE)", j, R))
        return R

    def result(self, runtime):
        B = self.B
        amp = B.to_numpy(self.exitAmp)
        ph = B.to_numpy(self.exitPhase)
        wp = amp * np.exp(1j * ph)
        sl = (slice(self.py, self.py + self.sizeY), slice(self.px, self.px + self.sizeX))
        idx = sorted(self.planes)
        n = self.nz
        dfs = np.array(self.defocus, float)
        al = np.array([self.planes[i].alpha if i in self.planes else np.nan for i in range(n)])
        dist = np.array([self.planes[i].distortions if i in self.planes else np.full(6, np.nan) for i in range(n)])
        fo = np.array([self.planes[i].frame_off if i in self.planes else [np.nan, np.nan] for i in range(n)], float)
        ws = np.array([self.planes[i].wave_shift if i in self.planes else [np.nan, np.nan] for i in range(n)], float)
        bg = B.to_numpy(self.incInt)[sl] if self.fitIncIntensity else None
        aligned, full, valid = self.aligned_frames(idx)
        fos = np.array(fo, float)
        for i in range(n):
            P = self.planes.get(i)
            if P is not None and P.shift_sub is not None and i != self.firstPlane:
                fos[i] = P.shift_sub
        return FRWR3Result(wave=wp[sl], wave_padded=wp, amplitude=amp[sl], phase=ph[sl], background=bg,
                           defoci_nm=dfs, alpha_rad=al, distortions=dist, frame_offsets=fo, wave_shifts=ws,
                           rvalues=self.rvalues, log=self.log, config=self.cfg, runtime_s=runtime,
                           backend=f"{B.name}/{B.device}", snapshots=self.snapshots, aligned=aligned,
                           aligned_full=full, aligned_valid=valid, reference_plane=int(self.firstPlane),
                           frame_offsets_subpixel=fos, pad_offset=(int(self.py), int(self.px)))

    def aligned_frames(self, idx=None):
        """Aligned, distortion-corrected experimental data (in memory; FRWR3 writes image_#.img).

        Returns (aligned ROI cut-outs, full frames warped with the fitted distortions and shifted
        to the geometry of the reference plane (zero outside the recorded area), mask of the
        pixels covered by all frames).
        """
        B = self.B
        idx = sorted(self.planes) if idx is None else idx
        n = self.nz
        aligned = np.full((n, self.sizeY, self.sizeX), np.nan, np.float32)
        full = np.zeros((n, self.NY, self.NX), np.float32)
        valid = np.ones((self.NY, self.NX), bool)
        ref = self.planes.get(self.firstPlane)
        rx, ry = (ref.frame_off if ref is not None else [self.roi[1], self.roi[0]])
        ones = B.ones((self.NY, self.NX))
        for i in idx:
            P = self.planes[i]
            if P.data is not None:
                aligned[i] = B.to_numpy(P.data)
            w = B.to_numpy(self.apply_distortions(i, P.distortions, P))
            cover = B.to_numpy(self._bilinear(ones, *self._warp_coords(P), np.ones((self.NY, self.NX)))) > 0.999
            dx, dy = int(P.frame_off[0] - rx), int(P.frame_off[1] - ry)
            out = np.zeros_like(w)
            cov = np.zeros_like(cover)
            ys = slice(max(0, -dy), min(self.NY, self.NY - dy))
            xs = slice(max(0, -dx), min(self.NX, self.NX - dx))
            ys2 = slice(ys.start + dy, ys.stop + dy)
            xs2 = slice(xs.start + dx, xs.stop + dx)
            out[ys, xs] = w[ys2, xs2]
            cov[ys, xs] = cover[ys2, xs2]
            full[i] = out
            valid &= cov
        return aligned, full, valid

    def _warp_coords(self, P):
        """Sampling coordinates of apply_distortions (for the coverage mask)."""
        dist = P.distortions
        zoom, rot = dist[DIST_ZOOM], dist[DIST_ROT]
        phiZ = np.arctan2(dist[DIST_ZOOMY], dist[DIST_ZOOMX])
        astZ = np.hypot(dist[DIST_ZOOMY], dist[DIST_ZOOMX])
        spiral, pincush = dist[DIST_SPIRAL] / 1024.0, dist[DIST_PINCUSH] / 1024.0
        r, phi = self._fr, self._fphi
        zf = 1.0 + zoom + astZ * np.cos(2.0 * (phi - phiZ)) + pincush * r
        ph = phi + rot + spiral * r
        x2 = self._fx0 + r * zf * np.cos(ph)
        y2 = self._fy0 + r * zf * np.sin(ph)
        if P.warp is not None:
            x2, y2 = x2 + P.warp[0], y2 + P.warp[1]
        return x2, y2


def B_exp(B, ph):
    return B.cos(ph) + 1j * B.sin(ph)


# ----------------------------------------------------------------------------- API
def run_frwr3(frames, params: ImagingParameters, config=None, roi=None, callback=None, init_wave=None,
              **overrides) -> FRWR3Result:
    """Run FRWR3 on full focal-series frames; see :class:`FRWR3`."""
    eng = FRWR3(frames, params, config, roi=roi, callback=callback, init_wave=init_wave, **overrides)
    return eng.run()


def frwr3_from_cdf(cdf_path, config=None, **overrides) -> FRWR3Result:
    """Convenience: read an frwr3 netCDF file (frames, parameters, ROI) and reconstruct."""
    from .frwr3_io import read_cdf
    frames, p, roi, info = read_cdf(cdf_path)
    cfg = load_config(config, **overrides)
    if cfg.get("alpha_mrad") is None and info.get("Microscope.alpha_mrad") is not None:
        cfg["alpha_mrad"] = float(info["Microscope.alpha_mrad"])
    if cfg.get("objective aperture (mrad)") is None and info.get("Microscope.ObjApert_mrad"):
        cfg["objective aperture (mrad)"] = float(info["Microscope.ObjApert_mrad"])
        cfg["objective aperture edge (mrad)"] = float(info.get("Microscope.ObjApertEdge_mrad", 0.0))
    nrm = str(info.get("DMtags.FRWR.normalized", "1")).strip()
    cfg["data normalized"] = int(float(nrm or 0) != 0)
    return run_frwr3(frames, p, cfg, roi=roi)


# ----------------------------------------------------------------------------- alignment utilities
def warp_frame(frame, dist, normalized=True, warp=None):
    """Apply FRWR3 distortions (rotation, zoom, zoomX, zoomY, spiral, pincushion) to one frame.

    Same model as ``FRWR3.apply_distortions`` (polar coordinates about the frame centre,
    bilinear interpolation).  Returns (warped frame, coverage mask).
    """
    from scipy.ndimage import map_coordinates
    frame = np.asarray(frame, float)
    NY, NX = frame.shape
    x0, y0 = (NX - 1) / 2.0, (NY - 1) / 2.0
    X, Y = np.meshgrid(np.arange(NX) - x0, np.arange(NY) - y0)
    r, phi = np.hypot(X, Y), np.arctan2(Y, X)
    d = np.asarray(dist, float)
    phiZ = np.arctan2(d[DIST_ZOOMY], d[DIST_ZOOMX])
    astZ = np.hypot(d[DIST_ZOOMY], d[DIST_ZOOMX])
    zf = 1.0 + d[DIST_ZOOM] + astZ * np.cos(2.0 * (phi - phiZ)) + d[DIST_PINCUSH] / 1024.0 * r
    ph = phi + d[DIST_ROT] + d[DIST_SPIRAL] / 1024.0 * r
    x2 = x0 + r * zf * np.cos(ph)
    y2 = y0 + r * zf * np.sin(ph)
    if warp is not None:
        x2, y2 = x2 + warp[0], y2 + warp[1]
    out = map_coordinates(frame, [y2, x2], order=1, mode="constant", cval=0.0)
    cover = (x2 >= 0) & (x2 <= NX - 1) & (y2 >= 0) & (y2 <= NY - 1)
    if not normalized:
        out = out * zf * zf
    return out * cover, cover


def scale_distortions(dist, binning):
    """Distortions fitted on b-times binned frames -> full-resolution frames
    (rotation and zooms are dimensionless; spiral and pincushion are per 1024 pixels of radius)."""
    d = np.array(dist, float)
    d[..., DIST_SPIRAL] /= binning
    d[..., DIST_PINCUSH] /= binning
    return d


def largest_valid_rectangle(mask):
    """Largest axis-aligned rectangle of True pixels -> (top, left, bottom, right) (exclusive)."""
    mask = np.asarray(mask, bool)
    ny, nx = mask.shape
    h = np.zeros(nx, int)
    best = (0, 0, 0, 0, 0)
    for y in range(ny):
        h = np.where(mask[y], h + 1, 0)
        stack = []
        for x in range(nx + 1):
            cur = h[x] if x < nx else 0
            start = x
            while stack and stack[-1][1] >= cur:
                s, hh = stack.pop()
                area = hh * (x - s)
                if area > best[0]:
                    best = (area, y - hh + 1, s, y + 1, x)
                start = s
            stack.append((start, cur))
    return best[1:]


def align_frames(frames, distortions, frame_offsets, reference_plane, binning=1, crop=True, normalized=True):
    """Warp and shift full-resolution frames with the distortions / offsets fitted by FRWR3.

    distortions, frame_offsets: from an ``FRWR3Result`` obtained on ``binning``-times binned
    frames (frame_offsets are the top-left corners (x, y) of the ROI in each warped frame).
    Returns (aligned stack, info) with the stack cropped to the largest rectangle covered by all
    frames if ``crop`` (info['crop'] = (top, left, bottom, right) in full-resolution pixels).
    """
    frames = np.asarray(frames, float)
    n, NY, NX = frames.shape
    b = int(binning)
    d_full = scale_distortions(distortions, b)
    fo = np.asarray(frame_offsets, float) * b
    ref = fo[reference_plane]
    out = np.zeros_like(frames)
    valid = np.ones((NY, NX), bool)
    for i in range(n):
        if not np.all(np.isfinite(d_full[i])) or not np.all(np.isfinite(fo[i])):
            continue
        dx, dy = int(round(fo[i, 0] - ref[0])), int(round(fo[i, 1] - ref[1]))
        fx, fy = fo[i, 0] - ref[0] - dx, fo[i, 1] - ref[1] - dy         # sub-pixel remainder
        w, cov = warp_frame(frames[i], d_full[i], normalized,
                            warp=(np.full((NY, NX), fx), np.full((NY, NX), fy)) if (fx or fy) else None)
        ys = slice(max(0, -dy), min(NY, NY - dy))
        xs = slice(max(0, -dx), min(NX, NX - dx))
        ys2 = slice(ys.start + dy, ys.stop + dy)
        xs2 = slice(xs.start + dx, xs.stop + dx)
        o = np.zeros_like(w)
        c = np.zeros_like(cov)
        o[ys, xs] = w[ys2, xs2]
        c[ys, xs] = cov[ys2, xs2]
        out[i] = o
        valid &= c
    info = {"distortions": d_full, "shifts_px": fo - ref, "reference_plane": int(reference_plane),
            "binning": b, "valid": valid}
    if crop:
        t, l, btm, r = largest_valid_rectangle(valid)
        info["crop"] = (int(t), int(l), int(btm), int(r))
        out = out[:, t:btm, l:r]
    return out, info
