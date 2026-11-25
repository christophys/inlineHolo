# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Interactive ipywidgets application (runs in Jupyter and Voila) for experimental
focal-series reconstruction:

1. upload a focal series (NXem/NeXus, TIFF stack or several TIFFs, DM3/DM4, EMD, MRC, NPY)
2. inspect / edit the extracted parameters (pixel size, voltage, defoci, Cs, astigmatism,
   coherence, ...)
3. fit defocus, two-fold astigmatism (and optionally Cs) to the Thon rings
4. align the series (drift correction)
5. reconstruct with any of the implemented methods (optionally half-set FRC)
6. export focal series, parameters and reconstructions as NXem
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import threading
import time

import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML

from . import ctffit, align, nxem
from .eln import WorkflowLog


def _md_to_html(md):
    """Minimal markdown (headings, bold, inline code, numbered list) for the workflow description."""
    import re
    out, in_list = [], False
    for line in md.splitlines():
        t = html.escape(line)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"`(.+?)`", r"<code style='font-size:11px'>\1</code>", t)
        m = re.match(r"^(\d+)\. (.*)$", t)
        if m:
            if not in_list:
                out.append("<ol style='margin:2px 0 2px 18px;padding:0'>")
                in_list = True
            out.append(f"<li>{m.group(2)}</li>")
            continue
        if in_list:
            out.append("</ol>")
            in_list = False
        if t.startswith("## "):
            out.append(f"<div style='font-weight:bold;margin-top:6px'>{t[3:]}</div>")
        elif t.strip():
            out.append(f"<div>{t}</div>")
    if in_list:
        out.append("</ol>")
    return "<div style='font-size:12px;line-height:1.4'>" + "".join(out) + "</div>"
from .app_dih import DIHTabMixin
from .app_nomad import NomadMixin
from .control import StopRequested
from .frc import binomial_split, frc, half_bit_threshold, align_global_phase
from .io_formats import load_series, load_many
from .optics import ABERRATION_NAMES, wavelength_nm
from .params import ImagingParameters
from .reconstruct import METHODS, reconstruct

def LAYOUT_LABEL():  # noqa: N802  -- a new Layout widget per use (shared widgets break multi-user servers)
    return W.Layout(width="190px")
STYLE = {"description_width": "190px"}


def _ft(desc, value=0.0, step=0.1, **kw):
    return W.FloatText(value=value, description=desc, step=step, style=STYLE,
                       layout=W.Layout(width="330px"), **kw)


def _circle(r):
    from matplotlib.patches import Circle
    return Circle((0, 0), r, fill=False, color="orange", lw=0.8)


def _bin(stack, b):
    if b <= 1:
        return stack
    n, ny, nx = stack.shape
    ny2, nx2 = ny // b * b, nx // b * b
    return stack[:, :ny2, :nx2].reshape(n, ny2 // b, b, nx2 // b, b).mean((2, 4))


class App(DIHTabMixin, NomadMixin):
    def __init__(self, data_dir=None, server_mode=False, work_dir=None, streaming_upload=None):
        """data_dir: folder with example data (tab 1).  server_mode: the app is shared by several users on a
        server (NOMAD dashboard): no access to the server's file system except data_dir and work_dir; all
        written files go to work_dir/output (work_dir: a private folder of this session, default: cwd).
        streaming_upload: browser uploads in chunks with progress (ipyvuetify FileInput, used by Solara; no size
        limit) instead of the ipywidgets FileUpload (one message); default: on in server mode."""
        self.data_dir = data_dir
        self.server_mode = bool(server_mode)
        self.streaming_upload = self.server_mode if streaming_upload is None else bool(streaming_upload)
        self.workflow_log = WorkflowLog()   # processing steps of the session (ELN, workflow, NXem description)
        self.work_dir = os.path.abspath(work_dir or os.getcwd())
        self.raw = None          # stack as loaded
        self.stack = None        # working stack (binned/cropped/aligned)
        self.meta = {}
        self.rec = None
        self.recs = []
        self.init_rec = None     # wave of the FRWR alignment (tab 4b), full resolution of the aligned stack
        self.maps = []           # derived maps not contained in a Reconstruction (DIH strain, ...)
        self.curves = []         # 1-D results (half-set FRC)
        self.nx_meta = nxem.default_metadata()
        self.fits = {}
        self._stop = threading.Event()
        self._busy = False
        self._thon_cache = None
        self._q_rel = (0.06, 0.8)
        self._suspend_thon = False
        self._build()

    # ------------------------------------------------------------------ UI build
    def _build(self):
        # ---------- load tab
        self.w_upload_progress = W.HTML()
        self.w_upload = None
        if self.streaming_upload:
            from .nomad_project import upload_button
            self.w_upload = upload_button(accept=",".join(self._EXT), label="Upload focal series file(s)")
            if self.w_upload is None:                # ipyvuetify not available (plain Jupyter / Voila)
                self.streaming_upload = False
        if self.w_upload is None:
            self.w_upload = W.FileUpload(accept=",".join(self._EXT), multiple=True, description="Upload file(s)",
                                         layout=W.Layout(width="220px"))
        self.w_md_box = self._build_metadata_form()      # NXem metadata (shown in tabs 1 and 6)
        self.w_demo = W.Dropdown(options=self._data_files(), description="or file in data folder",
                                 style={"description_width": "150px"}, layout=W.Layout(width="520px"))
        self.w_path = W.Text(value="", placeholder="path of a local file (no size limit), e.g. ../data/stackExp.dm3",
                             description="or local path", style={"description_width": "150px"},
                             layout=W.Layout(width="620px"))
        self.w_path_load = W.Button(description="load", icon="folder-open")
        self.w_path_load.on_click(lambda _: self._load_path(self.w_path.value.strip()))
        # simple file browser on the host running the kernel
        start = os.path.abspath(self.data_dir) if self.data_dir and os.path.isdir(self.data_dir) else self.work_dir
        self.w_br_dir = W.Text(value=start, description="folder", style={"description_width": "60px"},
                               layout=W.Layout(width="560px"))
        self.w_br_up = W.Button(description="up", icon="level-up", layout=W.Layout(width="70px"))
        self.w_br_list = W.Select(options=[], rows=9, layout=W.Layout(width="640px"))
        self.w_br_open = W.Button(description="load selected file", icon="folder-open", button_style="primary")
        self.w_br_dir.observe(lambda ch: self._browse(), names="value")
        self.w_br_up.on_click(lambda _: setattr(self.w_br_dir, "value", os.path.dirname(self.w_br_dir.value)))
        self.w_br_list.observe(self._on_browse_select, names="value")
        self.w_br_open.on_click(lambda _: self._load_path(self.w_path.value.strip()))
        self.w_br_folder = W.Button(description="load all frames of this folder", icon="files-o",
                                    layout=W.Layout(width="260px"),
                                    tooltip="single-image files of the folder as one series, ordered by the "
                                            "defocus in the file names (e.g. F-1000nm.dm3) or by their numbers")
        self.w_br_folder.on_click(lambda _: self._load_folder(self.w_br_dir.value))
        self.w_browser = W.Accordion(children=[W.VBox([W.HBox([self.w_br_dir, self.w_br_up]), self.w_br_list,
                                                       W.HBox([self.w_br_open, self.w_br_folder]),
                                                       W.HTML("<small>An indexed .img frame (e.g. image_3.img) "
                                                              "loads the whole series image_0.img, image_1.img, "
                                                              "...</small>")])],
                                     titles=("browse files on the computer running this app",))
        self.w_browser.selected_index = None
        self._browse()
        # conversion of non-NXem files to NXem
        self.w_conv_name = W.Text(value="", description="NXem file", style={"description_width": "150px"},
                                  layout=W.Layout(width="620px"))
        self.w_conv = W.Button(description="convert to NXem (.nxs)", icon="save", button_style="warning")
        self.w_conv.on_click(lambda _: self._convert())
        self.w_conv_status = W.HTML()
        self.w_conv_box = W.VBox([W.HTML("<b>Convert to NXem.</b> The loaded file is not a NeXus file. Check the "
                                         "imaging parameters in tab 2 and fill in the metadata required by NXem "
                                         "(marked *; prefilled from the file where possible — please verify), then "
                                         "write the full-resolution series as an NXem file:"),
                                  self.w_md_box, W.HBox([self.w_conv_name, self.w_conv]), self.w_conv_status],
                                 layout=W.Layout(display="none", border="1px solid #e0b050", padding="6px"))
        self.w_bin = W.Dropdown(options=[1, 2, 4, 8], value=1, description="binning", style={"description_width": "70px"},
                                layout=W.Layout(width="160px"))
        self.w_crop = W.IntText(value=0, description="central crop (px, 0=off)", style={"description_width": "160px"},
                                layout=W.Layout(width="260px"))
        self.w_apply_prep = W.Button(description="apply binning/crop", icon="crop")
        self.w_load_status = W.HTML()
        self.w_meta = W.Output(layout=W.Layout(max_height="300px", overflow="auto"))
        if self.streaming_upload:
            self.w_upload.observe(self._on_stream_files, names="file_info")
        else:
            self.w_upload.observe(self._on_upload, names="value")
        self.w_demo.observe(self._on_demo, names="value")
        self.w_apply_prep.on_click(lambda _: self._prepare())
        self._build_nomad()
        from .nomad_project import ProjectPicker
        self.project = ProjectPicker(folder="focal_series", on_change=self._project_changed)
        intro = ("<b>1. Load a focal series</b> — NXem/NeXus, TIFF stack or several single images (one per "
                 "defocus), DM3/DM4, FRWR3/QSTEM .img frames, EMD, MRC, NPY — ")
        intro += ("from your computer (choose the file(s) below; they are transferred in chunks with progress) or "
                  "from a NOMAD upload." if self.streaming_upload else
                  "from this computer, the data folder, or a NOMAD upload. Large files (e.g. 140 MB DM3 stacks) are "
                  "best loaded by path or from the data folder; browser uploads are limited by the web server's "
                  "message size.")
        tab_load = W.VBox([W.HTML(intro),
                           (W.VBox([self.project.box,
                                    W.HBox([W.HTML("<b>From your computer:</b>", layout=W.Layout(width="150px")),
                                            W.Box([self.w_upload], layout=W.Layout(width="400px"))]),
                                    self.w_upload_progress]) if self.streaming_upload else
                            W.HBox([self.w_upload, self.w_upload_progress])),
                           *([self.w_demo] if len(self.w_demo.options) > 1 or not self.server_mode else []),
                           *([] if self.server_mode else [W.HBox([self.w_path, self.w_path_load]), self.w_browser]),
                           self.nomad_import_box(),
                           W.HBox([self.w_bin, self.w_crop, self.w_apply_prep]),
                           self.w_load_status, self.w_conv_box,
                           W.Accordion(children=[self.w_meta], titles=("extracted metadata",))])

        # ---------- parameter tab
        self.w_kv = _ft("voltage (kV)", 300.0, 10)
        self.w_pix = _ft("pixel size (nm)", 0.05, 0.001)
        self.w_c3 = _ft("C3 = Cs (µm)", 0.0, 1.0)
        self.w_c5 = _ft("C5 (mm)", 0.0, 0.1)
        self.w_a1 = _ft("global A1 magnitude (nm)", 0.0, 0.1)
        self.w_a1ang = _ft("global A1 azimuth (deg)", 0.0, 1.0)
        self.w_b2 = _ft("coma C21a / B2 (nm)", 0.0, 1.0)
        self.w_b2b = _ft("coma C21b (nm)", 0.0, 1.0)
        self.w_fs = _ft("focal spread, rms (nm)", 3.0, 0.1)
        self.w_alpha = _ft("convergence, rms (mrad)", 0.1, 0.01)
        self.w_ap = _ft("obj. aperture (mrad, 0 = none)", 0.0, 1.0)
        self.w_mtf = W.Dropdown(options=["none", "sinc"], value="none", description="detector MTF",
                                style=STYLE, layout=W.Layout(width="330px"))
        self.w_lam = W.HTML()
        for w in (self.w_kv,):
            w.observe(lambda _: self._update_lambda(), names="value")
        self.w_series_start = _ft("Δf_offset at frame n0 (nm)", 0.0, 1)
        self.w_series_step = _ft("defocus step (nm)", 2.0, 0.1)
        self.w_series_m = _ft("non-linearity m (1 = linear)", 1.0, 0.5)
        self.w_series_n0 = W.IntText(value=-1, description="central frame n0 (−1: middle)", style=STYLE,
                                     layout=W.Layout(width="330px"))
        self.w_fill = W.Button(description="fill series (FRWR Eq. 1)", icon="list-ol", layout=W.Layout(width="210px"))
        self.w_fill.on_click(lambda _: self._fill_linear())
        self.w_table = W.VBox()
        self.w_param_json = W.Textarea(layout=W.Layout(width="680px", height="120px"))
        self.w_show_json = W.Button(description="show parameters as JSON")
        self.w_show_json.on_click(lambda _: setattr(self.w_param_json, "value",
                                                    json.dumps(self.params().to_dict(), indent=1)))
        tab_par = W.VBox([
            W.HTML("<b>2. Imaging parameters</b> (Krivanek notation; C1 &gt; 0 = overfocus; "
                   "coherence parameters are standard deviations of Gaussian distributions)"),
            W.HBox([W.VBox([self.w_kv, self.w_lam, self.w_pix, self.w_c3, self.w_c5, self.w_mtf]),
                    W.VBox([self.w_a1, self.w_a1ang, self.w_b2, self.w_b2b, self.w_fs, self.w_alpha, self.w_ap])]),
            W.HTML("<b>Per-image parameters</b> (defocus C1 and additional two-fold astigmatism C12a/C12b in nm; "
                   "untick to exclude an image)"),
            W.HTML("Fill the defocus column with the FRWR series Δf<sub>n</sub> = Δf<sub>offset</sub> + step · "
                   "sign(n − n<sub>0</sub>) · |n − n<sub>0</sub>|<sup>m</sup> (m = 1 linear, m = 2 quadratic, m = 3 "
                   "cubic: small steps near focus, large steps at large defocus):"),
            W.HBox([W.VBox([self.w_series_start, self.w_series_step]), W.VBox([self.w_series_m, self.w_series_n0]),
                    self.w_fill]),
            self.w_table, W.HBox([self.w_show_json]), self.w_param_json])

        # ---------- Thon tab
        self.w_idx = W.IntSlider(min=0, max=0, description="image", continuous_update=False)
        self.w_qmin = W.FloatSlider(value=0.1, min=0.0, max=5.0, step=0.005, description="q_min (1/nm)",
                                    continuous_update=True, readout_format=".3f",
                                    style={"description_width": "90px"}, layout=W.Layout(width="420px"))
        self.w_qmax = W.FloatSlider(value=4.0, min=0.01, max=5.0, step=0.005, description="q_max (1/nm)",
                                    continuous_update=True, readout_format=".3f",
                                    style={"description_width": "90px"}, layout=W.Layout(width="420px"))
        self.w_series_model = W.Dropdown(options=[("offset + scale × table defoci", "scale"),
                                                  ("FRWR Eq. 1 with m of tab 2 (offset, step)", "power"),
                                                  ("FRWR Eq. 1, fit also m", "power_fit")],
                                         value="scale", description="series model", style={"description_width": "90px"},
                                         layout=W.Layout(width="420px"))
        self.w_thon_stop = W.Button(description="stop", icon="stop", button_style="danger", disabled=True)
        self.w_thon_stop.on_click(lambda _: self._request_stop())
        self.w_thon_prog = W.FloatProgress(min=0, max=1, layout=W.Layout(width="300px"))
        self.w_thon_status = W.HTML()
        self.w_win = _ft("search window ± (nm)", 20.0, 1)
        self.w_amp = _ft("amplitude contrast w", 0.07, 0.01)
        self.w_fitastig = W.Checkbox(value=True, description="fit astigmatism", layout=W.Layout(width="auto"))
        self.w_fitc3 = W.Checkbox(value=False, description="also fit C3 (series)", layout=W.Layout(width="auto"))
        self.w_fit1 = W.Button(description="fit this image", icon="bullseye")
        self.w_fitall = W.Button(description="fit all images", icon="th")
        self.w_applyfit = W.Button(description="apply fits to parameters", icon="check", button_style="success")
        self.w_fitglob = W.Button(description="series fit", icon="link",
                                  tooltip="all images simultaneously: C1_n from the series model, common two-fold "
                                          "astigmatism; robust for images close to focus")
        self.w_fitglob.on_click(lambda _: self._fit_global(block=False))
        self.w_thon_out = W.Output()
        self.w_fit_out = W.Output()
        self.w_idx.observe(lambda _: self._show_thon(), names="value")
        for w in (self.w_qmin, self.w_qmax):
            w.observe(lambda _: self._on_q(), names="value")
        self.w_fit1.on_click(lambda _: self._fit(single=True, block=False))
        self.w_fitall.on_click(lambda _: self._fit(single=False, block=False))
        self.w_applyfit.on_click(lambda _: self._apply_fits())
        self.w_refresh = W.Button(description="refresh view", icon="refresh")
        self.w_refresh.on_click(lambda _: self._show_thon())
        self.w_pix.observe(lambda _: (setattr(self, "_thon_cache", None), self._update_q_ranges()), names="value")
        tab_thon = W.VBox([
            W.HTML("<b>3. Thon rings</b>: left half of the diffractogram = data (background removed), "
                   "right half = sin²(χ) model of the current parameters. Fitting searches C1 within ± window "
                   "around the current value, then refines C1 and A1 (Nelder–Mead)."),
            W.HBox([self.w_idx, self.w_refresh]), W.HBox([self.w_qmin, self.w_qmax]),
            W.HBox([self.w_win, self.w_amp]), W.HBox([self.w_fitastig, self.w_fitc3, self.w_series_model]),
            W.HBox([self.w_fit1, self.w_fitall, self.w_fitglob, self.w_applyfit, self.w_thon_stop]),
            W.HBox([self.w_thon_prog, self.w_thon_status]),
            self.w_thon_out, self.w_fit_out])

        # ---------- align tab
        self.w_hp = _ft("high-pass σ (px, 0=off)", 0.0, 1)
        self.w_seq = W.Checkbox(value=True, description="sequential (neighbour) registration", layout=W.Layout(width="auto"))
        self.w_align = W.Button(description="align series", icon="arrows")
        self.w_unalign = W.Button(description="undo alignment", icon="undo")
        self.w_align_out = W.Output()
        self.w_align.on_click(lambda _: self._align())
        self.w_unalign.on_click(lambda _: self._prepare())
        self.w_fa_bin = W.Dropdown(options=[1, 2, 4], value=2, description="binning for the fit",
                                   style={"description_width": "130px"}, layout=W.Layout(width="230px"))
        self.w_fa_dist = W.Checkbox(value=True, description="fit defocus-dependent distortions", indent=False, layout=W.Layout(width="auto"))
        self.w_fa_crop = W.Checkbox(value=True, description="crop to common area", indent=False, layout=W.Layout(width="auto"))
        self.w_fa_df = W.Checkbox(value=True, description="use refined defoci", indent=False, layout=W.Layout(width="auto"))
        self.w_fa_shift = W.IntText(value=40, description="max. drift (px)", style={"description_width": "100px"},
                                    layout=W.Layout(width="190px"))
        self.w_fa = W.Button(description="FRWR3 alignment", icon="magic", button_style="info")
        self.w_fa.on_click(lambda _: self._frwr3_align(block=False))
        self.w_fa_stop = W.Button(description="stop", icon="stop", button_style="danger", disabled=True)
        self.w_fa_stop.on_click(lambda _: self._request_stop())
        self.w_fa_prog = W.FloatProgress(min=0, max=1, layout=W.Layout(width="300px"))
        self.w_fa_status = W.HTML()
        self.w_fa_loops = W.IntText(value=10, description="final loops", style={"description_width": "80px"},
                                    layout=W.Layout(width="170px"))
        self.w_fa_unwrap = W.Checkbox(value=False, description="unwrap phase", indent=False, layout=W.Layout(width="auto"))
        self.w_fa_unwrap.observe(lambda _: self._show_init_rec(), names="value")
        self.w_fa_wave_out = W.Output()
        self.w_fa_fname = W.Text(value="frwr_aligned_initial_reconstruction.nxs", description="file name",
                                 layout=W.Layout(width="480px"))
        self.w_fa_save = W.Button(description="write NXem (.nxs)", icon="save", button_style="success",
                                  disabled=True)
        self.w_fa_save.on_click(lambda _: self._save_alignment())
        self.w_fa_save_out = W.HTML()
        self.w_fa_to5 = W.Button(description="use as start wave in tab 5", icon="share", disabled=True,
                                 layout=W.Layout(width="230px"))
        self.w_fa_to5.on_click(lambda _: self._use_init_rec())
        tab_align = W.VBox([W.HTML("<b>4a. Drift correction</b> by (sub-pixel) phase cross-correlation."),
                            W.HBox([self.w_hp, self.w_seq]), W.HBox([self.w_align, self.w_unalign]),
                            W.HTML("<b>4b. FRWR Alignment and Initial Reconstruction</b>: drift and "
                                   "defocus-dependent image distortions (rotation, magnification, elliptical "
                                   "magnification, spiral and pincushion distortion) are fitted self-consistently by "
                                   "FRWR3 on a binned copy together with the exit wave, and applied to the "
                                   "full-resolution frames, which are then cropped to the area covered by all "
                                   "images. The exit wave of this run (initial reconstruction) is interpolated to "
                                   "the full resolution; it can be stored and used as start wave in tab 5."),
                            W.HBox([self.w_fa_bin, self.w_fa_shift, self.w_fa_loops]),
                            W.HBox([self.w_fa_dist, self.w_fa_crop, self.w_fa_df]),
                            W.HBox([self.w_fa, self.w_fa_stop, self.w_fa_prog, self.w_fa_status]),
                            self.w_align_out,
                            W.HBox([W.HTML("<b>Initial reconstruction</b> (full resolution)"), self.w_fa_unwrap,
                                    self.w_fa_to5]),
                            self.w_fa_wave_out,
                            W.HTML("Store the aligned series, the initial reconstruction (exit wave, amplitude, "
                                   "phase) and all alignment results (distortions, drift, crop, refined defoci, "
                                   "astigmatism, convergence) with the NXem metadata of tab 1/6:"),
                            W.HBox([self.w_fa_fname, self.w_fa_save]), self.w_fa_save_out])

        # ---------- reconstruction tab
        import re as _re
        from .references import references_html

        def _plain(d):            # method description without citations (they are listed below)
            d = _re.sub(r"\s*\((?:[^()]*\d{4}[^()]*|[^()]*et al\.?[^()]*|this work|Teague[^()]*)\)", "", d)
            return d.strip()

        shown = [k for k in METHODS if k not in ("gd", "hybrid", "mcnn")]
        opts = [(f"{k} — {_plain(METHODS[k]['description'])}", k) for k in shown]
        self.w_method = W.Dropdown(options=opts, value="frwr", description="method",
                                   layout=W.Layout(width="560px"))
        self.w_iter = W.IntText(value=100, description="iterations")
        self.w_init = W.Dropdown(options=[("flat", "flat"), ("back-propagated image", "backprop"),
                                          ("PAM", "pam"), ("Schiske", "schiske"), ("TIE (polynomial)", "tie_poly"),
                                          ("TIE (full)", "tie_full"), ("MFTIE", "mftie"), ("GF-TIE", "gftie"),
                                          ("FRWR initial reconstruction (tab 4b)", "frwr_align"),
                                          ("last reconstruction (refine)", "last")],
                                 value="flat", description="start wave", layout=W.Layout(width="420px"))
        self.w_allframes = W.Checkbox(value=False, indent=False,
                                      description="FRWR: start with all frames (no 2-/3-plane start)", layout=W.Layout(width="auto"))
        self.w_init.observe(lambda ch: setattr(self.w_allframes, "value", ch["new"] in ("frwr_align", "last")),
                            names="value")
        self.w_reg = W.Text(value="auto", description="Wiener reg.")
        self.w_bg = W.Checkbox(value=False, description="incoherent background (FRWR)", layout=W.Layout(width="auto"))
        self.w_tv = W.FloatText(value=0.0, description="TV reg. (gd)")
        # FRWR3 (frwr, frwr_gf, gpfrwr): backend + YAML configuration with the frwr3 key names
        from .backend import available_backends
        self.w_backend = W.Dropdown(options=["auto", "numpy", "torch", "cupy"], value="auto",
                                    description="backend", style={"description_width": "70px"})
        self.w_backend_info = W.HTML(f"<small>available: {', '.join(available_backends())}</small>")
        self.w_frwr3_yaml = W.Textarea(
            value=("# FRWR3 keys (same names as the frwr3 control file), e.g.\n"
                   "# fit defocus: 2\n# max shift: 10\n# padding: 0.25\n# apply GF: 1\n"
                   "# fit incoherent background: [1, 0.1, 0.98]\n"),
            layout=W.Layout(width="560px", height="120px"))
        self.w_frwr3_upload = W.FileUpload(accept=".yaml,.yml,.conf", multiple=False,
                                           description="load .yaml/.conf")
        self.w_frwr3_upload.observe(lambda _: self._load_frwr3_conf(), names="value")
        self.w_frwr3_box = W.Accordion(children=[W.VBox([
            W.HTML("FRWR3 options (methods <code>frwr</code>, <code>frwr_gf</code>, <code>gpfrwr</code>); "
                   "the reconstruction runs in memory, on the GPU if available."),
            W.HBox([self.w_backend, self.w_backend_info]), self.w_frwr3_upload, self.w_frwr3_yaml])])
        self.w_frwr3_box.set_title(0, "FRWR3 settings (YAML)")
        self.w_refs_box = W.Accordion(children=[W.HTML(references_html(shown))],
                                      titles=("References for the reconstruction methods",))
        self.w_refs_box.selected_index = None
        self.w_frwr3_box.selected_index = None
        self.w_run = W.Button(description="reconstruct", icon="play", button_style="primary")
        self.w_half = W.Button(description="half-set FRC", icon="balance-scale")
        self.w_unwrap = W.Checkbox(value=False, description="unwrap phase", layout=W.Layout(width="auto"))
        self.w_prog = W.IntProgress(min=0, max=100, layout=W.Layout(width="400px"))
        self.w_rec_status = W.HTML()
        self.w_rec_out = W.Output()
        self.w_run.on_click(lambda _: self._run(block=False))
        self.w_rec_stop = W.Button(description="stop (keep current result)", icon="stop", button_style="danger",
                                   disabled=True, layout=W.Layout(width="230px"))
        self.w_rec_stop.on_click(lambda _: self._request_stop())
        self.w_half.on_click(lambda _: self._halfset())
        self.w_unwrap.observe(lambda _: self._show_rec(), names="value")
        tab_rec = W.VBox([W.HTML("<b>5. Reconstruction / refinement</b> of the exit wave in the object plane "
                                 "(C1 = 0, all given aberrations removed). A start wave from tab 4b or the last "
                                 "reconstruction is refined; FRWR can then use all frames from the first loop."),
                          self.w_method, W.HBox([self.w_iter, self.w_init]), self.w_allframes,
                          W.HBox([self.w_reg, self.w_tv]),
                          self.w_bg, self.w_frwr3_box, W.HBox([self.w_run, self.w_half, self.w_unwrap]),
                          W.HBox([self.w_rec_stop, self.w_prog, self.w_rec_status]), self.w_refs_box,
                          self.w_rec_out])
        tab_dih = self._build_dih_tab()

        # ---------- export tab
        self.w_fname = W.Text(value="focal_series_reconstruction.nxs", description="file name",
                              layout=W.Layout(width="480px"))
        self.w_save = W.Button(description="write NXem", icon="save", button_style="success")
        self.w_save_out = W.HTML()
        self.w_save.on_click(lambda _: self._save())
        self.w_derived = W.Checkbox(value=True, description="also store amplitude and phase maps", indent=False, layout=W.Layout(width="auto"))
        self.w_placeholder = W.Checkbox(value=False, indent=False,
                                        description="write 'unknown' for missing required metadata", layout=W.Layout(width="auto"))
        self.w_validate = W.Checkbox(value=True, indent=False, description="validate with pynxtools after writing", layout=W.Layout(width="auto"))
        self.w_eln = W.Checkbox(value=True, indent=False, layout=W.Layout(width="auto"),
                                description="also write a NOMAD ELN entry and the workflow (archive files)")
        self.w_eln_core = W.Checkbox(value=False, indent=False, layout=W.Layout(width="auto"),
                                     description="core NOMAD ELN (if the inlineHolo plugin is not installed in "
                                                 "the target NOMAD)")
        self.w_wf_svg = W.HTML()
        self.w_wf_desc = W.HTML()
        self.w_wf_refresh = W.Button(description="preview workflow", icon="sitemap", layout=W.Layout(width="180px"))
        self.w_wf_refresh.on_click(lambda _: self._show_workflow())
        self.w_wf_box = W.Accordion(children=[W.VBox([
            W.HTML("<small>The processing steps of this session as NOMAD will show them on the <i>workflow</i> card "
                   "of the entry (inputs → tasks → outputs), and the description written into the ELN entry and into "
                   "the NXem file.</small>"), self.w_wf_refresh,
            W.Box([self.w_wf_svg], layout=W.Layout(overflow="auto", border="1px solid #ddd", padding="4px")),
            self.w_wf_desc])], titles=("workflow of this session (preview)",))
        self.w_wf_box.selected_index = None
        tab_exp = W.VBox([W.HTML("<b>6. Export</b> focal series (as used), parameters, all reconstructions of this "
                                 "session (exit waves, amplitude/phase, incoherent background, diffusion field, "
                                 "refined FRWR3 parameters, convergence), DIH strain maps and half-set FRC curves "
                                 "to an NXem/NeXus HDF5 file that complies with the NXem application definition "
                                 "(pynxtools / pynxtools-em). Fields marked <b>*</b> are required by NXem."),
                          self.w_md_box,
                          W.HBox([self.w_derived, self.w_placeholder, self.w_validate]),
                          W.HBox([self.w_eln, self.w_eln_core]), W.HBox([self.w_fname, self.w_save]), self.w_save_out,
                          self.w_wf_box, self.nomad_export_box()])

        self.tabs = W.Tab(children=[tab_load, tab_par, tab_thon, tab_align, tab_rec, tab_exp, tab_dih],
                          titles=("1 Load", "2 Parameters", "3 Thon rings", "4 Align/Pre-reconstruct",
                                  "5 Reconstruct/Refine", "6 Export", "7 DIH strain"))
        self.ui = W.VBox([W.HTML("<h3>Inline electron holography — focal-series reconstruction</h3>"), self.tabs])
        self._update_lambda()

    def display(self):
        display(self.ui)
        return self

    # ------------------------------------------------------------------ helpers
    def _update_lambda(self):
        lam = wavelength_nm(self.w_kv.value * 1e3) if self.w_kv.value > 0 else np.nan
        self.w_lam.value = f"<span style='margin-left:200px'>λ = {lam * 1e3:.4f} pm</span>"

    def _status(self, w, msg, ok=True):
        color = "green" if ok else "crimson"
        w.value = f"<span style='color:{color}'>{msg}</span>"

    def _make_table(self, defoci, aa=None, ab=None):
        n = len(defoci)
        aa = np.zeros(n) if aa is None else aa
        ab = np.zeros(n) if ab is None else ab
        self.rows = []
        hdr = W.HBox([W.HTML("<b>#</b>", layout=W.Layout(width="40px")),
                      W.HTML("<b>use</b>", layout=W.Layout(width="60px")),
                      W.HTML("<b>C1 (nm)</b>", layout=W.Layout(width="130px")),
                      W.HTML("<b>C12a (nm)</b>", layout=W.Layout(width="130px")),
                      W.HTML("<b>C12b (nm)</b>", layout=W.Layout(width="130px")),
                      W.HTML("<b>mean counts</b>", layout=W.Layout(width="120px"))])
        lines = [hdr]
        for i in range(n):
            d = W.FloatText(value=float(np.nan_to_num(defoci[i])), layout=W.Layout(width="130px"))
            a = W.FloatText(value=float(aa[i]), layout=W.Layout(width="130px"))
            b = W.FloatText(value=float(ab[i]), layout=W.Layout(width="130px"))
            u = W.Checkbox(value=True, indent=False, layout=W.Layout(width="60px"))
            mc = f"{self.stack[i].mean():.4g}" if self.stack is not None and i < len(self.stack) else ""
            self.rows.append((u, d, a, b))
            lines.append(W.HBox([W.HTML(str(i), layout=W.Layout(width="40px")), u, d, a, b,
                                 W.HTML(mc, layout=W.Layout(width="120px"))]))
        self.w_table.children = lines

    def _series_n0(self, n=None):
        n = len(self.rows) if n is None else n
        v = int(self.w_series_n0.value)
        return (n - 1) // 2 if v < 0 or v >= n else v

    def series_values(self, n=None):
        """Defocus series of FRWR Eq. (1) from the fill widgets of tab 2."""
        n = len(self.rows) if n is None else n
        return ctffit.series_defoci(n, self.w_series_start.value, self.w_series_step.value,
                                    self.w_series_m.value, self._series_n0(n))

    def _fill_linear(self):
        """Fill the defocus column with FRWR Eq. (1) (linear for m = 1)."""
        for (u, d, a, b), v in zip(self.rows, self.series_values()):
            d.value = round(float(v), 3)
        self._show_thon()

    # ------------------------------------------------------------------ jobs, stop, thread-safe output
    def _request_stop(self):
        if not self._busy:
            return
        self._stop.set()
        self._job_status("stop requested — finishing the current step ...")

    def _job_status(self, msg):
        w = getattr(self, "_job_status_w", None)
        if w is not None:
            w.value = f"<span style='color:#b26a00'>{msg}</span>"

    def _check_stop(self):
        if self._stop.is_set():
            raise StopRequested()

    def _start_job(self, fn, run_buttons, stop_button, status_w, block=True):
        """Run fn() (in a background thread unless block) with the stop button enabled."""
        if self._busy:
            self._status(status_w, "another computation is still running — stop it first", ok=False)
            return
        self._stop.clear()
        self._busy = True
        self._job_status_w = status_w
        old = [(b, b.disabled) for b in run_buttons]

        def job():
            for b in run_buttons:
                b.disabled = True
            stop_button.disabled = False
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                self._status(status_w, f"error: {e!r}", ok=False)
            finally:
                for b, d in old:
                    b.disabled = d
                stop_button.disabled = True
                self._busy = False

        if block:
            job()
        else:
            threading.Thread(target=job, daemon=True).start()

    @staticmethod
    def _png_out(out, fig, dpi=90, append=False):
        """Show a matplotlib Figure in an Output widget (thread-safe: no capture context)."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        item = {"output_type": "display_data", "metadata": {},
                "data": {"image/png": base64.b64encode(buf.getvalue()).decode(), "text/plain": "<figure>"}}
        out.outputs = (tuple(out.outputs) if append else ()) + (item,)

    @staticmethod
    def _text_out(out, text):
        out.outputs = ({"output_type": "stream", "name": "stdout", "text": text},)

    @staticmethod
    def _frwr_eta(st, eng, label, t_start, extra_s=0.0):
        """Progress and remaining time of an FRWR3 run from its loop schedule (ScheduleETA).

        Loops are timed online by kind (alignment / re-reading / projection loops of the
        intermediate stages, final loops) and per image, and the stages still to come are
        predicted with their actual (growing) number of images."""
        from .control import ScheduleETA
        if st.get("eta") is None:
            st["eta"] = ScheduleETA(eng.planned_schedule(), t0=t_start)
            st["replanned"] = False
        eta = st["eta"]
        eta.update()
        if not st["replanned"] and getattr(eng, "_init_done", False) and label and not label.endswith("loop"):
            eta.replan(eng.planned_schedule(from_current=True)[1:])     # planes chosen by the initialization
            st["replanned"] = True
        el = time.time() - t_start
        rem = eta.remaining() + extra_s
        frac = el / max(el + rem, 1e-9)
        if not eta.ready:
            return min(frac, 0.05), f"{el:.0f} s elapsed, estimating the remaining time ..."
        return frac, f"{el:.0f} s elapsed, ≈ {rem:.0f} s left"

    @staticmethod
    def _eta(t0, frac):
        el = time.time() - t0
        if frac <= 0:
            return f"{el:.0f} s elapsed"
        return f"{el:.0f} s elapsed, ≈ {max(el / frac - el, 0):.0f} s left"

    def params(self, all_images=False) -> ImagingParameters:
        ab = {}
        if self.w_c3.value:
            ab["c_3_0"] = self.w_c3.value * 1e3
        if self.w_c5.value:
            ab["c_5_0"] = self.w_c5.value * 1e6
        if self.w_a1.value:
            ang = np.deg2rad(self.w_a1ang.value)
            ab["c_1_2_a"] = self.w_a1.value * np.cos(2 * ang)
            ab["c_1_2_b"] = self.w_a1.value * np.sin(2 * ang)
        if self.w_b2.value:
            ab["c_2_1_a"] = self.w_b2.value
        if self.w_b2b.value:
            ab["c_2_1_b"] = self.w_b2b.value
        rows = self.rows if all_images else [r for r in self.rows if r[0].value]
        return ImagingParameters(
            voltage_V=self.w_kv.value * 1e3, pixel_size_nm=self.w_pix.value,
            defoci_nm=np.array([r[1].value for r in rows]), aberrations=ab,
            astig_a_nm=np.array([r[2].value for r in rows]), astig_b_nm=np.array([r[3].value for r in rows]),
            focal_spread_nm=self.w_fs.value, convergence_rad=self.w_alpha.value * 1e-3,
            aperture_rad=(self.w_ap.value * 1e-3) or None,
            mtf={"type": "sinc"} if self.w_mtf.value == "sinc" else None)

    def used_stack(self):
        sel = [k for k, r in enumerate(self.rows) if r[0].value]
        return self.stack[sel]

    # ------------------------------------------------------------------ loading
    def _browse(self):
        d = os.path.expanduser(self.w_br_dir.value.strip() or ".")
        if not self._allowed_path(d):
            self.w_br_list.options = [("(not available on this server)", None)]
            return
        if not os.path.isdir(d):
            self.w_br_list.options = [("(folder not found)", None)]
            return
        try:
            names = sorted(os.listdir(d), key=str.lower)
        except OSError as e:
            self.w_br_list.options = [(f"(cannot list: {e.strerror})", None)]
            return
        dirs = [n for n in names if not n.startswith(".") and os.path.isdir(os.path.join(d, n))]
        files = [n for n in names if n.lower().endswith(self._EXT) and os.path.isfile(os.path.join(d, n))]
        opts = [("📁 ..", ("dir", os.path.dirname(os.path.abspath(d))))]
        opts += [(f"📁 {n}", ("dir", os.path.join(d, n))) for n in dirs]
        for n in files:
            sz = os.path.getsize(os.path.join(d, n))
            opts.append((f"📄 {n}   ({sz / 2**20:.1f} MB)", ("file", os.path.join(d, n))))
        self._browsing = True
        self.w_br_list.options = opts
        self.w_br_list.value = None
        self._browsing = False

    def _on_browse_select(self, change):
        v = change["new"]
        if not v or getattr(self, "_browsing", False):
            return
        kind, path = v
        if kind == "dir":
            self.w_br_dir.value = os.path.abspath(path)
        else:
            self.w_path.value = path

    def _out_dir(self):
        """Folder for all written files (private to the session in server mode)."""
        d = os.path.join(self.work_dir, "output")
        os.makedirs(d, exist_ok=True)
        return d

    def _allowed_path(self, path):
        """In server mode only files below work_dir and data_dir may be read."""
        if not self.server_mode:
            return True
        rp = os.path.realpath(os.path.expanduser(str(path)))
        roots = [self.work_dir] + ([self.data_dir] if self.data_dir else [])
        return any(rp == os.path.realpath(r) or rp.startswith(os.path.realpath(r) + os.sep) for r in roots)

    def _find_local(self, name, size):
        """Path of a local file with this name and size (upload from the same computer), else None."""
        if self.server_mode:
            return None
        dirs = [self.w_br_dir.value, self.data_dir, os.getcwd(),
                os.path.dirname(os.path.expanduser(self.w_path.value.strip())) if self.w_path.value.strip() else None,
                "~/Downloads", "~/Desktop", "~/Documents", "~"]
        seen = set()
        for d in dirs:
            if not d:
                continue
            d = os.path.abspath(os.path.expanduser(d))
            if d in seen:
                continue
            seen.add(d)
            f = os.path.join(d, name)
            try:
                if os.path.isfile(f) and os.path.getsize(f) == size:
                    return f
            except OSError:
                pass
        return None

    def _on_upload(self, change):
        files = change["new"]
        if not files:
            return
        try:
            if len(files) == 1:
                f0 = files[0]
                local = self._find_local(f0["name"], len(f0["content"]))
                if local is not None:              # same computer: load from the path (as 'load' would)
                    self.w_path.value = local
                    self.load(local)
                    self.w_load_status.value += (f"<br><small>(uploaded file found on this computer: {local}; "
                                                 f"read from there)</small>")
                    return
            items = [(bytes(f["content"]), f["name"]) for f in files]
            if len(items) == 1:
                stack, md, p = load_series(items[0][0], items[0][1])
            else:
                stack, md = load_many(items)
                p = None
            self._set_data(stack, md, p, ", ".join(n for _, n in items))
            if len(items) == 1:
                self._prefill_metadata(items[0][0], items[0][1])
        except Exception as e:  # noqa: BLE001
            self._status(self.w_load_status, f"could not read file: {e!r}", ok=False)

    def _on_stream_files(self, change):
        """Files chosen in the streaming upload (FileInput): copy them chunk by chunk into the session folder,
        with progress (file name, %, MB), then load them as a focal series."""
        info = change["new"] or []
        if not info:
            return
        threading.Thread(target=self._stream_job, args=(list(info),), daemon=True).start()

    def _project_changed(self, upload_id):
        """The chosen NOMAD project is also where the results go (tabs 5 and 6)."""
        if upload_id:
            self.w_nm_upload.value = upload_id
            self.w_nm_auto.value = True

    def _store_in_project(self, paths, w):
        """Put the uploaded files into the chosen NOMAD project (new one created if chosen); results then go there
        too.  Returns the result of ProjectPicker.store or None."""
        if not self.project.connected or not paths:
            return None
        t0 = time.time()

        def prog(sent, size):
            w.value = (f"<span style='color:#1565c0'>storing in the NOMAD project: {100 * sent / max(size, 1):.0f} % "
                       f"({sent / 1e6:.1f} of {size / 1e6:.1f} MB, {sent / max(time.time() - t0, 1e-3) / 1e6:.1f} "
                       f"MB/s)</span>")
        try:
            r = self.project.store(paths, progress=prog, tmp_dir=self.work_dir,
                                   default_name=os.path.splitext(os.path.basename(paths[0]))[0])
        except Exception as e:  # noqa: BLE001
            self._project_error = str(e)
            return None
        self._project_error = ""
        if r:
            self._project_changed(r["upload_id"])
        return r

    def _stored_html(self, r):
        err = getattr(self, "_project_error", "")
        if err:
            return f"<br><span style='color:crimson'>not stored in NOMAD: {html.escape(err)}</span>"
        if not r:
            return ""
        return (f"<br><span style='color:green'>stored in NOMAD project <a href='{r['gui_url']}' target='_blank' "
                f"rel='noopener'>{html.escape(r['upload_id'])}</a>/{html.escape(r['folder'])}; the results go "
                f"there too</span>")

    def _stream_job(self, info, chunk=4 << 20):
        files = self.w_upload.get_files(timeout=120)
        total = sum(int(f["size"]) for f in info) or 1
        done = 0
        t0 = time.time()
        d = os.path.join(self.work_dir, "uploads")
        os.makedirs(d, exist_ok=True)
        paths = []
        try:
            for k, f in enumerate(files):
                name = os.path.basename(f["name"]) or f"upload_{k}"
                size = int(f["size"])
                dst = os.path.join(d, name)
                got = 0
                with open(dst, "wb") as fo:
                    while got < size:
                        b = f["file_obj"].read(min(chunk, size - got))
                        if not b:
                            break
                        fo.write(b)
                        got += len(b)
                        done += len(b)
                        rate = done / max(time.time() - t0, 1e-3) / 1e6
                        self.w_upload_progress.value = (
                            f"<span style='color:#1565c0'>⬆ uploading <b>{html.escape(name)}</b>"
                            f"{f' ({k + 1} of {len(files)})' if len(files) > 1 else ''}: "
                            f"{100 * got / max(size, 1):.0f} % ({got / 1e6:.1f} of {size / 1e6:.1f} MB, "
                            f"{rate:.1f} MB/s)</span>"
                            f"<progress value='{done}' max='{total}' style='width:300px;margin-left:8px'></progress>")
                if got != size:
                    raise IOError(f"transfer of {name} incomplete ({got} of {size} bytes)")
                paths.append(dst)
        except Exception as e:  # noqa: BLE001
            self.w_upload_progress.value = f"<span style='color:crimson'>upload failed: {html.escape(repr(e))}</span>"
            return
        names = ", ".join(os.path.basename(p) for p in paths)
        self.w_upload_progress.value = (f"<span style='color:green'>✔ uploaded {html.escape(names)} "
                                        f"({total / 1e6:.1f} MB in {time.time() - t0:.0f} s) — loading ...</span>")
        stored = self._store_in_project(paths, self.w_upload_progress)
        try:
            if len(paths) == 1:
                self.load(paths[0])
                self.upload_files = paths
            else:
                items = [(open(p, "rb").read(), os.path.basename(p)) for p in paths]
                stack, md = load_many(items)
                self._set_data(stack, md, None, names, source_path=paths[0])
                self.upload_files = paths
            if stored is not None:
                self.nomad_ctx = {"url": stored["url"], "upload_id": stored["upload_id"], "path": stored["folder"],
                                  "files": stored["paths"]}
            self.w_upload_progress.value = (f"<span style='color:green'>✔ uploaded and loaded {html.escape(names)} "
                                            f"({total / 1e6:.1f} MB)</span>" + self._stored_html(stored))
        except Exception as e:  # noqa: BLE001
            self.w_upload_progress.value = (f"<span style='color:crimson'>uploaded {html.escape(names)}, but it "
                                            f"could not be read: {html.escape(repr(e))}</span>")

    def _on_demo(self, change):
        if change["new"]:
            stack, md, p = load_series(change["new"])
            self._set_data(stack, md, p, os.path.basename(change["new"]), source_path=change["new"])
            self._prefill_metadata(change["new"], change["new"])

    _EXT = (".nxs", ".h5", ".hdf5", ".tif", ".tiff", ".dm3", ".dm4", ".img", ".emd", ".mrc", ".st", ".npy", ".npz",
            ".ser")

    def _data_files(self):
        opts = [("—", None)]
        if self.data_dir and os.path.isdir(self.data_dir):
            for f in sorted(os.listdir(self.data_dir)):
                if f.lower().endswith(self._EXT) and not f.endswith("_noisefree.npy"):
                    opts.append((f, os.path.join(self.data_dir, f)))
        return opts

    def _load_path(self, path):
        if not path:
            return
        path = os.path.expanduser(path)
        if not self._allowed_path(path):
            self._status(self.w_load_status, "reading files of the server is not possible here", ok=False)
            return
        if not os.path.isfile(path):
            self._status(self.w_load_status, f"file not found: {path}", ok=False)
            return
        try:
            self.load(path)
        except Exception as e:  # noqa: BLE001
            self._status(self.w_load_status, f"could not read {path}: {e!r}", ok=False)

    def _load_folder(self, folder):
        from .io_formats import load_folder
        folder = os.path.expanduser(str(folder).strip())
        if not self._allowed_path(folder):
            self._status(self.w_load_status, "reading folders of the server is not possible here", ok=False)
            return
        try:
            stack, md = load_folder(folder)
        except Exception as e:  # noqa: BLE001
            self._status(self.w_load_status, f"could not load the folder {folder}: {e!r}", ok=False)
            return
        name = os.path.basename(os.path.normpath(folder)) or folder
        self._set_data(stack, md, None, f"{name} ({len(stack)} frames)",
                       source_path=os.path.join(folder, name))

    def load(self, path):
        """Programmatic loading (also used for testing)."""
        stack, md, p = load_series(path)
        self._set_data(stack, md, p, os.path.basename(path), source_path=path)
        self._prefill_metadata(path, path)
        return self

    def _set_data(self, stack, md, p, name, source_path=None):
        prev = getattr(self, "_defoci0", None)
        self.raw = np.asarray(stack, float)
        self.meta = md
        self.fits = {}
        self.align_info = None
        self.init_rec = None
        self.source_name = name
        self.source_path = source_path
        self.upload_files = []          # files of the series uploaded from the user's computer (server copies)
        kept = []
        ser = md.get("defocus_series")
        if ser:
            self.w_series_start.value = float(ser.get("offset", 0.0))
            self.w_series_step.value = float(ser.get("step", 1.0))
            self.w_series_m.value = float(ser.get("m", 1.0) or 1.0)
            n0 = ser.get("n0")
            self.w_series_n0.value = -1 if n0 is None or int(n0) == (len(stack) - 1) // 2 else int(n0)
        if p is not None:
            self.w_kv.value = p.voltage_V / 1e3
            self.w_pix.value = p.pixel_size_nm
            self.w_c3.value = p.aberrations.get("c_3_0", 0.0) / 1e3
            self.w_c5.value = p.aberrations.get("c_5_0", 0.0) / 1e6
            a, ang = np.hypot(p.aberrations.get("c_1_2_a", 0), p.aberrations.get("c_1_2_b", 0)), \
                0.5 * np.arctan2(p.aberrations.get("c_1_2_b", 0), p.aberrations.get("c_1_2_a", 0))
            self.w_a1.value, self.w_a1ang.value = round(float(a), 4), round(float(np.rad2deg(ang)), 3)
            self.w_b2.value = p.aberrations.get("c_2_1_a", 0.0)
            self.w_b2b.value = p.aberrations.get("c_2_1_b", 0.0)
            self.w_fs.value = p.focal_spread_nm
            self.w_alpha.value = p.convergence_rad * 1e3
            self.w_ap.value = (p.aperture_rad or 0) * 1e3
            self.w_mtf.value = "sinc" if p.mtf else "none"
            defoci, aa, ab = p.defoci_nm, p.astig_a_nm, p.astig_b_nm
        else:
            if "voltage_V" in md:
                self.w_kv.value = md["voltage_V"] / 1e3
            else:
                kept.append("voltage")
            if "pixel_size_nm" in md:
                self.w_pix.value = md["pixel_size_nm"]
            else:
                kept.append("pixel size")
                if getattr(self, "_bin_used", 1) != 1:
                    self.w_pix.value = self.w_pix.value / self._bin_used
            if "c_3_0" in md:
                self.w_c3.value = md["c_3_0"] / 1e3
            else:
                kept.append("Cs")
            n = len(self.raw)
            df = list(md.get("defoci_nm", []))
            defoci = np.array((df + [np.nan] * n)[:n], float)
            aa = ab = None
            if np.all(np.isnan(defoci)):
                if prev is not None and len(prev[0]) == n:
                    defoci, aa, ab = prev
                    kept.append("defoci and astigmatism")
                else:
                    defoci = self.series_values(n)
            if not kept:
                kept = []
            else:
                kept += ["focal spread, convergence, aperture"]
        self._defoci0 = (np.asarray(defoci, float), aa, ab)
        self._bin_used = 1                      # metadata pixel size refers to the raw data
        if hasattr(self, "rows"):
            del self.rows
        self._thon_cache = None
        self._suspend_thon = True
        self._prepare()
        self.w_idx.value = len(self.stack) // 2         # central frame of the series
        self._suspend_thon = False
        self._show_thon()
        with self.w_meta:
            self.w_meta.clear_output()
            show = {k: v for k, v in md.items() if k != "raw"}
            print(json.dumps(show, indent=1, default=str))
            if "raw" in md:
                print("\n--- raw metadata (truncated) ---")
                print(json.dumps(md["raw"], indent=1, default=str)[:20000])
        found = [k for k in ("pixel_size_nm", "voltage_V", "defoci_nm", "c_3_0") if k in md]
        if p is not None:
            found = ["all imaging parameters"]
        msg = (f"loaded <b>{name}</b>: {self.raw.shape[0]} images of {self.raw.shape[1]}×{self.raw.shape[2]} px; "
               f"metadata found: {', '.join(found) or 'none'} — please check tab 2.")
        if kept and prev is not None:
            msg += (f"<br><span style='color:#b26a00'>not in the file, kept from the previously loaded series: "
                    f"{', '.join(kept)}</span>")
        elif md.get("uncalibrated"):
            msg += "<br><span style='color:#b26a00'>the file is not calibrated: enter the pixel size in tab 2</span>"
        self._status(self.w_load_status, msg)
        self._prefill_from_hints(md.get("nxem_hints") or {})
        is_nx = str(name).lower().endswith((".nxs", ".h5", ".hdf5"))
        self.w_conv_box.layout.display = "none" if is_nx else "flex"
        self.w_conv_status.value = ""
        if not is_nx:
            base = os.path.splitext(source_path or name)[0]
            if source_path is None:
                base = os.path.join(self._out_dir(), os.path.basename(base))
            self.w_conv_name.value = base + ".nxs"
        else:
            self.tabs.selected_index = 1
        p0 = self.params(all_images=True)
        self.workflow_log = WorkflowLog()
        self.workflow_log.add("load", dict(
            source=str(name), file=os.path.basename(source_path) if source_path else str(name),
            n_images=int(self.raw.shape[0]), shape=f"{self.raw.shape[1]}×{self.raw.shape[2]}",
            pixel_size_nm=round(float(self.w_pix.value), 5), defocus_min_nm=round(float(np.min(p0.defoci_nm)), 2),
            defocus_max_nm=round(float(np.max(p0.defoci_nm)), 2)))

    def log_step(self, key, params=None, summary="", **kw):
        """Record a processing step (workflow of the session: ELN, NOMAD workflow, NXem description)."""
        try:
            return self.workflow_log.add(key, params, summary, **kw)
        except Exception:  # noqa: BLE001  -- logging must never break the app
            return None

    def _prepare(self):
        if self.raw is None:
            return
        s = _bin(self.raw, self.w_bin.value)
        c = self.w_crop.value
        if c and c < min(s.shape[1:]):
            y0 = (s.shape[1] - c) // 2
            x0 = (s.shape[2] - c) // 2
            s = s[:, y0:y0 + c, x0:x0 + c]
        factor = self.w_bin.value / getattr(self, "_bin_used", 1)
        if factor != 1:
            self.w_pix.value = self.w_pix.value * factor   # pixel size follows the binning
        self.stack = s
        self._bin_used = self.w_bin.value
        d, aa, ab = self._defoci0
        if hasattr(self, "rows") and len(self.rows) == len(d):
            d = np.array([r[1].value for r in self.rows])
            aa = np.array([r[2].value for r in self.rows])
            ab = np.array([r[3].value for r in self.rows])
        self._make_table(d, aa, ab)
        self.w_idx.max = len(s) - 1
        self._thon_cache = None
        self._update_q_ranges()
        self._status(self.w_load_status, f"working stack: {s.shape[0]} × {s.shape[1]} × {s.shape[2]} px "
                                         f"(binning {self.w_bin.value}, pixel size {self.w_pix.value:.4g} nm)")
        if self.w_bin.value != 1 or self.w_crop.value:
            self.log_step("prepare", dict(binning=int(self.w_bin.value), crop_px=int(self.w_crop.value or 0)),
                          replace=True)

    # ------------------------------------------------------------------ Thon rings
    def _thon_params(self, i):
        p = self.params(all_images=True)
        return p, p.aberrations_for(i)

    def _update_q_ranges(self):
        """Absolute q sliders (1/nm) up to the Nyquist frequency of the working stack."""
        pix = self.w_pix.value
        if not pix or pix <= 0:
            return
        qn = 0.5 / pix
        rel = self._q_rel
        self._suspend_thon = self._suspend_thon or False
        sus, self._suspend_thon = self._suspend_thon, True
        for w, r in ((self.w_qmin, rel[0]), (self.w_qmax, rel[1])):
            w.max = max(w.max, qn)
            w.step = float(f"{qn / 400:.2g}")
            w.value = r * qn
            w.max = qn
        self._suspend_thon = sus

    def _on_q(self):
        if self._suspend_thon or not self.w_pix.value:
            return
        qn = 0.5 / self.w_pix.value
        self._q_rel = (self.w_qmin.value / qn, self.w_qmax.value / qn)
        self._show_thon(quick=True)

    def _thon_figure(self, i, img, comp, qn, title, fit=None):
        """Figure: image | data/model spectrum (zoomed to q_max/0.9) | radial profiles."""
        from matplotlib.figure import Figure
        fig = Figure(figsize=(11, 3.6))
        ax = fig.subplots(1, 3)
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title(title, fontsize=9)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        qmin, qmax = self.w_qmin.value, self.w_qmax.value
        qv = min(qmax / 0.9, qn)                     # outer ring covers 90 % of the field of view
        ny, nx = comp.shape                      # fftshifted, zero frequency at (ny//2, nx//2)
        dqy, dqx = 2 * qn / ny, 2 * qn / nx
        hy, hx = max(int(np.ceil(qv / dqy)), 4), max(int(np.ceil(qv / dqx)), 4)
        cy, cx = ny // 2, nx // 2
        y0, y1, x0, x1 = max(cy - hy, 0), min(cy + hy + 1, ny), max(cx - hx, 0), min(cx + hx + 1, nx)
        crop = comp[y0:y1, x0:x1]
        ext = ((x0 - cx - 0.5) * dqx, (x1 - cx - 0.5) * dqx, (y0 - cy - 0.5) * dqy, (y1 - cy - 0.5) * dqy)
        dat = crop[:, :max(cx - x0, 1)]          # data half: contrast from the fitted annulus
        yy = (np.arange(y0, y1) - cy)[:, None] * dqy
        xx = (np.arange(x0, x0 + dat.shape[1]) - cx)[None, :] * dqx
        qq = np.hypot(yy, xx)
        ann = dat[(qq >= qmin) & (qq <= qmax)]
        lo, hi = np.percentile(ann if ann.size > 50 else dat, [1, 99.5])
        crop = crop.copy()
        crop[:, :cx - x0] = np.clip((dat - lo) / (hi - lo + 1e-12), 0, 1)
        ax[1].imshow(crop, cmap="gray", extent=ext, vmin=0, vmax=1, origin="lower", aspect="equal")
        for q in (qmin, qmax):
            ax[1].add_patch(_circle(q))
        ax[1].set_xlim(-qv, qv)
        ax[1].set_ylim(-qv, qv)
        ax[1].set_title("data | model", fontsize=9)
        ax[1].set_xlabel("q (1/nm)")
        if fit is not None:
            qr, Br, Mr = fit["profile"]
            ax[2].plot(qr, (Br - Br.mean()) / (Br.std() + 1e-12), lw=0.8, label="data (rot. avg.)")
            ax[2].plot(qr, (Mr - Mr.mean()) / (Mr.std() + 1e-12) - 3, lw=0.8, label="model (−3)")
            ax[2].axvspan(*fit["q_range"], color="orange", alpha=0.1)
            ax[2].set_xlim(0, qv)
            ax[2].legend(fontsize=7)
            ax[2].set_title(f"fit: score {fit['score']:.3f}", fontsize=9)
            ax[2].set_xlabel("q (1/nm)")
        else:
            ax[2].axis("off")
            ax[2].text(0.1, 0.5, "not fitted yet", transform=ax[2].transAxes)
        fig.tight_layout()
        return fig

    def _show_thon(self, quick=False):
        if self.stack is None or self._suspend_thon or not hasattr(self, "rows"):
            return
        i = min(self.w_idx.value, len(self.stack) - 1)
        p, ab = self._thon_params(i)
        if i in self.fits:
            ab = dict(ab, c_1_0=self.fits[i]["c_1_0"], c_1_2_a=self.fits[i]["c_1_2_a"],
                      c_1_2_b=self.fits[i]["c_1_2_b"])
        qn = 0.5 / p.pixel_size_nm
        tile = None if self.w_qmax.value < 0.3 * qn else "auto"     # zoomed in: full-image spectrum (finer dq)
        key = (i, p.pixel_size_nm, p.voltage_V, self.w_amp.value, id(self.stack), tile,
               tuple(sorted((k, round(float(v), 6)) for k, v in ab.items())))
        if self._thon_cache is None or self._thon_cache[0] != key:
            comp = ctffit.thon_composite(self.stack[i], p.pixel_size_nm, ab, p.voltage_V, self.w_amp.value,
                                         tile=tile)
            self._thon_cache = (key, comp)
        comp = self._thon_cache[1]
        fig = self._thon_figure(i, self.stack[i], comp, 0.5 / p.pixel_size_nm,
                                f"image {i}, C1 = {ab['c_1_0']:.1f} nm", self.fits.get(i))
        self._png_out(self.w_thon_out, fig)

    def _fit(self, single=True, block=True):
        if self.stack is None:
            return
        self._start_job(lambda: self._fit_job(single), [self.w_fit1, self.w_fitall, self.w_fitglob, self.w_applyfit],
                        self.w_thon_stop, self.w_thon_status, block)

    def _fit_job(self, single):
        p = self.params(all_images=True)
        idx = [min(self.w_idx.value, len(self.stack) - 1)] if single else list(range(len(self.stack)))
        qn = 0.5 / p.pixel_size_nm
        c3 = p.aberrations.get("c_3_0", 0.0)
        extra = {k: v for k, v in p.aberrations.items() if k not in ("c_1_0", "c_1_2_a", "c_1_2_b", "c_3_0")}
        c3_vals = [c3]
        if self.w_fitc3.value and not single:
            c3_vals = list(np.linspace(0.5, 1.5, 5) * c3) if c3 else list(np.linspace(-20e3, 20e3, 9))
        total = len(c3_vals) * len(idx)
        self.w_thon_prog.max = total
        self.w_thon_prog.value = 0
        t0 = time.time()
        self._text_out(self.w_fit_out, "fitting ...")
        best = None
        done = 0
        stopped = False
        last = [0.0]
        for c3v in c3_vals:
            res = {}
            for i in idx:
                if self._stop.is_set():
                    stopped = True
                    break
                d = p.defoci_nm[i]
                w = self.w_win.value
                ab = dict(p.aberrations_for(i), c_3_0=c3v)
                # show the image and spectrum being fitted (model with the start values)
                comp = ctffit.thon_composite(self.stack[i], p.pixel_size_nm, ab, p.voltage_V, self.w_amp.value)
                self._png_out(self.w_thon_out, self._thon_figure(
                    i, self.stack[i], comp, qn, f"fitting image {i} ({done + 1}/{total}), start C1 = {d:.1f} nm"))
                head = f"image {i} ({done + 1}/{total})" + (f", C3 = {c3v / 1e3:.2f} µm" if len(c3_vals) > 1 else "")

                def prog(stage, n_eval, score, head=head, done=done):
                    now = time.time()
                    if now - last[0] > 0.25:
                        last[0] = now
                        self.w_thon_status.value = (f"{head}: {stage}, {n_eval} evaluations, best score "
                                                    f"{score:.3f} — {self._eta(t0, done / total)}")
                    if self._stop.is_set():
                        raise StopRequested()

                r = ctffit.fit_thon(self.stack[i], p.pixel_size_nm, p.voltage_V, (d - w, d + w), c3v,
                                    q_min=self.w_qmin.value, q_max=self.w_qmax.value,
                                    fit_astig=self.w_fitastig.value, amp_contrast=self.w_amp.value,
                                    extra_aberrations=extra, progress=prog)
                res[i] = r
                done += 1
                self.w_thon_prog.value = done
                if r.get("stopped"):
                    stopped = True
                    break
            if res:
                score = np.mean([r["score"] for r in res.values()])
                if best is None or score > best[0]:
                    best = (score, c3v, res)
            if stopped or self._stop.is_set():
                stopped = True
                break
        if best is None:
            self._status(self.w_thon_status, "stopped before any image was fitted", ok=False)
            self._show_thon()
            return
        self.fits.update(best[2])
        self._fit_c3 = best[1]
        self._global_fit = None
        lines = [f"{'#':>3} {'C1 nominal':>11} {'C1 fit':>9} {'C12a':>7} {'C12b':>7} {'|A1|':>6} {'score':>6}"]
        for i, r in sorted(self.fits.items()):
            lines.append(f"{i:3d} {p.defoci_nm[i]:11.2f} {r['c_1_0']:9.2f} {r['c_1_2_a']:7.2f} {r['c_1_2_b']:7.2f} "
                         f"{np.hypot(r['c_1_2_a'], r['c_1_2_b']):6.2f} {r['score']:6.3f}")
        if len(c3_vals) > 1:
            lines.append(f"best C3 = {best[1] / 1e3:.2f} µm (mean score {best[0]:.3f})")
        lines.append("Note: the per-image C12a/b values include the global A1 of tab 2 when applied (global is "
                     "subtracted).")
        self._text_out(self.w_fit_out, "\n".join(lines) + "\n")
        self._status(self.w_thon_status, ("stopped: " if stopped else "done: ") +
                     f"{len(best[2])} image(s) fitted in {time.time() - t0:.0f} s"
                     + (" (last one: best values found so far)" if stopped else ""), ok=not stopped)
        self.log_step("thon", dict(mode="per image", images=len(best[2]), fit_astigmatism=bool(self.w_fitastig.value),
                                   fit_c3=bool(self.w_fitc3.value)),
                      f"{len(best[2])} image(s) fitted" + (" (stopped)" if stopped else "") + ".")
        self._show_thon()

    def _fit_global(self, block=True):
        if self.stack is None:
            return
        self._start_job(self._fit_global_job, [self.w_fit1, self.w_fitall, self.w_fitglob, self.w_applyfit],
                        self.w_thon_stop, self.w_thon_status, block)

    def _fit_global_job(self):
        p = self.params(all_images=True)
        model = self.w_series_model.value
        kind = "power" if model.startswith("power") else "scale"
        n = len(self.stack)
        maxiter = 1500
        self.w_thon_prog.max = maxiter
        self.w_thon_prog.value = 0
        t0 = time.time()
        last = [0.0]
        i0 = n // 2
        comp = ctffit.thon_composite(self.stack[i0], p.pixel_size_nm, p.aberrations_for(i0), p.voltage_V,
                                     self.w_amp.value)
        self._png_out(self.w_thon_out, self._thon_figure(
            i0, self.stack[i0], comp, 0.5 / p.pixel_size_nm,
            f"series fit of all {n} images (central image {i0} shown)"))
        self._text_out(self.w_fit_out, "series fit ...")

        def prog(stage, n_eval, maxit, best):
            now = time.time()
            if now - last[0] > 0.25:
                last[0] = now
                frac = min(n_eval / maxit, 1.0)
                self.w_thon_prog.value = min(n_eval, maxit)
                self.w_thon_status.value = (f"series fit ({n} images simultaneously): {stage}, {n_eval} score "
                                            f"evaluations (≤ {maxit} + offset search), best mean score "
                                            f"{best:.3f} — {self._eta(t0, frac) if n_eval else 'preparing'}"
                                            f" (upper bound; usually converges earlier)")
            if self._stop.is_set():
                raise StopRequested()

        n0 = int(self.w_series_n0.value)
        r = ctffit.fit_series_global(self.stack, p, window_nm=self.w_win.value, fit_astig=self.w_fitastig.value,
                                     amp_contrast=self.w_amp.value, q_min=self.w_qmin.value,
                                     q_max=self.w_qmax.value, model=kind, n0=None if n0 < 0 or n0 >= n else n0,
                                     m=self.w_series_m.value, fit_m=(model == "power_fit"), progress=prog,
                                     maxiter=maxiter)
        self._global_fit = r
        self.w_thon_prog.value = maxiter
        if r["model"] == "power":
            head = (f"FRWR Eq. 1: Δf_offset = {r['offset']:.2f} nm (frame n0 = {r['n0']}), step = {r['step']:.3f} nm, "
                    f"m = {r['m']:.3f}")
        else:
            head = f"offset = {r['offset']:.2f} nm, scale = {r['scale']:.4f}"
        lines = [head + f", common A1: C12a = {r['c_1_2_a']:.2f} nm, C12b = {r['c_1_2_b']:.2f} nm, "
                        f"mean score = {r['score']:.3f}"]
        for i, d in enumerate(r["defoci"]):
            lines.append(f"{i:3d}  C1 = {d:9.2f} nm   score = {r['per_image_scores'].get(i, float('nan')):.3f}")
        lines.append("press 'apply fits to parameters' to accept")
        self._text_out(self.w_fit_out, "\n".join(lines) + "\n")
        self._status(self.w_thon_status, ("stopped — best parameters so far kept" if r["stopped"] else "series fit done")
                     + f" ({time.time() - t0:.0f} s)", ok=not r["stopped"])
        self.log_step("thon", dict(mode="whole series", model=r.get("model", "linear"),
                                   fit_astigmatism=bool(self.w_fitastig.value)),
                      head.replace("\n", " ") + f", mean score {r['score']:.3f}.")
        self.fits = {}

    def _apply_fits(self):
        if getattr(self, "_global_fit", None) is not None or self.fits:
            self.log_step("thon_apply")
        if getattr(self, "_global_fit", None) is not None and not self.fits:
            r = self._global_fit
            for (u, d, a, b), c1 in zip(self.rows, r["defoci"]):
                d.value = round(float(c1), 3)
            if r.get("model") == "power":
                self.w_series_start.value = round(r["offset"], 3)
                self.w_series_step.value = round(r["step"], 4)
                self.w_series_m.value = round(r["m"], 4)
                self.w_series_n0.value = -1 if r["n0"] == (len(self.rows) - 1) // 2 else int(r["n0"])
            if self.w_fitastig.value:
                self.w_a1.value = float(np.hypot(r["c_1_2_a"], r["c_1_2_b"]))
                self.w_a1ang.value = float(np.rad2deg(0.5 * np.arctan2(r["c_1_2_b"], r["c_1_2_a"])))
            self._global_fit = None
            self._show_thon()
            return
        p = self.params(all_images=True)
        ga = p.aberrations.get("c_1_2_a", 0.0)
        gb = p.aberrations.get("c_1_2_b", 0.0)
        for i, r in self.fits.items():
            u, d, a, b = self.rows[i]
            d.value = round(r["c_1_0"], 3)
            if self.w_fitastig.value:
                a.value = round(r["c_1_2_a"] - ga, 3)
                b.value = round(r["c_1_2_b"] - gb, 3)
        if getattr(self, "_fit_c3", None) is not None and self.w_fitc3.value:
            self.w_c3.value = self._fit_c3 / 1e3
        self.fits = {}
        self._show_thon()

    # ------------------------------------------------------------------ alignment
    def _align(self):
        if self.stack is None:
            return
        p = self.params(all_images=True)
        out, sh = align.register_stack(self.stack, p.defoci_nm, sigma_hp=self.w_hp.value or None,
                                       sequential=self.w_seq.value)
        self.stack = out
        self.log_step("drift", dict(high_pass_sigma=self.w_hp.value or None, sequential=bool(self.w_seq.value)),
                      f"largest shift {float(np.abs(sh).max()):.1f} px.")
        with self.w_align_out:
            self.w_align_out.clear_output()
            fig, ax = plt.subplots(1, 2, figsize=(9, 3))
            ax[0].plot(p.defoci_nm, sh[:, 1], "o-", label="x")
            ax[0].plot(p.defoci_nm, sh[:, 0], "s-", label="y")
            ax[0].set_xlabel("C1 (nm)")
            ax[0].set_ylabel("shift (px)")
            ax[0].legend()
            ax[1].imshow(self.stack.std(0), cmap="gray")
            ax[1].set_title("std over aligned series")
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------ reconstruction
    def _options(self):
        m = self.w_method.value
        o = {}
        if m in ("gs", "iwfr", "frwr", "frwr_gf", "gpfrwr", "frwr_gd", "gpfrwr_gd", "gpfrwr_lsq", "mal",
                 "ml_poisson"):
            o["n_iter"] = self.w_iter.value
            o["init"] = self._start_wave()
        if m in ("pam", "schiske"):
            o["reg"] = "auto" if self.w_reg.value.strip() == "auto" else float(self.w_reg.value)
        if m in ("frwr", "frwr_gf", "gpfrwr", "frwr_gd", "gpfrwr_gd", "gpfrwr_lsq"):
            o["background"] = self.w_bg.value
        if m in ("frwr", "frwr_gf", "gpfrwr"):
            o["backend"] = self.w_backend.value
            txt = "\n".join(l for l in self.w_frwr3_yaml.value.splitlines() if l.strip() and not l.strip().startswith("#"))
            cfg = {}
            if txt.strip():
                import yaml
                cfg = yaml.safe_load(txt) or {}
                cfg = cfg if isinstance(cfg, dict) else {}
            if self.w_allframes.value:
                cfg.setdefault("start with all planes", 1)
            if cfg:
                o["config"] = cfg
        if m == "gftie":
            o["n_iter"] = self.w_iter.value
        if m == "ml_poisson":
            o.update(n_defocus=3, n_tilt=2)
        return m, o

    def _start_wave(self):
        """Start wave of tab 5: method name or a complex array (tab 4b result / last reconstruction)."""
        v = self.w_init.value
        if v not in ("frwr_align", "last"):
            return v
        shape = self.stack.shape[1:]
        if v == "frwr_align":
            w = None if self.init_rec is None else getattr(self, "init_wave_start", None)
            if w is None:
                raise ValueError("no FRWR initial reconstruction yet: run the FRWR alignment of tab 4b first")
        else:
            if self.rec is None:
                raise ValueError("no reconstruction yet")
            w = self.rec.wave
        w = np.asarray(w, complex)
        if w.shape != shape:
            raise ValueError(f"the start wave ({w.shape[0]}×{w.shape[1]} px) does not match the working stack "
                             f"({shape[0]}×{shape[1]} px) — the stack was changed after the alignment")
        return w

    def _use_init_rec(self):
        self.w_init.value = "frwr_align"
        self.w_allframes.value = True
        self.tabs.selected_index = 4

    def _load_frwr3_conf(self):
        v = self.w_frwr3_upload.value
        if not v:
            return
        item = v[0] if isinstance(v, (list, tuple)) else list(v.values())[0]
        name = item.get("name", "") if isinstance(item, dict) else getattr(item, "name", "")
        content = item["content"] if isinstance(item, dict) else item.content
        text = bytes(content).decode(errors="ignore")
        if name.endswith(".conf") or "%" in text:
            from .frwr3_io import read_conf
            cfg = read_conf(text)
            lines = [f"{k}: {v[0] if len(v) == 1 else v}" for k, v in cfg.items()]
            text = "\n".join(lines)
        self.w_frwr3_yaml.value = text

    def _run(self, block=True):
        if self.stack is None:
            return
        self._start_job(self._run_job, [self.w_run], self.w_rec_stop, self.w_rec_status, block)

    def _run_job(self):
        try:
            m, o = self._options()
        except ValueError as e:
            self._status(self.w_rec_status, str(e), ok=False)
            return
        p = self.params()
        data = self.used_stack()
        n_it = o.get("n_iter", 1)
        self.w_prog.max = max(n_it, 1)
        self.w_prog.value = 0
        t0 = time.time()
        last = {"psi": None, "hist": [], "t": 0.0}

        def cb(it, psi, cost):
            last["hist"].append(float(cost))
            if psi is not None:
                last["psi"] = psi
            total = getattr(cb, "planned", None) or n_it
            k = min(int(it) + 1, max(int(total), 1))
            frac = k / max(total, 1)
            eng = getattr(cb, "engine", None)
            txt = None
            if eng is not None:                 # FRWR3: loop schedule with learnt loop times
                frac, txt = self._frwr_eta(cb.st, eng, getattr(cb, "label", ""), t0)
            self.w_prog.max = 1000
            self.w_prog.value = int(round(1000 * frac))
            now = time.time()
            if now - last["t"] > 0.2 or k == self.w_prog.max:
                last["t"] = now
                self.w_rec_status.value = (f"{'loop' if cb.planned else 'iteration'} {k}/{total}, "
                                           f"cost {cost:.4g} — {txt or self._eta(t0, frac)}")
            if self._stop.is_set():
                raise StopRequested()

        cb.planned = None
        cb.engine = None
        cb.st = {}
        if "n_iter" in o:
            o["callback"] = cb
        stopped = False
        try:
            rec = reconstruct(data, p, m, **o)
            stopped = bool(rec.options.get("stopped_after_loops"))
        except StopRequested:
            psi = last["psi"]
            if psi is None:
                self._status(self.w_rec_status, "stopped before the first iteration finished — no result", ok=False)
                return
            psi = np.asarray(psi)
            ny, nx = data.shape[1:]
            if psi.shape != (ny, nx):                          # padded reconstruction: central part
                y0, x0 = (psi.shape[0] - ny) // 2, (psi.shape[1] - nx) // 2
                psi = psi[y0:y0 + ny, x0:x0 + nx]
            from .reconstruct import Reconstruction
            opts = {k: v for k, v in o.items() if k != "callback"}
            opts["stopped_after_iterations"] = len(last["hist"])
            rec = Reconstruction(psi, m, opts, list(last["hist"]), time.time() - t0)
            stopped = True
        rec.options.pop("callback", None)
        self.rec = rec
        self.recs.append((m, rec, p))
        hist = [h for h in (rec.history or []) if np.isscalar(h)]
        self.log_step("reconstruct", {**{k: v for k, v in rec.options.items() if np.isscalar(v) or v is None},
                                      "method": m},
                      f"{m}: " + (f"{len(rec.history)} iterations in " if rec.history else "")
                      + f"{rec.runtime_s:.1f} s"
                      + (f", final R = {hist[-1]:.4g}" if hist else "") + (" (stopped)" if stopped else "") + ".")
        if not stopped:
            self.w_prog.value = self.w_prog.max
        self._status(self.w_rec_status,
                     (f"{m} stopped after {len(rec.history)} {'loops' if cb.planned else 'iterations'} "
                      f"({time.time() - t0:.1f} s) — the current wave is kept as the result" if stopped
                      else f"{m} finished in {time.time() - t0:.1f} s"), ok=True)
        self._show_rec()

    def _show_rec(self):
        if self.rec is None:
            return
        w = self.rec.wave
        ph = np.angle(w)
        if self.w_unwrap.value:
            from skimage.restoration import unwrap_phase
            ph = unwrap_phase(ph)
        from matplotlib.figure import Figure
        fig = Figure(figsize=(13, 3.4))
        ax = fig.subplots(1, 4)
        im = ax[0].imshow(np.abs(w), cmap="gray")
        ax[0].set_title("amplitude")
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        im = ax[1].imshow(ph, cmap="gray" if self.w_unwrap.value else "twilight")
        ax[1].set_title("phase (rad)")
        fig.colorbar(im, ax=ax[1], fraction=0.046)
        F = np.fft.fftshift(np.log1p(np.abs(np.fft.fft2(w - w.mean()))))
        ax[2].imshow(F, cmap="magma")
        ax[2].set_title("|FFT(wave)|")
        if self.rec.history:
            h = np.asarray(self.rec.history, float)
            if h.min() > 0:                       # R-values / costs: plotted directly
                ax[3].semilogy(h)
                ax[3].set_title("R-value" if self.rec.method.startswith(("frwr", "gpfrwr")) else "cost")
            else:
                ax[3].semilogy(h - h.min() + 1e-12)
                ax[3].set_title("cost − min")
            ax[3].set_xlabel("iteration / loop")
        else:
            ax[3].axis("off")
        for a in ax[:3]:
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()
        self._png_out(self.w_rec_out, fig)

    def _strain(self):
        if self.rec is None:
            return
        from .strain import strain_1d
        e, perp = strain_1d(self.rec.wave, (self.w_gy.value, self.w_gx.value), self.w_pix.value,
                            sigma_px=self.w_ssig.value)
        e = e - np.median(e)
        src = self.recs[-1][0] if self.recs else "current reconstruction"
        self.maps.append(dict(kind="strain", array=e, g=(self.w_gx.value, self.w_gy.value),
                              sigma=self.w_ssig.value, source=src, params=self.params()))
        self.log_step("strain", dict(g=[self.w_gx.value, self.w_gy.value], sigma_px=self.w_ssig.value, source=src))
        with self.w_strain_out:
            self.w_strain_out.clear_output(wait=True)
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
            vm = np.percentile(np.abs(100 * e), 99)
            im = ax[0].imshow(100 * e, cmap="RdBu_r", vmin=-vm, vmax=vm)
            ax[0].set_title("ε_gg (%) relative to median")
            plt.colorbar(im, ax=ax[0], fraction=0.046)
            ax[1].plot(np.arange(e.shape[1]) * self.w_pix.value, 100 * e[e.shape[0] // 4: 3 * e.shape[0] // 4].mean(0))
            ax[1].set_xlabel("x (nm)")
            ax[1].set_ylabel("ε_gg (%) (central rows)")
            plt.tight_layout()
            plt.show()

    def _halfset(self):
        if self.stack is None:
            return
        data = self.used_stack()
        if not np.allclose(data[:, ::7, ::7], np.round(data[:, ::7, ::7])):
            self._status(self.w_rec_status, "half-set FRC needs integer electron counts "
                                            "(gain-normalized raw data)", ok=False)
            return
        m, o = self._options()
        p = self.params()

        def job():
            self._status(self.w_rec_status, "half-set FRC: reconstructing two half data sets ...")
            o.pop("callback", None)
            h1, h2 = binomial_split(data, rng=0)
            r1 = reconstruct(h1, p, m, **o)
            r2 = reconstruct(h2, p, m, **o)
            q, f, n = frc(align_global_phase(r1.wave, r2.wave), r2.wave, p.pixel_size_nm,
                          window=0.15)
            self.curves.append(dict(kind="half_set_frc", method=m, q=q, frc=f, threshold=half_bit_threshold(n),
                                    params=p))
            self.log_step("halfset", dict(method=m))
            from matplotlib.figure import Figure
            fig = Figure(figsize=(5, 3))
            ax = fig.subplots()
            ax.plot(q[1:], f[1:], label="half-set FRC")
            ax.plot(q[1:], half_bit_threshold(n)[1:], "k:", label="half-bit")
            ax.set_xscale("log")
            ax.set_xlabel("q (1/nm)")
            ax.set_ylim(-0.2, 1.02)
            ax.legend()
            fig.tight_layout()
            self._png_out(self.w_rec_out, fig, append=True)
            self._status(self.w_rec_status, "half-set FRC done")

        job()

    # ------------------------------------------------------------------ export
    # ------------------------------------------------------------------ NXem metadata form
    def _build_metadata_form(self):
        self.w_md = {}
        combos = {("sample", "physical_form"): ["bulk", "foil", "thin_film", "powder"],
                  ("sample", "identifier_type"): ["local", "DOI", "IGSN", "Hdl", "ARK", "PURL", "URL", "URN"],
                  ("instrument", "emitter_type"): ["cold_field_emission", "schottky", "thermionic_LaB6",
                                                   "thermionic_W"],
                  ("instrument", "vendor"): ["JEOL", "Thermo Fisher Scientific", "Hitachi", "Zeiss", "Nion"],
                  ("user", "role"): ["principal_investigator", "operator", "analyst"]}
        sections, titles = [], []
        for sec, fields in nxem.METADATA_FIELDS.items():
            rows = []
            for key, (label, default, req, helptext) in fields.items():
                desc = f"{label}{' *' if req else ''}"
                if isinstance(default, bool):
                    w = W.Checkbox(value=default, description=desc, indent=False, layout=W.Layout(width="auto"))
                elif (sec, key) in combos:
                    w = W.Combobox(value=str(default), options=combos[(sec, key)], description=desc,
                                   ensure_option=False, placeholder=helptext, style={"description_width": "230px"},
                                   layout=W.Layout(width="620px"))
                else:
                    w = W.Text(value=str(default), description=desc, placeholder=helptext,
                               style={"description_width": "230px"}, layout=W.Layout(width="620px"))
                w.tooltip = helptext
                w.observe(lambda ch: self._md_changed(), names="value")
                self.w_md[(sec, key)] = w
                rows.append(w)
            sections.append(W.VBox(rows))
            titles.append(sec)
        self.w_md_acc = W.Accordion(children=sections, titles=tuple(titles))
        self.w_md_status = W.HTML()
        self.w_md_upload = W.FileUpload(accept=".yaml,.yml,.json", multiple=False, description="load metadata")
        self.w_md_upload.observe(self._on_md_upload, names="value")
        self.w_md_save = W.Button(description="save metadata (YAML)", icon="download")
        self.w_md_save.on_click(lambda _: self._save_md())
        self.w_md_link = W.HTML()
        self._md_changed()
        return W.VBox([W.HTML("<b>NXem metadata</b> (reused for every file written in this session; can be "
                              "saved and reloaded as YAML)"), self.w_md_acc,
                       W.HBox([self.w_md_upload, self.w_md_save, self.w_md_link]), self.w_md_status])

    def metadata(self):
        m = nxem.default_metadata()
        for (sec, key), w in self.w_md.items():
            m[sec][key] = w.value
        return m

    def set_metadata(self, meta):
        for (sec, key), w in self.w_md.items():
            v = (meta or {}).get(sec, {}).get(key, None)
            if v is None or v == "":
                continue
            w.value = bool(v) if isinstance(w, W.Checkbox) else str(v)
        self._md_changed()

    def _md_changed(self):
        miss = nxem.missing_required(self.metadata())
        if miss:
            self.w_md_status.value = ("<span style='color:#b26a00'>required by NXem and still empty: "
                                      + ", ".join(f"{s}/{lab}" for s, _, lab in miss) + "</span>")
        else:
            self.w_md_status.value = "<span style='color:green'>all fields required by NXem are filled</span>"

    def _on_md_upload(self, change):
        v = change["new"]
        items = list(v.values()) if isinstance(v, dict) else list(v)
        if not items:
            return
        txt = bytes(items[0]["content"]).decode("utf-8")
        try:
            try:
                import yaml
                meta = yaml.safe_load(txt)
            except ImportError:
                meta = json.loads(txt)
            self.set_metadata(meta)
        except Exception as e:  # noqa: BLE001
            self.w_md_status.value = f"<span style='color:crimson'>could not read metadata: {e!r}</span>"

    def _save_md(self):
        meta = self.metadata()
        try:
            import yaml
            txt, ext = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), "yaml"
        except ImportError:
            txt, ext = json.dumps(meta, indent=2), "json"
        out_dir = self._out_dir()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"nxem_metadata.{ext}")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(txt)
        b64 = base64.b64encode(txt.encode()).decode()
        self.w_md_link.value = (f"<a download='nxem_metadata.{ext}' href='data:text/plain;base64,{b64}'>"
                                f"⬇ nxem_metadata.{ext}</a> (also in <code>{path}</code>)")

    _VENDORS = {"zeiss": "Carl Zeiss", "jeol": "JEOL", "fei": "FEI", "thermo": "Thermo Fisher Scientific",
                "hitachi": "Hitachi", "philips": "Philips", "nion": "Nion", "tescan": "Tescan"}
    _ELEMENTS = set("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se "
                    "Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb "
                    "Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu".split())

    def _prefill_from_hints(self, h):
        """Prefill empty NXem metadata fields from tags of the loaded file (user must verify)."""
        import re
        if not h:
            return

        def put(sec, key, val):
            w = self.w_md.get((sec, key))
            if w is not None and val not in (None, "") and not str(w.value).strip():
                w.value = str(val)

        mic = h.get("microscope", "")
        if mic:
            put("instrument", "name", mic)
            first, _, rest = mic.partition(" ")
            if first.lower() in self._VENDORS:
                put("instrument", "vendor", self._VENDORS[first.lower()])
                put("instrument", "model", rest or mic)
        spec = h.get("specimen", "")
        if spec:
            put("sample", "name", spec)
            els = [e for e in re.findall(r"[A-Z][a-z]?", spec) if e in self._ELEMENTS]
            if els and re.fullmatch(r"([A-Z][a-z]?\d*)+", spec.replace(" ", "")):
                put("sample", "atom_types", ", ".join(dict.fromkeys(els)))
        put("user", "name", h.get("operator", ""))
        t = h.get("acquisition_time", "")
        if t:
            m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})", t)
            if m:
                d, mo, y, H, M, S_ = m.groups()
                put("experiment", "start_time", f"{y}-{int(mo):02d}-{int(d):02d}T{int(H):02d}:{M}:{S_}")
            else:
                put("experiment", "start_time", t)
        put("instrument", "software", "DigitalMicrograph (FRWR acquisition plugin)" if mic else "")
        self._md_changed()

    def _convert(self):
        """Write the loaded (full-resolution) series + parameters + metadata as an NXem file."""
        if self.raw is None:
            return
        meta = self.metadata()
        miss = nxem.missing_required(meta)
        if miss:
            self.w_conv_status.value = ("<span style='color:crimson'>Please fill in first: "
                                        + ", ".join(f"{s}/{lab}" for s, _, lab in miss) + "</span>")
            return
        path = self.w_conv_name.value.strip()
        if not path.lower().endswith(".nxs"):
            path += ".nxs"
        if self.server_mode:
            path = os.path.join(self._out_dir(), os.path.basename(path))
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        p = self.params(all_images=True)
        p.pixel_size_nm = self.w_pix.value / getattr(self, "_bin_used", 1)      # raw (unbinned) pixel size
        extra = {k: v for k, v in self.meta.items() if k not in ("raw",)}
        extra["source_file"] = getattr(self, "source_name", "")
        dt = np.float32 if self.meta.get("source_dtype") in ("float32", "uint16", "int16", "uint8") else None
        nxem.write_focal_series(path, self.raw, p, description=f"focal series converted from {extra['source_file']}",
                                extra_metadata=extra, metadata=meta, dtype=dt)
        ok, probs, _ = nxem.validate(path)
        v = {True: "<span style='color:green'>✔ valid NXem (pynxtools)</span>",
             False: "<span style='color:crimson'>✘ not valid: " + "; ".join(probs[:5]) + "</span>",
             None: "<span style='color:#555'>validation skipped (pynxtools not installed)</span>"}[ok]
        self.w_conv_status.value = f"written <code>{path}</code> ({os.path.getsize(path) / 1e6:.1f} MB) {v}"
        self.w_demo.options = self._data_files()
        self._register_output(path)
        return path

    def _frwr3_align(self, block=True):
        if self.stack is None:
            return
        self._start_job(self._frwr3_align_job, [self.w_fa, self.w_align], self.w_fa_stop, self.w_fa_status, block)

    def _frwr3_align_job(self):
        p = self.params(all_images=True)
        self._status(self.w_fa_status, "FRWR3 alignment: preparing ...")
        t = time.time()
        st = {"n": 0, "planned": None, "work": None, "t": 0.0, "t0": None}
        self.w_fa_prog.max = 1.0
        self.w_fa_prog.value = 0

        def cb(label, j, R, eng=None):
            st["n"] += 1
            if st["t0"] is None:
                st["t0"] = time.time()
            if st["planned"] is None and eng is not None:
                try:
                    st["planned"], st["work"] = eng.planned_loops(), max(eng.planned_work()[1], 1.0)
                except Exception:  # noqa: BLE001
                    st["planned"], st["work"] = 0, None
            tot = st["planned"] or 0
            if eng is not None:
                # loop times differ strongly (distortion fits) and grow with the number of images
                n_px = self.stack.shape[0] * self.stack.shape[1] * self.stack.shape[2]
                frac, txt = self._frwr_eta(st, eng, label, t, extra_s=2e-7 * n_px)   # + warping at full size
            else:
                frac = min(st["n"] / tot, 1.0) if tot else 0.0
                txt = self._eta(t, frac)
            self.w_fa_prog.value = frac
            now = time.time()
            if now - st["t"] > 0.2:
                st["t"] = now
                self.w_fa_status.value = (f"{label} {j}: R = {R:.4f} — loop {st['n']}"
                                          + (f" of ≈ {tot}" if tot else "") + f", {txt}")
            if self._stop.is_set():
                raise StopRequested()

        try:
            out, info = align.frwr3_align(self.stack, p, binning=self.w_fa_bin.value,
                                          max_shift_px=self.w_fa_shift.value, fit_distortions=self.w_fa_dist.value,
                                          crop=self.w_fa_crop.value, backend=self.w_backend.value, callback=cb,
                                          final_loops=self.w_fa_loops.value)
        except StopRequested:
            out, info = None, {"stopped": True}
        except Exception as e:  # noqa: BLE001
            self._status(self.w_fa_status, f"FRWR3 alignment failed: {e!r}", ok=False)
            return
        if out is None or info.get("stopped"):
            self._status(self.w_fa_status, f"alignment stopped after {time.time() - t:.0f} s — working stack unchanged",
                         ok=False)
            return
        self.w_fa_prog.value = self.w_fa_prog.max
        self.stack = out
        self._thon_cache = None
        self.align_info = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()
                           if k not in ("result", "valid", "wave", "wave_start")}
        self.align_info["binning_fit"] = int(self.w_fa_bin.value)
        if self.w_fa_df.value:
            for r, d in zip(self.rows, info["refined_defoci_nm"]):
                r[1].value = float(round(d, 1))
        self._make_init_rec(info, time.time() - t)
        d = info["distortions"]
        from matplotlib.figure import Figure
        fig = Figure(figsize=(15, 3.2))
        ax = fig.subplots(1, 4)
        z = np.asarray(info["refined_defoci_nm"]) * 1e-3
        ax[0].plot(z, info["shifts_px"][:, 0], "o-", label="x")
        ax[0].plot(z, info["shifts_px"][:, 1], "s-", label="y")
        ax[0].set_ylabel("shift (px)")
        ax[0].legend()
        ax[1].plot(z, 1e3 * d[:, 0], "o-")
        ax[1].set_ylabel("rotation (mrad)")
        ax[2].plot(z, 1e3 * d[:, 1], "o-", label="zoom")
        ax[2].plot(z, 1e3 * np.hypot(d[:, 2], d[:, 3]), "s-", label="|elliptical|")
        ax[2].set_ylabel("magnification change (1e-3)")
        ax[2].legend()
        ax[3].imshow(out.std(0), cmap="gray")
        ax[3].set_title("std over aligned series")
        for a in ax[:3]:
            a.set_xlabel("C1 (µm)")
        fig.tight_layout()
        self._png_out(self.w_align_out, fig)
        self._show_init_rec()
        cb_ = info.get("crop")
        msg = (f"FRWR3 alignment done in {time.time() - t:.0f} s (R = {info['R_final']:.4f}); working stack "
               f"{out.shape[1]}×{out.shape[2]} px"
               + (f", cropped to rows {cb_[0]}–{cb_[2]}, columns {cb_[1]}–{cb_[3]}" if cb_ else ""))
        self._status(self.w_fa_status, msg)
        self._status(self.w_load_status, msg)

    def _make_init_rec(self, info, runtime):
        """Reconstruction object for the FRWR alignment wave (full resolution, aligned-stack geometry)."""
        from .reconstruct import Reconstruction
        if "wave" not in info:
            self.init_rec = None
            self.w_fa_save.disabled = self.w_fa_to5.disabled = True
            return
        res = info["result"]
        opts = {"binning_fit": int(self.w_fa_bin.value), "wave_box_in_aligned_stack": list(info["wave_box"]),
                "upsampling": "Fourier interpolation of the padded FRWR3 wave",
                "fit_distortions": bool(self.w_fa_dist.value), "max_drift_px": int(self.w_fa_shift.value),
                "final_loops": int(self.w_fa_loops.value)}
        self.init_wave_start = np.asarray(info["wave_start"], complex)
        extra = {"frwr3": res,
                 "distortions_full_resolution": np.asarray(info["distortions"], float).tolist(),
                 "drift_px_xy": np.asarray(info["shifts_px"], float).tolist(),
                 "crop_top_left_bottom_right": list(info.get("crop") or []),
                 "reference_plane": int(info["reference_plane"])}
        self.init_rec = Reconstruction(info["wave"], "frwr3_alignment", opts, [r[2] for r in res.rvalues],
                                       runtime, None, extra)
        self.recs = [r for r in self.recs if r[0] != "frwr_initial"] + [("frwr_initial", self.init_rec,
                                                                          self._init_params())]
        rv = [r[2] for r in res.rvalues]
        self.log_step("frwr_align", {**opts, "runtime_s": round(float(runtime), 1)},
                      (f"final R = {rv[-1]:.4g}; " if rv else "")
                      + f"largest drift {float(np.abs(np.asarray(info['shifts_px'], float)).max()):.1f} px.",
                      replace=True)
        self.w_fa_save.disabled = self.w_fa_to5.disabled = False
        self.w_fa_save_out.value = ""
        base = os.path.splitext(os.path.basename(str(getattr(self, "source_name", "") or "series").split(",")[0]))[0]
        self.w_fa_fname.value = f"{base}_frwr_aligned.nxs"

    def _init_params(self):
        """Parameters of the initial reconstruction: table values (refined defoci), working pixel size."""
        return self.params(all_images=True)

    def _show_init_rec(self):
        if self.init_rec is None:
            self.w_fa_wave_out.outputs = ()
            return
        w = self.init_rec.wave
        ph = np.angle(w * np.exp(-1j * np.angle(np.mean(w))))
        if self.w_fa_unwrap.value:
            from skimage.restoration import unwrap_phase
            ph = unwrap_phase(ph)
        from matplotlib.figure import Figure
        fig = Figure(figsize=(13, 3.6))
        ax = fig.subplots(1, 3)
        im = ax[0].imshow(np.abs(w), cmap="gray")
        ax[0].set_title(f"amplitude ({w.shape[0]}×{w.shape[1]} px)")
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        im = ax[1].imshow(ph, cmap="gray" if self.w_fa_unwrap.value else "twilight")
        ax[1].set_title("phase (rad)")
        fig.colorbar(im, ax=ax[1], fraction=0.046)
        h = np.asarray(self.init_rec.history, float)
        if h.size:
            ax[2].semilogy(h)
            ax[2].set_xlabel("loop")
            ax[2].set_title("R-value (all loops)")
        for a in ax[:2]:
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()
        self._png_out(self.w_fa_wave_out, fig)

    def _save_alignment(self, path=None):
        """Write aligned series + initial reconstruction + alignment results to an NXem file."""
        if self.init_rec is None or self.stack is None:
            return None
        meta = self.metadata()
        miss = nxem.missing_required(meta)
        if miss and not self.w_placeholder.value and path is None:
            self.w_fa_save_out.value = ("<span style='color:crimson'>Please fill in the required NXem fields ("
                                        + ", ".join(f"{s}/{lab}" for s, _, lab in miss)
                                        + ") in tab 1 or 6, or tick \"write 'unknown' for missing required "
                                          "metadata\" in tab 6.</span>")
            return None
        if path is None:
            out_dir = self._out_dir()
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, os.path.basename(self.w_fa_fname.value.strip() or "frwr_aligned.nxs"))
        p = self._init_params()
        extra = {k: v for k, v in self.meta.items() if k not in ("raw", "nxem_hints")}
        extra["frwr3_alignment"] = self.align_info
        nxem.write_focal_series(path, self.stack, p, extra_metadata=extra, metadata=meta,
                                description="focal series aligned and distortion-corrected by the FRWR alignment "
                                            "(inlineholo app, tab 4b)")
        nxem.write_reconstruction(path, self.init_rec, p, label="FRWR initial reconstruction", derived=True)
        val = self._validation_html(path) if self.w_validate.value else ""
        self.w_fa_save_out.value = f"written <code>{path}</code> ({os.path.getsize(path) / 1e6:.1f} MB){val}"
        self._register_output(path)
        return path

    def _validation_html(self, path):
        ok, probs, undoc = nxem.validate(path)
        if ok is None:
            return "<br><span style='color:#555'>validation skipped: " + probs[0] + "</span>"
        if ok:
            return ("<br><span style='color:green'>✔ valid according to the NXem application definition "
                    f"(pynxtools){'; ' + str(len(undoc)) + ' undocumented items' if undoc else ''}</span>")
        return ("<br><span style='color:crimson'>✘ NOT valid according to NXem:</span><br><small>"
                + "<br>".join(probs[:20]) + "</small>")

    def _prefill_metadata(self, path_or_bytes, name):
        """Take user metadata from a loaded NXem file (or simple heuristics for other formats)."""
        try:
            if str(name).lower().endswith((".nxs", ".h5", ".hdf5")):
                src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
                self.set_metadata(nxem.read_metadata(src))
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ export
    def save(self, path, metadata=None, derived=True, placeholders=True):
        """Write the session (focal series, all reconstructions, maps, curves) to an NXem file."""
        p = self.params()
        meta = json.loads(json.dumps(metadata or self.metadata(), default=str))
        self.log_step("export", dict(file=os.path.basename(path)), replace=True)
        from .eln import workflow_description
        wf_md = workflow_description(self.workflow_log)
        ex = meta.setdefault("experiment", {})
        ex["description"] = ((str(ex.get("description")).strip() + "\n\n") if ex.get("description") else "") + wf_md
        extra = {k: v for k, v in self.meta.items() if k != "raw"}
        if getattr(self, "align_info", None):
            extra["frwr3_alignment"] = self.align_info
        nxem.write_focal_series(path, self.used_stack(), p, description="experimental focal series (inlineholo app)",
                                extra_metadata=extra, metadata=meta)
        for m, rec, pr in self.recs:
            nxem.write_reconstruction(path, rec, pr, label=m, derived=derived)
        for d in self.maps:
            if d["kind"] == "strain":
                nxem.write_strain(path, d["array"], d["params"], d["g"], d["sigma"], source=d["source"])
            elif d["kind"] == "dih":
                opts = {k: v for k, v in d["options"].items() if v is not None}
                nxem.write_map(path, d["array"], d["params"], d["title"], f"dih_{d['key']}", method="dih_many_beam",
                               options=opts, units=None,     # 'real' is NX_UNITLESS: the unit is in long_name
                               long_name=d["title"] + {"nm": " (nm)", "mrad": " (rad)"}.get(d["unit"],
                                                                                            " (dimensionless)"))
        for c in self.curves:
            nxem.write_curve(path, c["q"], {"frc": c["frc"], "half_bit_threshold": c["threshold"]}, c["params"],
                             f"half-set FRC ({c['method']})", "half_set_frc", axis_name="q", axis_units="1/nm",
                             method=c["method"])
        return path

    def _save(self):
        if self.stack is None:
            return
        meta = self.metadata()
        miss = nxem.missing_required(meta)
        if miss and not self.w_placeholder.value:
            self.w_save_out.value = ("<span style='color:crimson'>Please fill in the required NXem fields ("
                                     + ", ".join(f"{s}/{lab}" for s, _, lab in miss)
                                     + ") or tick \"write 'unknown' for missing required metadata\".</span>")
            return
        fn = self.w_fname.value.strip() or "reconstruction.nxs"
        out_dir = self._out_dir()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, os.path.basename(fn))
        self.save(path, meta, derived=self.w_derived.value)
        size = os.path.getsize(path)
        link = ""
        if size < 60e6:
            b64 = base64.b64encode(open(path, "rb").read()).decode()
            link = (f"<a download='{os.path.basename(path)}' href='data:application/x-hdf5;base64,{b64}'>"
                    f"⬇ download {os.path.basename(path)}</a>")
        val = ""
        if self.w_validate.value:
            ok, probs, undoc = nxem.validate(path)
            if ok is None:
                val = "<br><span style='color:#555'>validation skipped: " + probs[0] + "</span>"
            elif ok:
                val = ("<br><span style='color:green'>✔ valid according to the NXem application definition "
                       f"(pynxtools){'; ' + str(len(undoc)) + ' undocumented items' if undoc else ''}</span>")
            else:
                val = ("<br><span style='color:crimson'>✘ NOT valid according to NXem:</span><br><small>"
                       + "<br>".join(probs[:20]) + "</small>")
        if miss:
            val += ("<br><span style='color:#b26a00'>written with 'unknown' for: "
                    + ", ".join(f"{s}/{lab}" for s, _, lab in miss) + "</span>")
        self.last_nxem = path
        eln_files, eln_msg = [], ""
        if self.w_eln.value:
            target = self._nomad_target() if self.w_nm_auto.value else None
            try:
                eln_files = self.write_eln(path, target)
                eln_msg = ("<br>NOMAD ELN and workflow: " + ", ".join(
                    f"<code>{os.path.basename(f)}</code>" for f in eln_files if f.endswith(".yaml")))
            except Exception as e:  # noqa: BLE001
                eln_msg = f"<br><span style='color:#b26a00'>ELN / workflow not written: {e!r}</span>"
        self.w_save_out.value = (f"written <code>{path}</code> ({size / 1e6:.1f} MB, {len(self.recs)} "
                                 f"reconstruction(s), {len(self.maps)} strain map(s), {len(self.curves)} "
                                 f"FRC curve(s)) {link}{val}{eln_msg}")
        self._show_workflow()
        self._register_outputs([path] + eln_files)

    # ------------------------------------------------------------------ ELN and workflow
    def _eln_name(self, nxem_path):
        return os.path.splitext(os.path.basename(nxem_path))[0]

    def _input_files_for_eln(self, target=None):
        """(paths of the focal series in the upload, local files to upload with the results).
        target: (upload_id, folder) of the NOMAD upload the results go to, or None (local folder)."""
        ctx = getattr(self, "nomad_ctx", None)
        if ctx and target and ctx.get("upload_id") == target[0] and ctx.get("files"):
            return list(ctx["files"]), []                       # the series is already in that upload
        local = [f for f in (getattr(self, "upload_files", None) or []) if os.path.isfile(f)]
        if not local and getattr(self, "source_path", None) and os.path.isfile(self.source_path):
            local = [self.source_path]
        pre = (target[1].strip("/") + "/") if target and target[1].strip("/") else ""
        names = [pre + os.path.basename(f) for f in local] or [str(getattr(self, "source_name", "focal series"))]
        return names, local

    def _eln_settings(self):
        p = self.params(all_images=True)
        return {"acceleration_voltage": float(p.voltage_V), "pixel_size": float(self.w_pix.value),
                "number_of_images": int(len(p.defoci_nm)), "defocus_min": float(np.min(p.defoci_nm)),
                "defocus_max": float(np.max(p.defoci_nm)), "spherical_aberration": float(p.aberrations.get("c_3_0", 0)),
                "focal_spread": float(p.focal_spread_nm), "convergence_semi_angle": float(p.convergence_rad * 1e3)}

    def _eln_reconstructions(self):
        out = []
        for m, rec, _ in self.recs:
            hist = [h for h in (rec.history or []) if np.isscalar(h)]
            out.append({"method": m, "iterations": len(rec.history or []), "runtime": round(float(rec.runtime_s), 2),
                        "final_r_value": float(hist[-1]) if hist else None})
        return out

    def write_eln(self, nxem_path, target=None):
        """ELN entry, datasets and workflow archives (+ SVG preview) next to the NXem file.  target: (upload_id,
        folder) of the NOMAD upload the files will go to (references are paths in that upload), or None.
        Returns the files to upload with the NXem file (archives and, if needed, the focal series)."""
        from . import eln
        folder = os.path.dirname(os.path.abspath(nxem_path))
        name = self._eln_name(nxem_path)
        inputs, local = self._input_files_for_eln(target)
        pre = (target[1].strip("/") + "/") if target and target[1].strip("/") else ""
        hist = [h for _, rec, _ in self.recs[-1:] for h in (rec.history or []) if np.isscalar(h)]
        summary = (f"{len(self.recs)} reconstruction(s): " + ", ".join(m for m, _, _ in self.recs)
                   + (f"; last R = {hist[-1]:.4g}" if hist else "")) if self.recs else None
        paths = eln.write_eln(folder, name, self.workflow_log, inputs, pre + os.path.basename(nxem_path), prefix=pre,
                              schema="core" if self.w_eln_core.value else "plugin", settings=self._eln_settings(),
                              metadata=self.metadata(), results_summary=summary,
                              reconstructions=self._eln_reconstructions())
        self.last_workflow = paths
        files = [v for k, v in paths.items() if k != "svg"] + [paths["svg"]]
        return files + [f for f in local if os.path.abspath(f) != os.path.abspath(nxem_path)]

    def _nomad_target(self):
        uid = self.w_nm_upload.value.strip()
        return (uid, self.w_nm_results.value.strip().strip("/")) if uid else ("", self.w_nm_results.value.strip())

    def _show_workflow(self):
        """Preview of the workflow (as on NOMAD's workflow card) and of the description."""
        from . import eln
        name = self._eln_name(getattr(self, "last_nxem", None) or self.w_fname.value or "session")
        log = eln.WorkflowLog(self.workflow_log)
        if not any(st["key"] == "export" for st in log):
            log.add("export", dict(file=self.w_fname.value))
        inputs, _ = self._input_files_for_eln(None)
        arch = eln.build_archives(name, log, inputs, os.path.basename(self.w_fname.value or name + ".nxs"))
        wf = arch[f"{name}.workflow.archive.yaml"]["workflow2"]
        self.w_wf_svg.value = eln.workflow_svg(wf)
        self.w_wf_desc.value = _md_to_html(eln.workflow_description(log))
