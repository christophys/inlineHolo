# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Tab 7 of the app: many-beam dark-field inline holography (DIH) strain analysis.

Port of the DM "DIH-GPA" tools (createDFstack.s, DFHoloAllBeamsGPA.s and its displacement
variant): several DF reconstructions (one per reflection g) are combined into strain / rotation
maps (or the displacement field) by weighted least squares; see :mod:`inlineholo.dih`.
"""
from __future__ import annotations

import os
import tempfile

import ipywidgets as W
import numpy as np

from . import dih

_S = {"description_width": "initial"}


def _ft(desc, v, w="150px"):
    return W.FloatText(value=v, description=desc, style=_S, layout=W.Layout(width=w))


def _it(desc, v, w="120px"):
    return W.IntText(value=v, description=desc, style=_S, layout=W.Layout(width=w))


class DIHTabMixin:
    """Widgets and callbacks of tab 7 (mixed into :class:`inlineholo.app.App`)."""

    def _build_dih_tab(self):
        self.dih_refls = []
        self.dih_result = None
        # ---- single reflection from the current reconstruction (former tab 5 section)
        self.w_gy = W.FloatText(value=0.0, description="g_y (1/nm)", layout=W.Layout(width="200px"))
        self.w_gx = W.FloatText(value=3.857, description="g_x (1/nm)", layout=W.Layout(width="200px"))
        self.w_ssig = W.FloatText(value=1.0, description="smooth (px)", layout=W.Layout(width="200px"))
        self.w_strain = W.Button(description="ε_gg of current reconstruction", icon="arrows-h",
                                 layout=W.Layout(width="250px"),
                                 tooltip="strain along g from the geometric phase of the wave of tab 5")
        self.w_strain.on_click(lambda _: self._strain())
        self.w_strain_out = W.Output()
        # ---- lattice and geometry
        self.w_dih_preset = W.Dropdown(options=["custom"] + list(dih.LATTICES), value="Si", description="crystal",
                                       style=_S, layout=W.Layout(width="260px"))
        a, b, c, al, be, ga = dih.LATTICES["Si"]
        self.w_dih_a, self.w_dih_b, self.w_dih_c = _ft("a (nm)", a), _ft("b (nm)", b), _ft("c (nm)", c)
        self.w_dih_al, self.w_dih_be, self.w_dih_ga = _ft("α (°)", al), _ft("β (°)", be), _ft("γ (°)", ga)
        self.w_dih_zone = W.Text(value="0 0 1", description="zone axis [uvw]", style=_S,
                                 layout=W.Layout(width="220px"))
        self.w_dih_x = W.Text(value="1 0 0", description="x axis along (hkl)", style=_S,
                              layout=W.Layout(width="220px"))
        self.w_dih_rot = _ft("rotation of crystal vs image x (°)", 0.0, "300px")
        self.w_dih_preset.observe(self._dih_preset, names="value")
        for w in (self.w_dih_a, self.w_dih_b, self.w_dih_c, self.w_dih_al, self.w_dih_be, self.w_dih_ga,
                  self.w_dih_zone, self.w_dih_x, self.w_dih_rot):
            w.observe(lambda _: self._dih_table(), names="value")
        # ---- adding reflections
        self.w_dih_hkl = W.Text(value="2 0 0", description="reflection (h k l)", style=_S,
                                layout=W.Layout(width="230px"))
        self.w_dih_add_rec = W.Button(description="add current reconstruction (tab 5)", icon="plus",
                                      layout=W.Layout(width="270px"))
        self.w_dih_add_init = W.Button(description="add FRWR initial reconstruction (tab 4b)", icon="plus",
                                       layout=W.Layout(width="300px"))
        self.w_dih_add_rec.on_click(lambda _: self._dih_add_current("rec"))
        self.w_dih_add_init.on_click(lambda _: self._dih_add_current("init"))
        self.w_dih_path = W.Text(value="", placeholder="file(s) or folder: amplitude/phase pairs, complex .img/.npy, "
                                                       "DM DF stacks; several separated by ;",
                                 layout=W.Layout(width="640px"))
        self.w_dih_add_path = W.Button(description="add from path", icon="folder-open")
        self.w_dih_add_path.on_click(lambda _: self._dih_add_paths(self.w_dih_path.value))
        self.w_dih_upload = W.FileUpload(accept=".dm3,.dm4,.img,.npy,.tif,.tiff", multiple=True,
                                         description="upload files")
        self.w_dih_upload.observe(self._dih_on_upload, names="value")
        self.w_dih_rows = W.VBox()
        self.w_dih_msg = W.HTML()
        # ---- analysis options
        self.w_dih_mode = W.ToggleButtons(options=[("strain / rotation", "strain"), ("displacement", "displacement")],
                                          value="strain", style={"button_width": "130px"})
        self.w_dih_weight = W.Dropdown(options=[("intensity A² (DM default)", "intensity"),
                                                ("amplitude A", "amplitude"), ("uniform", "uniform")],
                                       value="intensity", description="weights", style=_S,
                                       layout=W.Layout(width="300px"))
        self.w_dih_refcorr = W.Checkbox(value=True, indent=False,
                                        description="reference correction (zero mean gradient in reference area)", layout=W.Layout(width="auto"))
        self.w_dih_box = [_it(n, -1, "110px") for n in ("top", "left", "bottom", "right")]
        self.w_dih_res = _ft("resolution (nm, 0 = none)", 1.0, "240px")
        self.w_dih_smooth = _ft("smooth edge (fraction)", 0.0, "220px")
        self.w_dih_lim = _ft("display ± limit (%)", 2.0, "200px")
        self.w_dih_i1 = W.Dropdown(options=[], description="overlay", style=_S, layout=W.Layout(width="260px"))
        self.w_dih_i2 = W.Dropdown(options=[], description="with", style=_S, layout=W.Layout(width="240px"))
        self.w_dih_ovl_kind = W.ToggleButtons(options=["amplitude", "phase"], value="amplitude",
                                              style={"button_width": "90px"})
        self.w_dih_ovl = W.Button(description="show overlay", icon="clone")
        self.w_dih_ovl.on_click(lambda _: self._dih_overlay())
        self.w_dih_auto = W.Button(description="auto-align amplitudes", icon="arrows",
                                   tooltip="integer shifts by phase correlation of the amplitudes (first used "
                                           "reflection = reference)")
        self.w_dih_auto.on_click(lambda _: self._dih_autoalign())
        self.w_dih_run = W.Button(description="compute maps", icon="play", button_style="primary")
        self.w_dih_run.on_click(lambda _: self._dih_compute())
        self.w_dih_out = W.Output()
        self.w_dih_status = W.HTML()
        lat = W.VBox([W.HBox([self.w_dih_preset, self.w_dih_a, self.w_dih_b, self.w_dih_c]),
                      W.HBox([self.w_dih_al, self.w_dih_be, self.w_dih_ga]),
                      W.HBox([self.w_dih_zone, self.w_dih_x, self.w_dih_rot])])
        return W.VBox([
            W.HTML("<b>7. Dark-field inline holography (DIH): strain from one or several reflections.</b> "
                   "Reconstruct the dark-field focal series of each reflection g (tabs 1–5) and add the waves "
                   "here, or load amplitude/phase maps, complex waves (e.g. FRWR3 <code>avgWave.img</code>) or "
                   "DF stacks of the DM script <i>createDFstack</i>. The phase gradients of all reflections are "
                   "combined by weighted least squares into the 2D displacement gradient (port of the DM script "
                   "<i>Many-Beam DIH-GPA</i>): ε<sub>xx</sub>, ε<sub>yy</sub>, ε<sub>xy</sub> and the rotation "
                   "ω; with one reflection only ε<sub>gg</sub> along g. x points right, y up, z (zone axis) "
                   "towards the viewer."),
            W.HTML("<b>Crystal and orientation</b> (2D reflection vectors from the unit cell):"), lat,
            W.HTML("<b>Reflections</b>"),
            W.HBox([self.w_dih_hkl, self.w_dih_add_rec, self.w_dih_add_init]),
            *([] if getattr(self, "server_mode", False) else [W.HBox([self.w_dih_path, self.w_dih_add_path])]),
            self.w_dih_upload,
            self.w_dih_rows, self.w_dih_msg,
            W.HTML("<b>Alignment of the reflections</b> (shifts in pixels, x right / y down; edit in the table):"),
            W.HBox([self.w_dih_i1, self.w_dih_i2, self.w_dih_ovl_kind, self.w_dih_ovl, self.w_dih_auto]),
            W.HTML("<b>Analysis</b> — reference area (pixels; −1: central half):"),
            W.HBox(self.w_dih_box),
            W.HBox([self.w_dih_mode, self.w_dih_weight]), self.w_dih_refcorr,
            W.HBox([self.w_dih_res, self.w_dih_smooth, self.w_dih_lim]),
            W.HBox([self.w_dih_run, self.w_dih_status]), self.w_dih_out,
            W.HTML("<hr><b>Single reflection, direct</b>: ε<sub>gg</sub> = −ĝ·∇φ<sub>g</sub>/(2π|g|) of the "
                   "current reconstruction of tab 5 for a given g (relative to the median of the field of view)."),
            W.HBox([self.w_gx, self.w_gy, self.w_ssig, self.w_strain]), self.w_strain_out])

    # ------------------------------------------------------------------ helpers
    def _dih_preset(self, ch):
        if ch["new"] in dih.LATTICES:
            vals = dih.LATTICES[ch["new"]]
            for w, v in zip((self.w_dih_a, self.w_dih_b, self.w_dih_c, self.w_dih_al, self.w_dih_be,
                             self.w_dih_ga), vals):
                w.value = v

    def dih_lattice(self):
        return (self.w_dih_a.value, self.w_dih_b.value, self.w_dih_c.value, self.w_dih_al.value,
                self.w_dih_be.value, self.w_dih_ga.value)

    def _dih_geometry(self):
        zone = dih.parse_hkl(self.w_dih_zone.value)
        xh = dih.parse_hkl(self.w_dih_x.value)
        if zone is None or xh is None:
            raise ValueError("zone axis and x axis must be given as three integers")
        return dict(lattice=self.dih_lattice(), zone=zone, x_hkl=xh, rotation_deg=self.w_dih_rot.value)

    def _dih_gvec(self, hkl):
        try:
            return dih.g_vectors_2d([hkl], **self._dih_geometry())[0]
        except Exception:  # noqa: BLE001
            return None

    def _dih_table(self):
        rows = [W.HBox([W.HTML("<b>use</b>", layout=W.Layout(width="45px")),
                        W.HTML("<b>name</b>", layout=W.Layout(width="230px")),
                        W.HTML("<b>h k l</b>", layout=W.Layout(width="100px")),
                        W.HTML("<b>shift x</b>", layout=W.Layout(width="85px")),
                        W.HTML("<b>shift y</b>", layout=W.Layout(width="85px")),
                        W.HTML("<b>g (1/nm)</b>", layout=W.Layout(width="150px")),
                        W.HTML("<b>size, pixel</b>", layout=W.Layout(width="160px"))])]
        for k, r in enumerate(self.dih_refls):
            u = W.Checkbox(value=r.use, indent=False, layout=W.Layout(width="45px"))
            h = W.Text(value=" ".join(str(v) for v in r.hkl), layout=W.Layout(width="100px"))
            sx = W.FloatText(value=r.shift_px[0], layout=W.Layout(width="85px"))
            sy = W.FloatText(value=r.shift_px[1], layout=W.Layout(width="85px"))
            g = self._dih_gvec(r.hkl)
            gl = W.HTML(f"({g[0]:.3f}, {g[1]:.3f})" if g is not None else "?", layout=W.Layout(width="150px"))
            rm = W.Button(icon="trash", layout=W.Layout(width="40px"), tooltip="remove")

            def upd(_, k=k, u=u, h=h, sx=sx, sy=sy, gl=gl):
                r = self.dih_refls[k]
                r.use = u.value
                hk = dih.parse_hkl(h.value)
                if hk is not None:
                    r.hkl = hk
                    g = self._dih_gvec(hk)
                    gl.value = f"({g[0]:.3f}, {g[1]:.3f})" if g is not None else "?"
                r.shift_px = (sx.value, sy.value)

            for w in (u, h, sx, sy):
                w.observe(upd, names="value")
            rm.on_click(lambda _, k=k: self._dih_remove(k))
            rows.append(W.HBox([u, W.HTML(r.name, layout=W.Layout(width="230px")), h, sx, sy, gl,
                                W.HTML(f"{r.amplitude.shape[1]}×{r.amplitude.shape[0]}, {r.pixel_nm:.4g} nm",
                                       layout=W.Layout(width="160px")), rm]))
        self.w_dih_rows.children = rows if self.dih_refls else [W.HTML("<i>no reflections yet</i>")]
        opts = [(f"{k}: {r.name}", k) for k, r in enumerate(self.dih_refls)]
        for w, d in ((self.w_dih_i1, 0), (self.w_dih_i2, 1)):
            cur = w.value
            w.options = opts
            if opts:
                w.value = cur if cur in [o[1] for o in opts] else min(d, len(opts) - 1)

    def _dih_remove(self, k):
        del self.dih_refls[k]
        self._dih_table()

    def _dih_msg(self, msg, ok=True):
        self._status(self.w_dih_msg, msg, ok)

    # ------------------------------------------------------------------ adding data
    def _dih_add_current(self, which):
        hkl = dih.parse_hkl(self.w_dih_hkl.value)
        if hkl is None:
            self._dih_msg("enter the reflection as three integers, e.g. 2 0 0", ok=False)
            return
        rec = self.rec if which == "rec" else getattr(self, "init_rec", None)
        if rec is None:
            self._dih_msg("no reconstruction yet (tab 5)" if which == "rec" else
                          "no FRWR initial reconstruction yet (tab 4b)", ok=False)
            return
        name = f"{getattr(self, 'source_name', '') or 'series'} · {rec.method} {hkl}"
        r = dih.from_wave(rec.wave, hkl, self.w_pix.value, name, source=str(getattr(self, "source_name", "")))
        r.meta = {"method": rec.method, "defoci_nm": [float(v) for v in self.params().defoci_nm]}
        self.dih_refls.append(r)
        self._dih_table()
        self._dih_msg(f"added {name}")

    def _dih_add_paths(self, text):
        paths = []
        for p in [t.strip() for t in str(text).split(";") if t.strip()]:
            p = os.path.expanduser(p)
            if not self._allowed_path(p):
                self._dih_msg("reading files of the server is not possible here", ok=False)
                return
            if os.path.isdir(p):
                fs = sorted(os.path.join(p, f) for f in os.listdir(p)
                            if f.lower().endswith((".dm3", ".dm4", ".img", ".npy", ".tif", ".tiff"))
                            and ("amplitude" in f.lower() or "wave" in f.lower()))
                paths += fs
            elif os.path.isfile(p):
                paths.append(p)
            else:
                self._dih_msg(f"not found: {p}", ok=False)
                return
        if not paths:
            self._dih_msg("no files found", ok=False)
            return
        skipped = []
        try:
            new, settings = dih.load_reflections(paths, skipped=skipped)
        except Exception as e:  # noqa: BLE001
            self._dih_msg(f"could not read: {e!r}", ok=False)
            return
        self._dih_take(new, settings, skipped)

    def _dih_on_upload(self, change):
        files = change["new"]
        if not files:
            return
        d = tempfile.mkdtemp(prefix="dih_")
        paths = []
        for f in files:
            p = os.path.join(d, f["name"])
            with open(p, "wb") as fh:
                fh.write(bytes(f["content"]))
            paths.append(p)
        skipped = []
        try:
            new, settings = dih.load_reflections(paths, skipped=skipped)
        except Exception as e:  # noqa: BLE001
            self._dih_msg(f"could not read: {e!r}", ok=False)
            return
        for r in new:
            r.source = os.path.basename(r.source)
        self._dih_take(new, settings, skipped)

    def _dih_take(self, new, settings, skipped=()):
        self.dih_refls += new
        if settings.get("lattice"):
            self.w_dih_preset.value = "custom"
            for w, v in zip((self.w_dih_a, self.w_dih_b, self.w_dih_c, self.w_dih_al, self.w_dih_be,
                             self.w_dih_ga), settings["lattice"]):
                w.value = float(v)
        if settings.get("zone"):
            self.w_dih_zone.value = " ".join(str(v) for v in settings["zone"])
        if settings.get("x_hkl"):
            self.w_dih_x.value = " ".join(str(v) for v in settings["x_hkl"])
        if settings.get("ref_box"):
            for w, v in zip(self.w_dih_box, settings["ref_box"]):
                w.value = int(v)
        if settings.get("resolution_nm"):
            self.w_dih_res.value = float(settings["resolution_nm"])
        if "smooth" in settings:
            self.w_dih_smooth.value = float(settings["smooth"])
        self._dih_table()
        miss = [r.name for r in new if r.hkl == (0, 0, 0)]
        self._dih_msg(f"added {len(new)} reflection(s)" + (f"; set h k l for: {', '.join(miss)}" if miss else "")
                      + ("; lattice, zone, reference area and resolution taken from the DFHoloGPA tags"
                         if settings else "")
                      + (f"; skipped (no phase file / not complex): {', '.join(skipped)}" if skipped else ""),
                      ok=not miss)

    # ------------------------------------------------------------------ alignment
    def _dih_overlay(self):
        if len(self.dih_refls) < 2:
            self._dih_msg("add at least two reflections", ok=False)
            return
        i1, i2 = self.w_dih_i1.value, self.w_dih_i2.value
        sel = []
        for k in (i1, i2):
            r = self.dih_refls[k]
            sel.append(dih.DFReflection(r.hkl, r.amplitude, r.phase, r.pixel_nm, shift_px=r.shift_px))
        _, A, P = dih.common_frame(sel)
        kind = self.w_dih_ovl_kind.value
        im = dih.overlay_rgb(*(A if kind == "amplitude" else
                               [dih.phase_gradients(p, 1.0)[0] for p in P]))
        from matplotlib.figure import Figure
        fig = Figure(figsize=(6, 6))
        ax = fig.subplots()
        ax.imshow(im)
        ax.set_title(f"green: {self.dih_refls[i1].name}\nred: {self.dih_refls[i2].name}"
                     + (" (x-gradient of the phase)" if kind == "phase" else ""), fontsize=8)
        self._dih_draw_box(ax, A.shape[1:])
        fig.tight_layout()
        self._png_out(self.w_dih_out, fig)

    def _dih_autoalign(self):
        if len(self.dih_refls) < 2:
            self._dih_msg("add at least two reflections", ok=False)
            return
        used = [r for r in self.dih_refls if r.use]
        sh = dih.auto_shift(used)
        for r, s in zip(used, sh):
            r.shift_px = (round(s[0]), round(s[1]))
        self._dih_table()
        self._dih_msg("shifts: " + ", ".join(f"{r.name}: ({r.shift_px[0]:.0f}, {r.shift_px[1]:.0f})" for r in used))

    def _dih_box(self, shape):
        v = [w.value for w in self.w_dih_box]
        if min(v) < 0:
            return None
        return tuple(v)

    def _dih_draw_box(self, ax, shape):
        from matplotlib.patches import Rectangle
        box = self._dih_box(shape)
        if box is None:
            ny, nx = shape
            box = (ny // 4, nx // 4, 3 * ny // 4, 3 * nx // 4)
        t, l, b, r = box
        ax.add_patch(Rectangle((l, t), r - l, b - t, fill=False, ec="k", ls="--", lw=0.8))

    # ------------------------------------------------------------------ analysis
    def _dih_compute(self):
        used = [r for r in self.dih_refls if r.use]
        if not used:
            self._dih_msg("no reflection selected", ok=False)
            return None
        pix = {round(r.pixel_nm, 6) for r in used}
        warn = "" if len(pix) == 1 else (f"<br><span style='color:#b26a00'>different pixel sizes {sorted(pix)} "
                                         f"nm: the first one is used</span>")
        try:
            geo = self._dih_geometry()
            shape = (max(r.amplitude.shape[0] for r in used), max(r.amplitude.shape[1] for r in used))
            kw = dict(ref_box=self._dih_box(shape), weight=self.w_dih_weight.value,
                      resolution_nm=self.w_dih_res.value or None, smooth=self.w_dih_smooth.value, **geo)
            if self.w_dih_mode.value == "strain":
                res = dih.strain_from_phases(self.dih_refls, reference_correction=self.w_dih_refcorr.value, **kw)
            else:
                res = dih.displacement_from_phases(self.dih_refls, **kw)
        except Exception as e:  # noqa: BLE001
            self._status(self.w_dih_status, f"error: {e!r}", ok=False)
            return None
        res["mode"] = self.w_dih_mode.value
        res["settings"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}
        self.dih_result = res
        self._dih_show(res)
        self._dih_to_export(res)
        n = len(used)
        self._status(self.w_dih_status, f"{n} reflection(s): "
                     + ("ε_gg along g only (one direction)" if "e_gg" in res else
                        ("ε_xx, ε_yy, ε_xy, rotation" if res["mode"] == "strain" else "u_x, u_y"))
                     + " — added to the NXem export (tab 6)" + warn)
        return res

    def _dih_maps(self, res):
        if "e_gg" in res:
            return [("e_gg", res["e_gg"], "strain ε_gg along g", "%")]
        if res["mode"] == "strain":
            return [("e_xx", res["e_xx"], "ε_xx", "%"), ("e_yy", res["e_yy"], "ε_yy", "%"),
                    ("e_xy", res["e_xy"], "ε_xy (symmetric shear)", "%"),
                    ("rotation", res["rotation"], "rotation ω (counter-clockwise)", "mrad")]
        return [("u_x", res["u_x"], "u_x", "nm"), ("u_y", res["u_y"], "u_y", "nm")]

    def _dih_show(self, res):
        from matplotlib.figure import Figure
        maps = self._dih_maps(res)
        fig = Figure(figsize=(4.2 * len(maps) + 1, 4.2))
        axs = np.atleast_1d(fig.subplots(1, len(maps)))
        lim = self.w_dih_lim.value
        for ax, (key, m, title, unit) in zip(axs, maps):
            v = 100 * m if unit == "%" else (1e3 * m if unit == "mrad" else m)
            if unit in ("%", "mrad"):
                im = ax.imshow(v, cmap="RdBu_r", vmin=-lim * (10 if unit == "mrad" else 1),
                               vmax=lim * (10 if unit == "mrad" else 1))
            else:
                vm = np.nanpercentile(np.abs(v), 99)
                im = ax.imshow(v, cmap="RdBu_r", vmin=-vm, vmax=vm)
            ax.set_title(f"{title} ({unit})", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
            self._dih_draw_box(ax, m.shape)
        g = np.asarray(res["g"])
        ax0 = axs[0]
        c = np.array(m.shape[::-1]) * 0.12
        sc = 0.08 * min(m.shape) / max(np.abs(g).max(), 1e-9)
        for (gx, gy), hkl in zip(g, res["hkl"]):
            ax0.annotate("", xy=(c[0] + sc * gx, c[1] - sc * gy), xytext=tuple(c),
                         arrowprops=dict(arrowstyle="->", color="k", lw=1))
            ax0.text(c[0] + 1.15 * sc * gx, c[1] - 1.15 * sc * gy, "".join(str(v) for v in hkl), fontsize=7)
        fig.tight_layout()
        self._png_out(self.w_dih_out, fig)

    def _dih_to_export(self, res):
        self.maps = [m for m in self.maps if m.get("kind") != "dih"]
        from .params import ImagingParameters
        if hasattr(self, "rows") and self.rows:
            p = self.params(all_images=True)
            p.pixel_size_nm = float(res["pixel_nm"])
        else:
            p = ImagingParameters(voltage_V=self.w_kv.value * 1e3, pixel_size_nm=float(res["pixel_nm"]),
                                  defoci_nm=[0.0])
        for key, m, title, unit in self._dih_maps(res):
            self.maps.append(dict(kind="dih", key=key, array=np.asarray(m, float), title=f"DIH {title}",
                                  unit=unit, params=p,  # stored dimensionless / rad / nm
                                  options={"reflections": [list(h) for h in res["hkl"]],
                                           "g_inv_nm": np.asarray(res["g"]).tolist(),
                                           "names": res.get("names", []), **res["settings"]}))
