# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""NOMAD import (tab 1) and export (tab 6) of the app.

Works like the NOMAD upload of the TEM Alignment Twin: NOMAD production, the test server or an
Oasis (API URL); an API token (NOMAD user menu -> API tokens / app token, with permission to read
and write uploads), kept only in memory for this session; the upload ID of the NOMAD project
(upload); a subfolder in that upload for the results.  Files imported from an upload can be
written back to the same upload, automatically after every saved file if wanted.
"""
from __future__ import annotations

import os
import re
import threading

import ipywidgets as W

from .nomad import NOMAD_URLS, NomadClient, NomadError, default_results_folder

_S = {"description_width": "110px"}
_IMG_IDX = re.compile(r"^(.*?)(\d+)\.img$", re.IGNORECASE)


class NomadMixin:
    """Widgets and callbacks for NOMAD (mixed into :class:`inlineholo.app.App`)."""

    _NOMAD_HELP = ("You need a NOMAD account and an <b>API token</b>: create one in NOMAD (your user menu → API "
                   "tokens / app token, with permission to read and write uploads), or ask the owner of the NOMAD "
                   "project for a token and the <b>upload ID</b> of the project. The token is kept only for this "
                   "session (not stored on disk). The computer running this app needs internet access to NOMAD.")

    def _build_nomad(self):
        self.generated_files = []
        self.nomad_ctx = None
        lay = W.Layout(width="440px")
        self.w_nm_url = W.Dropdown(options=[("nomad-lab.eu (production)", "production"),
                                            ("nomad-lab.eu (test, deleted regularly)", "test"),
                                            ("other (Oasis): API URL →", "custom")],
                                   value="production", description="NOMAD", style=_S, layout=lay)
        self.w_nm_custom = W.Text(placeholder="https://your-oasis/nomad-oasis/api/v1", description="API URL",
                                  style=_S, layout=lay, disabled=True)
        self.w_nm_url.observe(lambda ch: setattr(self.w_nm_custom, "disabled", ch["new"] != "custom"), names="value")
        self.w_nm_token = W.Password(placeholder="kept for this session only", description="API token", style=_S,
                                     layout=lay)
        self.w_nm_upload = W.Text(placeholder="ID of the NOMAD project (upload)", description="Upload ID", style=_S,
                                  layout=lay)
        self.w_nm_my = W.Button(description="my uploads", icon="list", tooltip="list your NOMAD uploads (token)")
        self.w_nm_uploads = W.Dropdown(options=[], description="", layout=W.Layout(width="330px"))
        self.w_nm_uploads.observe(lambda ch: ch["new"] and setattr(self.w_nm_upload, "value", ch["new"]),
                                  names="value")
        self.w_nm_my.on_click(lambda _: self._nm_list_uploads())
        # ---- tab 1: browse / import
        self.w_nm_path = W.Text(value="", description="folder", placeholder="(top level of the upload)", style=_S,
                                layout=lay)
        self.w_nm_up = W.Button(description="up", icon="level-up", layout=W.Layout(width="70px"))
        self.w_nm_browse = W.Button(description="list files", icon="folder-open")
        self.w_nm_list = W.Select(options=[], rows=8, layout=W.Layout(width="640px"))
        self.w_nm_load = W.Button(description="download and load selected", icon="download", button_style="primary",
                                  layout=W.Layout(width="230px"),
                                  tooltip="an indexed .img frame (image_3.img) loads the whole image_*.img series")
        self.w_nm_load_folder = W.Button(description="download folder as series", icon="files-o",
                                         layout=W.Layout(width="220px"),
                                         tooltip="all single-image files of this folder, ordered by the defocus in "
                                                 "their names (F-1000nm.dm3) or by their numbers")
        self.w_nm_auto = W.Checkbox(value=False, indent=False,
                                    description="push results automatically back to this NOMAD upload (every file "
                                                "written in tabs 1, 4b and 6)", layout=W.Layout(width="auto"))
        self.w_nm_results = W.Text(value=default_results_folder(), description="Subfolder", style=_S, layout=lay,
                                   placeholder="e.g. inlineholo_results_2026-11-06")
        self.w_nm_status1 = W.HTML()
        self.w_nm_up.on_click(lambda _: self._nm_up())
        self.w_nm_browse.on_click(lambda _: self._nm_browse())
        self.w_nm_list.observe(self._nm_select, names="value")
        self.w_nm_load.on_click(lambda _: self._nm_import(block=False))
        self.w_nm_load_folder.on_click(lambda _: self._nm_import(folder=True, block=False))
        # ---- tab 6: export
        self.w_nm_newname = W.Text(placeholder="used when creating a new upload", description="New upload name",
                                   style=_S, layout=lay)
        self.w_nm_push = W.Button(description="upload generated files", icon="cloud-upload", button_style="warning",
                                  layout=W.Layout(width="220px"))
        self.w_nm_push.on_click(lambda _: self._nm_push(block=False))
        self.w_nm_files = W.HTML()
        self.w_nm_status6 = W.HTML()

    def _nm_settings(self):
        return W.VBox([W.HBox([self.w_nm_url, self.w_nm_custom]), W.HBox([self.w_nm_token, self.w_nm_upload]),
                       W.HBox([self.w_nm_my, self.w_nm_uploads])])

    def nomad_import_box(self):
        box = W.VBox([W.HTML("<b>Import from NOMAD</b>: browse the raw files of a NOMAD upload and load a focal "
                             "series (any format of this tab). " + self._NOMAD_HELP), self._nm_settings(),
                      W.HBox([self.w_nm_path, self.w_nm_up, self.w_nm_browse]), self.w_nm_list,
                      W.HBox([self.w_nm_load, self.w_nm_load_folder]),
                      W.HBox([self.w_nm_results]), self.w_nm_auto, self.w_nm_status1])
        acc = W.Accordion(children=[box], titles=("import from NOMAD (and push results back)",))
        acc.selected_index = None
        return acc

    def nomad_export_box(self):
        return W.VBox([W.HTML("<hr><b>Upload the generated files to NOMAD</b>: sends every file written in this "
                              "session (NXem files of tabs 1, 4b and 6) to a NOMAD upload. With the upload ID of the "
                              "project the files were imported from (filled in automatically), the results go back "
                              "into that project; a <b>subfolder</b> collects the files of this session. Without an "
                              "upload ID a new upload is created. " + self._NOMAD_HELP),
                       self._nm_settings(), W.HBox([self.w_nm_results, self.w_nm_newname]),
                       self.w_nm_files, W.HBox([self.w_nm_push]), self.w_nm_status6])

    # ------------------------------------------------------------------ helpers
    def _nm_client(self):
        url = self.w_nm_custom.value.strip() if self.w_nm_url.value == "custom" else self.w_nm_url.value
        if not url:
            raise NomadError("enter the API URL of the Oasis")
        return NomadClient(url, self.w_nm_token.value, opener=getattr(self, "_nm_opener", None))

    def _nm_msg(self, w, msg, ok=True):
        self._status(w, msg, ok)

    def _nm_list_uploads(self):
        try:
            ups = self._nm_client().uploads()
        except NomadError as e:
            self._nm_msg(self.w_nm_status1, str(e), ok=False)
            self._nm_msg(self.w_nm_status6, str(e), ok=False)
            return
        self.w_nm_uploads.options = [(f"{n or '(no name)'} · {u}" + (" · published" if p else ""), u)
                                     for u, n, p in ups]
        if ups and not self.w_nm_upload.value.strip():
            self.w_nm_upload.value = ups[0][0]
        self._nm_msg(self.w_nm_status1, f"{len(ups)} upload(s)")

    def _nm_up(self):
        self.w_nm_path.value = "/".join(self.w_nm_path.value.strip("/").split("/")[:-1])
        self._nm_browse()

    def _nm_browse(self):
        uid = self.w_nm_upload.value.strip()
        if not uid:
            self._nm_msg(self.w_nm_status1, "enter the upload ID (or choose one of 'my uploads')", ok=False)
            return
        try:
            items = self._nm_client().listdir(uid, self.w_nm_path.value.strip("/"))
        except NomadError as e:
            self._nm_msg(self.w_nm_status1, str(e), ok=False)
            return
        self._nm_items = items
        base = self.w_nm_path.value.strip("/")
        opts = []
        for name, is_file, size in sorted(items, key=lambda t: (t[1], t[0].lower())):
            p = f"{base}/{name}" if base else name
            if not is_file:
                opts.append((f"📁 {name}", ("dir", p)))
            elif name.lower().endswith(self._EXT):
                opts.append((f"📄 {name}   ({(size or 0) / 2**20:.1f} MB)", ("file", p)))
        self._nm_browsing = True
        self.w_nm_list.options = opts
        self.w_nm_list.value = None
        self._nm_browsing = False
        self._nm_msg(self.w_nm_status1, f"{len(opts)} folder(s)/supported file(s) in /{base}")

    def _nm_select(self, ch):
        v = ch["new"]
        if not v or getattr(self, "_nm_browsing", False):
            return
        if v[0] == "dir":
            self.w_nm_path.value = v[1]
            self._nm_browse()

    def _nm_cache(self, uid):
        d = os.path.abspath(os.path.join("nomad_downloads", re.sub(r"[^A-Za-z0-9_.-]", "_", uid)))
        os.makedirs(d, exist_ok=True)
        return d

    # ------------------------------------------------------------------ import
    def _nm_import(self, folder=False, block=True):
        uid = self.w_nm_upload.value.strip()
        sel = self.w_nm_list.value
        if not uid:
            self._nm_msg(self.w_nm_status1, "enter the upload ID", ok=False)
            return
        if not folder and (not sel or sel[0] != "file"):
            self._nm_msg(self.w_nm_status1, "select a file in the list", ok=False)
            return
        base = self.w_nm_path.value.strip("/")
        items = getattr(self, "_nm_items", [])
        if folder:
            names = [n for n, f, _ in items if f and n.lower().endswith(self._EXT)]
        else:
            name = sel[1].split("/")[-1]
            m = _IMG_IDX.match(name)
            names = [name]
            if m:                                   # indexed .img series: all frames with the same prefix
                names = [n for n, f, _ in items if f and (mm := _IMG_IDX.match(n)) and mm.group(1) == m.group(1)]
        paths = [f"{base}/{n}" if base else n for n in names]
        if not paths:
            self._nm_msg(self.w_nm_status1, "no supported files in this folder", ok=False)
            return

        def job():
            try:
                cl = self._nm_client()
                cache = self._nm_cache(uid)
                local = []
                for k, p in enumerate(paths):
                    self._nm_msg(self.w_nm_status1, f"downloading {k + 1}/{len(paths)}: {p} ...")
                    local.append(cl.download(uid, p, cache))
            except NomadError as e:
                self._nm_msg(self.w_nm_status1, str(e), ok=False)
                return
            self.nomad_ctx = {"url": cl.url, "upload_id": uid, "path": base, "files": paths}
            if folder:
                self._load_folder(os.path.dirname(local[0]))
            else:
                sel_local = local[[os.path.basename(x) for x in local].index(sel[1].split("/")[-1])]
                self._load_path(sel_local)
            self._nm_msg(self.w_nm_status1, f"loaded from NOMAD upload {uid}: {len(paths)} file(s) of /{base} "
                         f"(cached in {os.path.dirname(local[0])})"
                         + ("; results will be pushed to " + uid + "/" + self.w_nm_results.value.strip("/")
                            if self.w_nm_auto.value else ""))
        if block:
            job()
        else:
            threading.Thread(target=job, daemon=True).start()

    # ------------------------------------------------------------------ export
    def _register_output(self, path):
        """Remember a written file for the NOMAD export; push it right away if automatic push is on."""
        if not path:
            return
        path = os.path.abspath(path)
        if path not in self.generated_files:
            self.generated_files.append(path)
        self._nm_files_html()
        if self.w_nm_auto.value and self.w_nm_upload.value.strip():
            self._nm_push([path], block=getattr(self, "_nm_block", False), auto=True)

    def _nm_files_html(self):
        fs = [f for f in self.generated_files if os.path.isfile(f)]
        self.w_nm_files.value = ("<small>files of this session: " + (", ".join(
            f"<code>{os.path.basename(f)}</code> ({os.path.getsize(f) / 1e6:.1f} MB)" for f in fs) or "none yet")
            + "</small>")

    def _nm_push(self, files=None, block=True, auto=False):
        files = [f for f in (files or self.generated_files) if os.path.isfile(f)]
        uid = self.w_nm_upload.value.strip()
        folder = self.w_nm_results.value.strip().strip("/")
        if not files:
            self._nm_msg(self.w_nm_status6, "no files written yet (save in tab 6, 4b or convert in tab 1)", ok=False)
            return None
        out = {}

        def job():
            ws = (self.w_nm_status6, self.w_nm_status1) if auto else (self.w_nm_status6,)
            for w in ws:
                self._nm_msg(w, f"uploading {len(files)} file(s) to NOMAD{(' upload ' + uid) if uid else ''} ...")
            try:
                r = self._nm_client().upload(files, upload_id=uid, folder=folder, upload_name=self.w_nm_newname.value)
            except NomadError as e:
                for w in ws:
                    self._nm_msg(w, "NOMAD: " + str(e), ok=False)
                return
            out.update(r)
            if not uid and r.get("upload_id"):
                self.w_nm_upload.value = r["upload_id"]
            link = f" · <a href='{r['gui_url']}' target='_blank' rel='noopener'>open in NOMAD</a>" if r.get(
                "gui_url") else ""
            for w in ws:
                w.value = (f"<span style='color:green'>{r['files']} file(s) uploaded to {r['upload_id']}"
                           f"{'/' + r['folder'] if r['folder'] else ''}</span>{link}")
        if block:
            job()
            return out
        threading.Thread(target=job, daemon=True).start()
        return None
