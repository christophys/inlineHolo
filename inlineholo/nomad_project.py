# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Shared helpers of the NOMAD dashboards (same file in inlineHolo, fit4DSTEM and temIngest):

* :func:`upload_button` -- the chunked file transfer of ipyvuetify (``FileInput``) with a visible "Upload" label
  next to the paperclip;
* :class:`ProjectPicker` -- choose the NOMAD project (= upload) that the uploaded files and the results go to: one of
  the user's unpublished uploads, or a new one as long as the user's limit of unpublished uploads
  (``services.upload_limit`` of the NOMAD configuration, 10 by default) is not reached;
* :func:`put_files` -- files into a folder of an upload as one zip built on disk and streamed (any size).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import ipywidgets as W

NEW = "__new__"
NONE = "__none__"


class ProjectError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- upload button
def upload_button(accept="", multiple=True, label="Upload file(s)", width="360px"):
    """ipyvuetify FileInput (chunked transfer, ``.get_files()``) showing the paperclip together with the word
    *Upload* (the plain widget shows only the paperclip).  None if ipyvuetify is not available."""
    try:
        import traitlets
        from ipyvuetify.extra import FileInput
    except Exception:  # noqa: BLE001
        return None
    tpl = FileInput.template.default_value
    attrs = (f' label="{label}" prepend-icon="mdi-paperclip" variant="outlined" density="compact" hide-details'
             f' style="min-width:{width};max-width:{width}"')
    tpl = tpl.replace("<v-file-input", "<v-file-input" + attrs, 1)

    class LabeledFileInput(FileInput):
        template = traitlets.Unicode(tpl).tag(sync=True)

    return LabeledFileInput(multiple=multiple, accept=accept, show_progress=False)


# ---------------------------------------------------------------------------------------------- NOMAD API
def _req(url, token, method, path, params=None, data=None, timeout=60):
    q = ("?" + urllib.parse.urlencode(params, doseq=True)) if params else ""
    hdr = {"Accept": "application/json", "Authorization": "Bearer " + token}
    req = urllib.request.Request(url.rstrip("/") + path + q, data=data, headers=hdr, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode()).get("detail")
        except Exception:  # noqa: BLE001
            detail = ""
        if isinstance(detail, list):
            detail = "; ".join(str(d.get("msg", d)) for d in detail)
        detail = " ".join(str(detail or "").split())
        if e.code == 400 and "limit" in detail.lower():
            raise ProjectError("you have reached the maximum number of open (unpublished) projects in NOMAD: "
                               "publish or delete one, or choose an existing project") from None
        raise ProjectError(f"NOMAD: {detail or e.reason} ({e.code})") from None
    except urllib.error.URLError as e:
        raise ProjectError(f"NOMAD is not reachable ({e.reason})") from None


def upload_limit():
    """Maximum number of unpublished uploads per user of this NOMAD (None if not running inside NOMAD)."""
    try:
        from nomad.config import config
        return int(config.services.upload_limit)
    except Exception:  # noqa: BLE001
        return None


def list_projects(url, token):
    """(me, projects, n_open): projects = [dict(upload_id, name, created)] of the unpublished uploads the user may
    write to (newest first); n_open = number of unpublished uploads of which the user is the main author (they count
    for the limit)."""
    me = _req(url, token, "GET", "/users/me")
    uid = me.get("user_id")
    d = _req(url, token, "GET", "/uploads", {"is_published": "false", "page_size": 500,
                                              "order_by": "upload_create_time", "order": "desc"})
    projects, n_open = [], 0
    for u in d.get("data") or []:
        if u.get("published"):
            continue
        if u.get("main_author", uid) == uid:
            n_open += 1
        writers = set(u.get("writers") or []) | {u.get("main_author")} | set(u.get("coauthors") or [])
        if "writers" not in u or uid in writers:
            projects.append({"upload_id": u["upload_id"], "name": u.get("upload_name") or "",
                             "created": (u.get("upload_create_time") or "")[:16].replace("T", " ")})
    return me, projects, n_open


def create_project(url, token, name):
    r = _req(url, token, "POST", "/uploads", {"upload_name": name} if name else None, data=b"")
    uid = r.get("upload_id") or (r.get("data") or {}).get("upload_id")
    if not uid:
        raise ProjectError("NOMAD did not return the ID of the new project")
    return uid


def put_files(url, token, files, upload_id, folder="", progress=None, tmp_dir=None, timeout=3600, chunk=4 << 20):
    """Files ([(local path, name in the folder)] or paths) into ``folder`` of the upload: one zip (stored) built on
    disk and streamed with ``progress(sent, total)``.  NOMAD extracts and processes it."""
    items = [(f, os.path.basename(f)) if isinstance(f, str) else tuple(f) for f in files]
    items = [(p, a) for p, a in items if os.path.isfile(p)]
    if not items:
        raise ProjectError("no files")
    folder = "/".join(s for s in str(folder or "").replace("\\", "/").split("/") if s)
    if ".." in folder.split("/"):
        raise ProjectError("invalid folder")
    if tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
    fd, zpath = tempfile.mkstemp(suffix=".zip", dir=tmp_dir)
    os.close(fd)
    try:
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for p, a in items:
                a = a.replace(os.sep, "/").strip("/")
                if ".." in a.split("/"):
                    raise ProjectError(f"invalid name {a}")
                z.write(p, a)
        size = os.path.getsize(zpath)
        with open(zpath, "rb") as fo:
            class Body:
                sent = 0

                def read(self, n=-1):
                    b = fo.read(chunk if n is None or n < 0 else min(n, chunk))
                    Body.sent += len(b)
                    if progress:
                        progress(Body.sent, size)
                    return b

                def __len__(self):
                    return size
            name = "upload_%s.zip" % dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"/uploads/{upload_id}/raw/{urllib.parse.quote(folder)}"
            q = "?" + urllib.parse.urlencode({"file_name": name})
            req = urllib.request.Request(url.rstrip("/") + path + q, data=Body(), method="PUT", headers={
                "Accept": "application/json", "Authorization": "Bearer " + token,
                "Content-Type": "application/octet-stream", "Content-Length": str(size)})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    r.read()
            except urllib.error.HTTPError as e:
                raise ProjectError(f"NOMAD refused the files ({e.code}): "
                                   f"{e.read().decode(errors='replace')[:300]}") from None
            except urllib.error.URLError as e:
                raise ProjectError(f"NOMAD is not reachable ({e.reason})") from None
    finally:
        os.remove(zpath)
    return [f"{folder}/{a}" if folder else a for _p, a in items]


def gui_url(url, upload_id):
    return url.split("/api/v1")[0] + f"/gui/user/uploads/upload/id/{upload_id}"


# ---------------------------------------------------------------------------------------------- picker widget
class ProjectPicker:
    """Row of widgets: NOMAD project (new / one of the user's unpublished uploads / not stored), name of a new
    project, folder for the uploaded files.  ``connect(url, token)`` lists the projects; ``store(files)`` creates
    the project if needed and puts the files into it (returns dict(upload_id, folder, paths, url, gui_url)).

    ``limit``: maximum number of unpublished uploads (default: from the NOMAD configuration when running in NOMAD);
    when it is reached, "new project" is not offered.  ``allow_none``: offer "keep only in this session"."""

    def __init__(self, folder="raw", limit=None, allow_none=True, label="NOMAD project", on_change=None):
        self.url = self.token = None
        self.limit = upload_limit() if limit is None else limit
        self.allow_none = allow_none
        self.n_open = 0
        self.projects = []
        self.on_change = on_change
        self.w_sel = W.Dropdown(options=[], description=label, style={"description_width": "110px"},
                                layout=W.Layout(width="470px"))
        self.w_name = W.Text(placeholder="name of the new project", layout=W.Layout(width="280px"))
        self.w_folder = W.Text(value=folder, description="folder", style={"description_width": "50px"},
                               layout=W.Layout(width="230px"))
        self.w_refresh = W.Button(icon="refresh", tooltip="refresh the list of your projects",
                                  layout=W.Layout(width="40px"))
        self.w_refresh.on_click(lambda _: self.refresh())
        self.w_info = W.HTML()
        self.w_sel.observe(self._changed, names="value")
        self.box = W.VBox([W.HBox([self.w_sel, self.w_name, self.w_refresh]), W.HBox([self.w_folder, self.w_info])],
                          layout=W.Layout(display="none"))

    @property
    def connected(self):
        return bool(self.url and self.token)

    def connect(self, url, token):
        self.url, self.token = url, token
        self.box.layout.display = "flex"
        return self.refresh()

    def can_create(self):
        return self.limit is None or self.n_open < self.limit

    def refresh(self, select=None):
        if not self.connected:
            return False
        try:
            _me, self.projects, self.n_open = list_projects(self.url, self.token)
        except ProjectError as e:
            self.w_info.value = f"<small style='color:crimson'>{e}</small>"
            return False
        cur = select or self.w_sel.value
        opts = []
        if self.can_create():
            opts.append(("➕ new project", NEW))
        opts += [(f"{p['name'] or '(no name)'} · {p['upload_id']}" + (f" · {p['created']}" if p["created"] else ""),
                  p["upload_id"]) for p in self.projects]
        if self.allow_none:
            opts.append(("— keep only in this session —", NONE))
        vals = [v for _l, v in opts]
        self.w_sel.options = opts
        self.w_sel.value = cur if cur in vals else (NEW if NEW in vals else (vals[0] if vals else None))
        self._changed()
        return True

    def _changed(self, _ch=None):
        v = self.w_sel.value
        self.w_name.layout.display = "flex" if v == NEW else "none"
        lim = (f"{self.n_open} of {self.limit} open projects" if self.limit is not None
               else f"{self.n_open} open project(s)")
        if not self.can_create():
            msg = (f"<span style='color:#b26a00'>{lim}: the maximum is reached, so no new project can be created — "
                   f"choose an existing one (or publish / delete one in NOMAD)</span>")
        elif v == NEW:
            msg = f"a new project is created with the first upload ({lim})"
        elif v == NONE:
            msg = "files are kept only in this session (results can still be uploaded later)"
        else:
            msg = "the uploaded files and the results go into this project"
        self.w_info.value = f"<small>{msg}</small>"
        if self.on_change:
            self.on_change(self.selected())

    def selected(self):
        """ID of the chosen existing project (None for "new" and "keep only in this session")."""
        v = self.w_sel.value
        return None if v in (None, NONE, NEW) else v

    def store(self, files, progress=None, tmp_dir=None, default_name=""):
        """Create the project if "new" is chosen, then put the files into its folder.  None if "keep only in this
        session" is chosen."""
        if not self.connected or self.w_sel.value in (None, NONE):
            return None
        uid = self.w_sel.value
        if uid == NEW:
            if not self.can_create():
                raise ProjectError("the maximum number of open projects is reached: choose an existing project")
            name = self.w_name.value.strip() or default_name or time.strftime("data %Y-%m-%d %H:%M")
            uid = create_project(self.url, self.token, name)
            self.refresh(select=uid)
        paths = put_files(self.url, self.token, files, uid, self.w_folder.value.strip(), progress=progress,
                          tmp_dir=tmp_dir)
        return {"upload_id": uid, "folder": self.w_folder.value.strip().strip("/"), "paths": paths, "url": self.url,
                "gui_url": gui_url(self.url, uid)}
