# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Solara page of the NOMAD dashboard: the inline-holography app plus the NOMAD session.

* The NOMAD login of the user (the ``Authorization`` cookie the NOMAD GUI sets for its API path) is
  passed to the NOMAD import/export of tabs 1 and 6: "this NOMAD" is preselected and no API token is
  needed (the token of the GUI expires after a while; reload the dashboard or paste an app token).
* Tab 5 gets "run on a NOMAD worker": the reconstruction is started as the NOMAD action
  ``inlineholo_nomad.actions:inlineholo_reconstruction`` on a file or folder of an upload, and the
  NXem result is added to that upload.
* The app runs in server mode: users cannot read or write files of the server; every session has its
  own working folder (deleted when the session ends).
"""
import json
import os
import shutil
import tempfile
import threading
import urllib.parse

import ipywidgets as W
import solara

from inlineholo.app import App
from inlineholo.nomad import NomadClient, NomadError

ACTION_ID = "inlineholo_nomad.actions:inlineholo_reconstruction"
ACTION_METHODS = ["frwr", "gpfrwr", "frwr_gf", "frwr_gd", "gpfrwr_lsq", "mal", "ml_poisson", "iwfr", "gs", "tie",
                  "tie_full", "tie_poly", "tie_fp", "mftie", "gftie", "pam", "schiske"]


def nomad_token(cookies):
    """Access token of the NOMAD GUI from its ``Authorization`` cookie ('Bearer <token>', URL-encoded)."""
    raw = (cookies or {}).get("Authorization") or ""
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    tok = urllib.parse.unquote(str(raw)).strip()
    return tok[7:].strip() if tok.lower().startswith("bearer ") else tok


def _h(headers, key):
    v = (headers or {}).get(key) or (headers or {}).get(key.lower())
    return (v[0] if isinstance(v, (list, tuple)) and v else v) or ""


def api_candidates(headers, root_path):
    """API URLs of this NOMAD to try: INLINEHOLO_NOMAD_API, the URL the browser used, the server config."""
    out = []
    env = os.environ.get("INLINEHOLO_NOMAD_API")
    if env:
        out.append(env.rstrip("/"))
    base = (root_path or "").split("/dashboards/")[0].rstrip("/")
    host = _h(headers, "x-forwarded-host") or _h(headers, "host")
    proto = _h(headers, "x-forwarded-proto") or "http"
    if host:
        out.append(f"{proto.split(',')[0].strip()}://{host.split(',')[0].strip()}{base}/api/v1")
    try:
        from nomad.config import config
        out.append(config.api_url(ssl=False, api_host="localhost").rstrip("/") + "/v1")
        out.append(config.api_url().rstrip("/") + "/v1")
    except Exception:  # noqa: BLE001
        pass
    seen, res = set(), []
    for u in out:
        if u not in seen:
            seen.add(u)
            res.append(u)
    return res


class NomadSession:
    """Connects an :class:`App` to the NOMAD it is served from."""

    def __init__(self, app: App, token: str, candidates):
        self.app, self.token, self.candidates = app, token, list(candidates)
        self.url = None
        self.user = None
        self.instances = []
        self.w_banner = W.HTML()
        self._build_action_panel()

    # ------------------------------------------------------------------ connection
    def connect(self):
        if not self.token:
            self.w_banner.value = ("<small>Not signed in to NOMAD (open the dashboard from the NOMAD GUI after "
                                   "logging in), or use an API token in tab 1.</small>")
            return False
        err = ""
        for url in self.candidates:
            try:
                me = NomadClient(url, self.token, timeout=10)._req("GET", "/users/me")
            except NomadError as e:
                err = str(e)
                continue
            self.url, self.user = url, me
            break
        if self.url is None:
            self.w_banner.value = f"<small style='color:#b26a00'>NOMAD API not reachable from the app: {err}</small>"
            return False
        a = self.app
        a.w_nm_url.value = "custom"
        a.w_nm_custom.value = self.url
        a.w_nm_token.value = self.token
        name = self.user.get("name") or self.user.get("username") or self.user.get("user_id", "")
        self.w_banner.value = (f"<small>Signed in to NOMAD as <b>{name}</b> ({self.url}). Import a focal series "
                               f"from one of your uploads in tab 1; results go back in tab 6, or run the "
                               f"reconstruction on a NOMAD worker in tab 5.</small>")
        try:
            a._nm_list_uploads()
        except Exception:  # noqa: BLE001
            pass
        return True

    def client(self):
        if self.url is None:
            raise NomadError("not connected to NOMAD")
        tok = self.app.w_nm_token.value.strip() or self.token
        return NomadClient(self.url, tok)

    # ------------------------------------------------------------------ action
    def _build_action_panel(self):
        a = self.app
        lay = W.Layout(width="440px")
        st = {"description_width": "110px"}
        self.w_ac_upload = W.Text(description="Upload ID", style=st, layout=lay)
        W.link((a.w_nm_upload, "value"), (self.w_ac_upload, "value"))
        self.w_ac_path = W.Text(description="file / folder", placeholder="path in the upload, e.g. raw/series.dm3",
                                style=st, layout=W.Layout(width="560px"))
        self.w_ac_take = W.Button(description="take selection of tab 1", icon="hand-o-left",
                                  layout=W.Layout(width="210px"))
        meth = a.w_method.value if a.w_method.value in ACTION_METHODS else "frwr"
        self.w_ac_method = W.Dropdown(options=ACTION_METHODS, value=meth, description="method", style=st,
                                      layout=W.Layout(width="300px"))
        self.w_ac_iter = W.IntText(value=a.w_iter.value, description="iterations", style=st,
                                   layout=W.Layout(width="220px"))
        self.w_ac_align = W.Checkbox(value=False, description="FRWR alignment first (raw series)", indent=False,
                                     layout=W.Layout(width="auto"))
        self.w_ac_folder = W.Text(description="result folder", style=st, layout=lay)
        W.link((a.w_nm_results, "value"), (self.w_ac_folder, "value"))
        self.w_ac_params = W.Checkbox(value=True, indent=False, layout=W.Layout(width="auto"),
                                      description="send voltage, pixel size, Cs and defocus series of tab 2 (used "
                                                  "only where the file has none)")
        self.w_ac_start = W.Button(description="start on NOMAD worker", icon="play", button_style="primary",
                                   layout=W.Layout(width="220px"))
        self.w_ac_refresh = W.Button(description="refresh status", icon="refresh", layout=W.Layout(width="150px"))
        self.w_ac_status = W.HTML()
        self.w_ac_take.on_click(lambda _: self._take())
        self.w_ac_start.on_click(lambda _: self._start())
        self.w_ac_refresh.on_click(lambda _: self._refresh())
        box = W.VBox([W.HTML("Runs the reconstruction as the NOMAD action <i>Inline holography: reconstruct a focal "
                             "series</i> on a NOMAD worker, directly on a file or folder of a NOMAD upload (no "
                             "download to this app). The NXem result (amplitude, phase, imaging parameters) is "
                             "added to the upload and processed by NOMAD. The action can also be started from "
                             "the Actions page of NOMAD."),
                      W.HBox([self.w_ac_upload]), W.HBox([self.w_ac_path, self.w_ac_take]),
                      W.HBox([self.w_ac_method, self.w_ac_iter, self.w_ac_align]), self.w_ac_folder,
                      self.w_ac_params, W.HBox([self.w_ac_start, self.w_ac_refresh]), self.w_ac_status])
        self.panel = W.Accordion(children=[box], titles=("run on a NOMAD worker (NOMAD action)",))
        self.panel.selected_index = None

    def _take(self):
        a = self.app
        sel = a.w_nm_list.value
        base = a.w_nm_path.value.strip("/")
        if sel and sel[0] == "file":
            self.w_ac_path.value = sel[1]
        else:
            self.w_ac_path.value = base
        ctx = getattr(a, "nomad_ctx", None)
        if not self.w_ac_path.value and ctx:
            self.w_ac_path.value = (ctx.get("files") or [ctx.get("path", "")])[0]

    def action_data(self):
        a = self.app
        d = dict(upload_id=self.w_ac_upload.value.strip(), path=self.w_ac_path.value.strip().strip("/"),
                 method=self.w_ac_method.value, n_iter=int(self.w_ac_iter.value), align=bool(self.w_ac_align.value),
                 output_folder=self.w_ac_folder.value.strip().strip("/") or "inlineholo_results")
        if self.w_ac_params.value and a.raw is not None:
            p = a.params(all_images=True)
            d.update(voltage_kV=float(p.voltage_V / 1e3), pixel_size_nm=float(a.w_pix.value),
                     cs_mm=float(p.aberrations.get("c_3_0", 0.0) / 1e6),
                     focal_spread_nm=float(p.focal_spread_nm), convergence_mrad=float(p.convergence_rad * 1e3))
            df = [float(v) for v in p.defoci_nm]
            if len(df) > 1 and all(abs((df[k + 1] - df[k]) - (df[1] - df[0])) < 1e-6 for k in range(len(df) - 1)):
                d.update(defocus_start_nm=df[0], defocus_step_nm=df[1] - df[0])
        return d

    def _start(self):
        d = self.action_data()
        if not d["upload_id"] or not d["path"]:
            self.app._status(self.w_ac_status, "enter the upload ID and the file or folder", ok=False)
            return
        try:
            body = json.dumps({"data": d}).encode()
            r = self.client()._req("POST", f"/actions/{urllib.parse.quote(ACTION_ID, safe='')}/start", data=body,
                                   ctype="application/json")
        except NomadError as e:
            self.app._status(self.w_ac_status, f"could not start the action: {e}", ok=False)
            return
        iid = r.get("action_instance_id", "")
        self.instances.append((iid, d))
        self.app._status(self.w_ac_status, f"started: {iid}")
        self._refresh()

    def _refresh(self):
        rows = []
        for iid, d in self.instances[-10:]:
            try:
                cl = self.client()
                s = cl._req("GET", f"/actions/{urllib.parse.quote(iid, safe='')}/status").get("status", "?")
                extra = ""
                if s == "COMPLETED":
                    res = cl._req("GET", f"/actions/{urllib.parse.quote(iid, safe='')}/result")
                    res = res.get("result", res) if isinstance(res, dict) else res
                    extra = (res or {}).get("message", "") if isinstance(res, dict) else str(res)
            except NomadError as e:
                s, extra = "?", str(e)
            gui = self.url.split("/api/")[0] + f"/gui/user/uploads/upload/id/{d['upload_id']}" if self.url else ""
            rows.append(f"<tr><td>{d['method']}</td><td>{d['upload_id']}/{d['path']}</td><td><b>{s}</b></td>"
                        f"<td>{extra}</td><td><a href='{gui}' target='_blank'>upload</a></td></tr>")
        self.w_ac_status.value = ("<table style='font-size:12px'><tr><th>method</th><th>input</th><th>status</th>"
                                  "<th></th><th></th></tr>" + "".join(rows) + "</table>") if rows else ""


def make_app(cookies=None, headers=None, root_path=None, work_dir=None):
    """App + NOMAD session for one browser session (also used by the tests)."""
    app = App(data_dir=os.environ.get("INLINEHOLO_DATA") or None, server_mode=True, work_dir=work_dir)
    sess = NomadSession(app, nomad_token(cookies), api_candidates(headers, root_path))
    tab5 = app.tabs.children[4]
    tab5.children = tuple(tab5.children) + (sess.panel,)
    app.ui = W.VBox([sess.w_banner, *app.ui.children])
    return app, sess


@solara.component
def Layout(children=[]):  # noqa: B006  (Solara convention) — plain layout, no Solara app bar/branding
    solara.Column(children=children, style={"padding": "4px 8px", "background": "transparent"})


@solara.component
def Page():
    cookies = solara.lab.cookies.value
    headers = solara.lab.headers.value
    from solara.server import settings

    def build():
        wd = tempfile.mkdtemp(prefix="inlineholo_session_")
        app, sess = make_app(cookies, headers, settings.main.root_path, work_dir=wd)
        threading.Thread(target=sess.connect, daemon=True).start()
        return app, sess, wd

    app, sess, wd = solara.use_memo(build, [])

    def cleanup():
        return lambda: shutil.rmtree(wd, ignore_errors=True)

    solara.use_effect(cleanup, [])
    solara.Title("Inline holography")
    solara.display(app.ui)
