# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Import from and export to NOMAD (production, test server or an Oasis) via its REST API v1.

Same conventions as the NOMAD upload of the TEM Alignment Twin (tem_twin3/nexus_export.py):
an API token (NOMAD user menu -> API tokens / app token, with permission to read and write
uploads) is sent as ``Authorization: Bearer <token>``; the *upload ID* identifies the NOMAD
project (upload); files are put into a *subfolder* of that upload; without an upload ID a new
upload is created.

Endpoints used (nomad-lab v1 API, ``nomad/app/v1/routers/uploads``):

* ``GET  /uploads``                              uploads of the user (``data``: upload_id, upload_name, published)
* ``GET  /uploads/{id}/rawdir/{path}``           folder listing (``directory_metadata.content``: name, is_file, size;
                                                 paginated with ``page``/``page_size``)
* ``GET  /uploads/{id}/raw/{path}``              download of one raw file
* ``PUT  /uploads/{id}/raw/{folder}?file_name=`` add files (a .zip is extracted into the folder)
* ``POST /uploads?file_name=&upload_name=``      new upload from a .zip

The token is never written to disk.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import zipfile

NOMAD_URLS = {"production": "https://nomad-lab.eu/prod/v1/api/v1",
              "test": "https://nomad-lab.eu/prod/v1/test/api/v1"}


class NomadError(RuntimeError):
    pass


class NomadClient:
    """Minimal NOMAD API v1 client (urllib only).  ``opener`` can replace urllib.request.urlopen."""

    def __init__(self, url="production", token="", opener=None, timeout=300):
        url = str(url or "production").strip()
        self.url = NOMAD_URLS.get(url, url).rstrip("/")
        self.token = str(token or "").strip()
        self.opener = opener or urllib.request.urlopen
        self.timeout = timeout

    # ------------------------------------------------------------------ http
    def _req(self, method, path, params=None, data=None, ctype=None, raw=False):
        q = ("?" + urllib.parse.urlencode(params)) if params else ""
        hdr = {"Accept": "application/json" if not raw else "*/*"}
        if self.token:
            hdr["Authorization"] = "Bearer " + self.token
        if ctype:
            hdr["Content-Type"] = ctype
        req = urllib.request.Request(self.url + path + q, data=data, headers=hdr, method=method)
        try:
            with self.opener(req, timeout=self.timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as ex:
            detail = ""
            try:
                detail = ex.read().decode(errors="replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            if ex.code in (401, 403):
                raise NomadError(f"NOMAD refused the request ({ex.code}): check the API token and its permissions "
                                 f"(uploads read/write) and whether the upload is already published. {detail}")
            if ex.code == 404:
                raise NomadError(f"not found on NOMAD (404): check the upload ID and the path. {detail}")
            raise NomadError(f"NOMAD request failed: {ex} {detail}")
        except urllib.error.URLError as ex:
            raise NomadError(f"NOMAD is not reachable ({ex.reason}); this computer needs internet access to "
                             f"{self.url}")
        if raw:
            return body
        return json.loads(body.decode() or "{}")

    @staticmethod
    def _qp(path):
        return urllib.parse.quote(str(path or "").strip("/"))

    # ------------------------------------------------------------------ reading
    def uploads(self, page_size=100):
        """[(upload_id, upload_name, published)] of the user (needs a token)."""
        if not self.token:
            raise NomadError("an API token is needed to list your uploads")
        r = self._req("GET", "/uploads", {"page_size": page_size, "order_by": "upload_create_time",
                                          "order": "desc"})
        return [(d.get("upload_id"), d.get("upload_name") or "", bool(d.get("published")))
                for d in r.get("data", [])]

    def listdir(self, upload_id, path=""):
        """[(name, is_file, size)] of a folder of the upload (all pages)."""
        out, page = [], 1
        while True:
            r = self._req("GET", f"/uploads/{self._qp(upload_id)}/rawdir/{self._qp(path)}",
                          {"page_size": 500, "page": page})
            dm = r.get("directory_metadata")
            if dm is None:
                if r.get("file_metadata") is not None:
                    raise NomadError(f"{path} is a file, not a folder")
                break
            items = dm.get("content", [])
            out += [(e.get("name"), bool(e.get("is_file")), e.get("size")) for e in items]
            pag = r.get("pagination") or {}
            if len(items) < 500 or not pag or page * 500 >= int(pag.get("total", 0) or 0):
                break
            page += 1
        return out

    def download(self, upload_id, path, dest_dir):
        """Download one raw file of the upload into dest_dir (keeps its relative path); returns the local path."""
        rel = str(path).strip("/")
        if ".." in rel.split("/"):
            raise NomadError("invalid path")
        body = self._req("GET", f"/uploads/{self._qp(upload_id)}/raw/{self._qp(rel)}", raw=True)
        local = os.path.join(dest_dir, *rel.split("/"))
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        with open(local, "wb") as f:
            f.write(body)
        return local

    # ------------------------------------------------------------------ writing
    def upload(self, paths, upload_id="", folder="", upload_name="", base_dir=None):
        """Put local files into an upload (zip, extracted by NOMAD into ``folder``).

        Without ``upload_id`` a new upload is created (named ``upload_name``).  Files keep their
        path relative to ``base_dir`` (default: only their names).  Returns dict(upload_id, folder,
        files, gui_url)."""
        if not self.token:
            raise NomadError("a NOMAD API token is needed for uploading")
        paths = [p for p in paths if os.path.isfile(p)]
        if not paths:
            raise NomadError("no files to upload")
        folder = str(folder or "").strip().strip("/")
        if ".." in folder.split("/"):
            raise NomadError("invalid subfolder")
        upload_id = str(upload_id or "").strip()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in paths:
                arc = os.path.relpath(p, base_dir).replace(os.sep, "/") if base_dir else os.path.basename(p)
                if not upload_id and folder:                    # a new upload: the folder inside the zip
                    arc = folder + "/" + arc
                z.write(p, arc)
        zipname = "inlineholo_%s.zip" % dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        ctype = "application/octet-stream"
        if upload_id:
            r = self._req("PUT", f"/uploads/{self._qp(upload_id)}/raw/{self._qp(folder)}", {"file_name": zipname},
                          data=buf.getvalue(), ctype=ctype)
        else:
            params = {"file_name": zipname}
            if upload_name:
                params["upload_name"] = str(upload_name)
            r = self._req("POST", "/uploads", params, data=buf.getvalue(), ctype=ctype)
        uid = r.get("upload_id") or (r.get("data") or {}).get("upload_id") or upload_id
        return {"upload_id": uid, "folder": folder, "files": len(paths), "gui_url": self.gui_url(uid)}

    def gui_url(self, upload_id):
        if not upload_id:
            return None
        return self.url.replace("/api/v1", "/gui/user/uploads/upload/id/" + upload_id)


def default_results_folder():
    return "inlineholo_results_" + dt.date.today().isoformat()
