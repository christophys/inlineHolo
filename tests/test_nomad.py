# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""NOMAD import / export against a local mock of the NOMAD API v1 (no internet access needed)."""
import io
import json
import os
import sys
import threading
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import inlineholo as ih  # noqa: E402
from inlineholo.nomad import NomadClient, NomadError  # noqa: E402

TOKEN = "secret-token"


class MockNomad:
    """uploads -> {path: bytes}; implements the endpoints used by inlineholo.nomad."""

    def __init__(self):
        self.uploads = {"U1": {}}
        self.names = {"U1": "focal series project"}
        self.published = set()
        self.requests = []
        self.extra = {}      # (method, path) -> f(query, body) -> (code, obj): further endpoints for other tests
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj=None, raw=None):
                body = raw if raw is not None else json.dumps(obj or {}).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _auth(self):
                return self.headers.get("Authorization") == "Bearer " + TOKEN

            def _route(self, method):
                u = urllib.parse.urlparse(self.path)
                q = dict(urllib.parse.parse_qsl(u.query))
                parts = urllib.parse.unquote(u.path).split("/")[3:]          # after /api/v1
                mock.requests.append((method, "/".join(parts), q))
                if not self._auth():
                    return self._send(401, {"detail": "not authorized"})
                key = (method, "/".join(parts))
                if key in mock.extra:
                    n = int(self.headers.get("Content-Length", 0) or 0)
                    code, obj = mock.extra[key](q, self.rfile.read(n) if n else b"")
                    return self._send(code, obj)
                if parts == ["uploads"] and method == "GET":
                    return self._send(200, {"data": [{"upload_id": k, "upload_name": mock.names.get(k, ""),
                                                      "published": k in mock.published} for k in mock.uploads]})
                if parts == ["uploads"] and method == "POST":
                    uid = "U%d" % (len(mock.uploads) + 1)
                    mock.uploads[uid] = {}
                    mock.names[uid] = q.get("upload_name", "")
                    self._extract(uid, "")
                    return self._send(200, {"upload_id": uid, "data": {"upload_id": uid}})
                if len(parts) >= 3 and parts[0] == "uploads":
                    uid, kind, path = parts[1], parts[2], "/".join(parts[3:]).strip("/")
                    if uid not in mock.uploads:
                        return self._send(404, {"detail": "no upload"})
                    files = mock.uploads[uid]
                    if kind == "rawdir" and method == "GET":
                        if path in files:
                            return self._send(200, {"path": path, "access": "unpublished",
                                                    "file_metadata": {"name": path.split("/")[-1]}})
                        pre = path + "/" if path else ""
                        content = {}
                        for f, b in files.items():
                            if f.startswith(pre):
                                rest = f[len(pre):].split("/")
                                if len(rest) == 1:
                                    content[rest[0]] = {"name": rest[0], "is_file": True, "size": len(b)}
                                else:
                                    content.setdefault(rest[0], {"name": rest[0], "is_file": False, "size": 0})
                        items = sorted(content.values(), key=lambda e: e["name"])
                        ps, pg = int(q.get("page_size", 10)), int(q.get("page", 1))
                        page = items[(pg - 1) * ps: pg * ps]
                        return self._send(200, {"path": path, "access": "unpublished",
                                                "directory_metadata": {"name": path, "content": page},
                                                "pagination": {"page_size": ps, "page": pg, "total": len(items)}})
                    if kind == "raw" and method == "GET":
                        if path not in files:
                            return self._send(404, {"detail": "no file"})
                        return self._send(200, raw=files[path])
                    if kind == "raw" and method == "PUT":
                        if uid in mock.published:
                            return self._send(403, {"detail": "published"})
                        self._extract(uid, path)
                        return self._send(200, {"upload_id": uid, "data": {"upload_id": uid}})
                return self._send(400, {"detail": "unknown"})

            def _extract(self, uid, folder):
                n = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(n) if n else b""
                if not body:                        # empty upload (POST /uploads without a file)
                    return
                z = zipfile.ZipFile(io.BytesIO(body))
                for it in z.infolist():
                    name = (folder + "/" + it.filename) if folder else it.filename
                    mock.uploads[uid][name.strip("/")] = z.read(it)

            def do_GET(self):
                self._route("GET")

            def do_PUT(self):
                self._route("PUT")

            def do_POST(self):
                self._route("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d/api/v1" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def mock(monkeypatch):
    for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    m = MockNomad()
    yield m
    m.server.shutdown()


def test_client_list_download_upload(mock, tmp_path):
    mock.uploads["U1"].update({f"series/F{v}nm.npy": b"x" * 10 for v in range(600)})   # > one rawdir page
    cl = NomadClient(mock.url, TOKEN)
    assert cl.uploads() == [("U1", "focal series project", False)]
    top = cl.listdir("U1")
    assert top == [("series", False, 0)]
    assert len(cl.listdir("U1", "series")) == 600
    p = cl.download("U1", "series/F5nm.npy", str(tmp_path))
    assert open(p, "rb").read() == b"x" * 10 and p.endswith(os.path.join("series", "F5nm.npy"))
    f = tmp_path / "result.nxs"
    f.write_bytes(b"nexus")
    r = cl.upload([str(f)], upload_id="U1", folder="results")
    assert r["upload_id"] == "U1" and mock.uploads["U1"]["results/result.nxs"] == b"nexus"
    assert r["gui_url"].endswith("/gui/user/uploads/upload/id/U1")
    r = cl.upload([str(f)], folder="session1", upload_name="new project")
    assert r["upload_id"] == "U2" and mock.uploads["U2"] == {"session1/result.nxs": b"nexus"}
    with pytest.raises(NomadError, match="token"):
        NomadClient(mock.url, "wrong").listdir("U1")
    mock.published.add("U1")
    with pytest.raises(NomadError, match="published"):
        cl.upload([str(f)], upload_id="U1")


def test_app_import_from_nomad_and_push_back(mock, tmp_path, monkeypatch):
    pytest.importorskip("pynxtools")
    import matplotlib
    matplotlib.use("Agg")
    from inlineholo.app import App
    from inlineholo.frwr3_io import write_img
    p = ih.simulate.hr_parameters(n_images=4, start=-30, step=20)
    I, _ = ih.simulate.simulate_focal_series(ih.simulate.hr_object(64, 0.05), p, dose_per_image=None,
                                             n_defocus=1, n_tilt=1)
    src = tmp_path / "src"
    src.mkdir()
    for i, fr in enumerate(I):
        write_img(str(src / f"image_{i}.img"), fr.astype(np.float32), dx=0.5, dy=0.5, t=10 * p.defoci_nm[i])
        mock.uploads["U1"][f"raw/stack1/image_{i}.img"] = (src / f"image_{i}.img").read_bytes()
    mock.uploads["U1"]["raw/stack1/notes.txt"] = b"not an image"
    monkeypatch.chdir(tmp_path)
    a = App()
    a.w_nm_url.value = "custom"
    a.w_nm_custom.value = mock.url
    a.w_nm_token.value = TOKEN
    a._nm_list_uploads()
    assert a.w_nm_upload.value == "U1"
    a._nm_browse()
    a.w_nm_list.value = [o[1] for o in a.w_nm_list.options if o[0].endswith("raw")][0]     # folder -> enter it
    a.w_nm_list.value = [o[1] for o in a.w_nm_list.options if "stack1" in o[0]][0]
    assert a.w_nm_path.value == "raw/stack1" and not any("notes" in o[0] for o in a.w_nm_list.options)
    a.w_nm_list.value = [o[1] for o in a.w_nm_list.options if "image_2.img" in o[0]][0]
    a.w_nm_auto.value = True
    a._nm_block = True
    a._nm_import(block=True)
    assert a.stack is not None and a.stack.shape == (4, 64, 64)                       # the whole .img series
    assert np.allclose(a.params().defoci_nm, p.defoci_nm) and a.nomad_ctx["upload_id"] == "U1"
    # saving in tab 6 pushes the file back into the same upload (results subfolder)
    a.w_placeholder.value = True
    a.w_fname.value = "result.nxs"
    a._save()
    folder = a.w_nm_results.value
    assert f"{folder}/result.nxs" in mock.uploads["U1"]
    assert "uploaded to U1" in a.w_nm_status6.value
    # manual upload of all generated files to a new upload
    a.w_nm_upload.value = ""
    a.w_nm_newname.value = "copy"
    r = a._nm_push(block=True)
    assert r["upload_id"] == "U2" and a.w_nm_upload.value == "U2" and mock.names["U2"] == "copy"
