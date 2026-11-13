# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""NOMAD plugin: headless pipeline, the action (without Temporal / NOMAD services), the dashboard page
(against the mock NOMAD API) and the dashboard server (Solara mounted in FastAPI)."""
import json
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import inlineholo as ih  # noqa: E402
from inlineholo.pipeline import params_from_metadata, run  # noqa: E402

from test_nomad import TOKEN, mock  # noqa: E402,F401  (fixture)


def _series(tmp_path, n=4, size=64):
    p = ih.simulate.hr_parameters(n_images=n, start=-30, step=20)
    I, _ = ih.simulate.simulate_focal_series(ih.simulate.hr_object(size, 0.05), p, dose_per_image=None,
                                             n_defocus=1, n_tilt=1)
    return I, p


def test_pipeline_nxem_and_folder(tmp_path):
    I, p = _series(tmp_path)
    f = tmp_path / "s.nxs"
    ih.nxem.write_focal_series(f, I, p, is_simulation=True)
    msgs = []
    r = run(str(f), str(tmp_path / "out"), method="tie_full", callback=lambda *a: msgs.append(a))
    assert os.path.isfile(r["files"][0]) and r["n_images"] == 4 and msgs[-1][0] == "write"
    st, p2, _, recs = ih.nxem.read_focal_series(r["files"][0])
    assert st.shape == I.shape and np.allclose(p2.defoci_nm, p.defoci_nm)
    # folder of single frames named by defocus; parameters given as overrides
    d = tmp_path / "frames"
    d.mkdir()
    for fr, z in zip(I, p.defoci_nm):
        np.save(d / f"F{z:+.0f}nm.npy", fr)
    r = run(str(d), str(tmp_path / "o2"), method="frwr", n_iter=3,       # defoci from the names, the rest given
            overrides=dict(voltage_kV=p.voltage_V / 1e3, pixel_size_nm=p.pixel_size_nm,
                           cs_mm=p.aberrations.get("c_3_0", 0) / 1e6))
    assert r["R_final"] is not None and os.path.basename(r["files"][0]) == "frames_frwr.nxs"


def test_params_overrides():
    p = params_from_metadata({"voltage_V": 200e3, "pixel_size_nm": 0.1}, None, 3,
                             dict(defocus_start_nm=-10, defocus_step_nm=10, cs_mm=1.2, convergence_mrad=0.2))
    assert np.allclose(p.defoci_nm, [-10, 0, 10]) and p.aberrations["c_3_0"] == 1.2e6
    assert np.isclose(p.convergence_rad, 2e-4) and p.voltage_V == 200e3
    with pytest.raises(ValueError, match="defocus"):
        params_from_metadata({}, None, 3, {})


def test_action_run_in_upload(tmp_path):
    pytest.importorskip("temporalio")
    pytest.importorskip("nomad.config")
    from inlineholo_nomad.workflows.activities import input_files, run_in_upload
    from inlineholo_nomad.workflows.models import ReconstructFocalSeriesInput
    from inlineholo.frwr3_io import write_img
    I, p = _series(tmp_path)
    up = tmp_path / "upload"                         # stands for the raw files of a NOMAD upload
    (up / "raw").mkdir(parents=True)
    ih.nxem.write_focal_series(up / "raw" / "series.nxs", I, p, is_simulation=True)
    (up / "img").mkdir()
    for i, fr in enumerate(I):
        write_img(str(up / "img" / f"image_{i}.img"), fr.astype(np.float32), dx=0.5, dy=0.5, t=10 * p.defoci_nm[i])
    (up / "img" / "notes.txt").write_text("x")
    added = []

    def is_file(q):
        return (up / q).is_file()

    def listdir(q):
        return [(e.name, e.is_file()) for e in sorted((up / q).iterdir())]

    def copy_out(q, local):
        shutil.copy(up / q, local)

    def add_files(files, folder):
        for f in files:
            (up / folder).mkdir(parents=True, exist_ok=True)
            shutil.copy(f, up / folder / os.path.basename(f))
            added.append(f"{folder}/{os.path.basename(f)}")

    fs, is_folder = input_files("img/image_2.img", is_file, listdir)
    assert sorted(fs) == [f"img/image_{i}.img" for i in range(4)] and not is_folder
    data = ReconstructFocalSeriesInput(upload_id="U1", user_id="me", path="raw/series.nxs", method="tie_full")
    res = run_in_upload(data, is_file=is_file, listdir=listdir, copy_out=copy_out, add_files=add_files,
                        tmp_root=str(tmp_path))
    assert res.files == ["inlineholo_results/series_tie_full.nxs"] == added and res.n_images == 4
    assert ih.nxem.validate(str(up / res.files[0]))[0] in (True, None)
    data = ReconstructFocalSeriesInput(upload_id="U1", user_id="me", path="img/image_0.img", method="frwr", n_iter=3,
                                       output_folder="res", output_name="frwr_from_img")
    res = run_in_upload(data, is_file=is_file, listdir=listdir, copy_out=copy_out, add_files=add_files)
    assert res.files == ["res/frwr_from_img.nxs"] and res.R_final is not None
    from temporalio.exceptions import ApplicationError
    with pytest.raises(ApplicationError):
        run_in_upload(ReconstructFocalSeriesInput(upload_id="U1", path="../etc"), is_file=is_file, listdir=listdir,
                      copy_out=copy_out, add_files=add_files)


def test_dashboard_page_session(mock, tmp_path, monkeypatch):
    pytest.importorskip("solara")
    pytest.importorskip("nomad.config")
    import matplotlib
    matplotlib.use("Agg")
    from inlineholo_nomad.dashboards.page import api_candidates, make_app, nomad_token
    assert nomad_token({"Authorization": "Bearer%20abc"}) == "abc"
    c = api_candidates({"host": ["oasis.example.org"], "x-forwarded-proto": ["https"]},
                       "/nomad-oasis/dashboards/inlineholo")
    assert "https://oasis.example.org/nomad-oasis/api/v1" in c
    started = []
    mock.extra[("GET", "users/me")] = lambda q, b: (200, {"user_id": "u1", "name": "Ada Lovelace"})

    def start(q, b):
        started.append(json.loads(b))
        return 200, {"action_instance_id": "inst-1"}

    mock.extra[("POST", "actions/inlineholo_nomad.actions:inlineholo_reconstruction/start")] = start
    mock.extra[("GET", "actions/inst-1/status")] = lambda q, b: (200, {"status": "COMPLETED"})
    mock.extra[("GET", "actions/inst-1/result")] = lambda q, b: (200, {"message": "written res/x.nxs"})
    monkeypatch.setenv("INLINEHOLO_NOMAD_API", mock.url)
    monkeypatch.chdir(tmp_path)
    a, s = make_app({"Authorization": "Bearer " + TOKEN}, {}, "/nomad-oasis/dashboards/inlineholo",
                    work_dir=str(tmp_path / "w"))
    assert s.connect() and "Ada Lovelace" in s.w_banner.value
    assert a.w_nm_url.value == "custom" and a.w_nm_custom.value == mock.url and a.w_nm_upload.value == "U1"
    # server mode: no access to the server's files
    a._load_path("/etc/hostname")
    assert "not possible" in a.w_load_status.value and a._find_local("x", 1) is None
    assert s.panel in a.tabs.children[4].children
    s.w_ac_path.value = "raw/series.nxs"
    s.w_ac_method.value = "tie"
    s._start()
    assert started and started[0]["data"]["upload_id"] == "U1" and started[0]["data"]["method"] == "tie"
    assert "COMPLETED" in s.w_ac_status.value and "written res/x.nxs" in s.w_ac_status.value
    # batch with approval of the alignment
    bstarted, signals = [], []

    def bstart(q, b):
        bstarted.append(json.loads(b))
        return 200, {"action_instance_id": "inst-2"}

    mock.extra[("POST", "actions/inlineholo_nomad.actions:inlineholo_batch/start")] = bstart
    mock.extra[("GET", "actions/inst-2/status")] = lambda q, b: (200, {"status": "RUNNING"})
    mock.extra[("GET", "actions/inst-2")] = lambda q, b: (200, {"signal_input_requests": [
        {"signal_fn_name": "approve_alignment", "title": "Approve the alignment", "description": "2 of 2 aligned",
         "content": "| # | series |\n|---|---|\n| 0 | a |\n| 1 | b |\n\nApprove all."}]})
    mock.extra[("POST", "actions/inst-2/signal-input")] = lambda q, b: (signals.append(json.loads(b)) or 200,
                                                                         {"status": "signal_input_submitted"})
    s.w_bt_paths.value = "raw/a.nxs\nraw/b.nxs"
    s._bt_start()
    d = bstarted[0]["data"]
    assert d["paths"] == ["raw/a.nxs", "raw/b.nxs"] and d["approve_alignment"] and d["reconstruct_on_gpu"]
    assert "defocus_start_nm" not in d and s.pending == "inst-2"
    assert "<td style='padding:1px 8px'>b</td>" in s.w_bt_signal.value and s.w_bt_ok_all.layout.display == "flex"
    s.w_bt_sel.value = "1"
    s._bt_approve(False)
    assert signals == [{"signal_fn_name": "approve_alignment",
                        "data": {"approve_all": False, "approved": [1],
                                 "comment": "approved in the inline-holography dashboard"}}]
    # files written by the app go to the session folder
    assert a._out_dir().startswith(str(tmp_path / "w"))


def test_dashboard_server_mounted(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("solara")
    pytest.importorskip("nomad.config")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from inlineholo_nomad.dashboards.server import create_app
    outer = FastAPI()                                  # as NOMAD mounts it
    outer.mount("/nomad-oasis/dashboards/inlineholo", create_app())
    cl = TestClient(outer)
    assert cl.get("/nomad-oasis/dashboards/inlineholo/healthz").json()["status"] == "ok"
    r = cl.get("/nomad-oasis/dashboards/inlineholo/")
    assert r.status_code == 200 and "solara" in r.text.lower()
    assert "/nomad-oasis/dashboards/inlineholo/static" in r.text
