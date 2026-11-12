# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Non-linear defocus series (FRWR Eq. 1), stoppable fits/alignment/reconstruction, file browser."""
import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import inlineholo as ih  # noqa: E402
from inlineholo import simulate as sim  # noqa: E402
from inlineholo.control import StopRequested  # noqa: E402


def _app():
    import matplotlib
    matplotlib.use("Agg")
    from inlineholo.app import App
    return App


def test_series_defoci_eq1():
    d = ih.ctffit.series_defoci(7, 10.0, 2.0, m=2.0)
    assert np.allclose(d, [-8, 2, 8, 10, 12, 18, 28])
    assert np.allclose(ih.ctffit.series_defoci(4, 0.0, 1.0, 1.0, n0=0), [0, 1, 2, 3])


def test_power_series_fit_recovers_eq1():
    k = np.arange(7) - 3
    true = -20.0 + 6.0 * np.sign(k) * np.abs(k) ** 1.5
    p = sim.hr_parameters(n_images=7, start=0, step=1, aberrations={"c_3_0": 1e4})
    p.defoci_nm = true
    gt = sim.hr_object(192, 0.04, crystal_radius_nm=0.1)
    I, _ = sim.simulate_focal_series(gt, p, dose_per_image=20000, n_defocus=1, n_tilt=1)
    # nominal series with wrong offset and step, correct m: fit offset/step
    p.defoci_nm = ih.ctffit.series_defoci(7, -16.0, 5.6, 1.5)
    r = ih.ctffit.fit_series_global(I, p, window_nm=8, fit_astig=False, model="power", m=1.5)
    assert abs(r["offset"] + 20) < 1.0 and abs(r["step"] - 6) < 0.3, r
    assert np.abs(r["defoci"] - true).max() < 2.0
    # m fitted as well (nominal series: m = 1.3, slightly wrong offset)
    p.defoci_nm = ih.ctffit.series_defoci(7, -18.0, 6.5, 1.3)
    r = ih.ctffit.fit_series_global(I, p, window_nm=4, fit_astig=False, model="power", m=1.3, fit_m=True)
    assert abs(r["m"] - 1.5) < 0.1 and np.abs(r["defoci"] - true).max() < 3.0, r


def test_fits_can_be_stopped():
    p = sim.hr_parameters(n_images=3, start=-40, step=20, aberrations={"c_3_0": 1e4})
    gt = sim.hr_object(128, 0.04, crystal_radius_nm=0.4)
    I, _ = sim.simulate_focal_series(gt, p, dose_per_image=5000, n_defocus=1, n_tilt=1)
    calls = []

    def stop_after(n):
        def prog(*a):
            calls.append(a)
            if len(calls) >= n:
                raise StopRequested()
        return prog

    r = ih.ctffit.fit_thon(I[0], 0.04, 300e3, (-60, -20), 1e4, progress=stop_after(5))
    assert r["stopped"] and np.isfinite(r["c_1_0"])
    calls.clear()
    r = ih.ctffit.fit_series_global(I, p, window_nm=5, progress=stop_after(20))
    assert r["stopped"] and len(r["defoci"]) == 3


def test_app_series_fill_q_sliders_and_threaded_stop(tmp_path, monkeypatch):
    App = _app()
    p = sim.hr_parameters(n_images=5, start=-40, step=10, aberrations={"c_3_0": 1e4})
    gt = sim.hr_object(128, 0.04, crystal_radius_nm=1.0)
    I, _ = sim.simulate_focal_series(gt, p, dose_per_image=2000, n_defocus=1, n_tilt=1)
    f = tmp_path / "s.nxs"
    ih.nxem.write_focal_series(f, I, p, is_simulation=True)
    monkeypatch.chdir(tmp_path)
    a = App(data_dir=str(tmp_path)).load(str(f))
    # auto refresh of the Thon view with the central frame
    assert a.w_idx.value == 2 and a.w_thon_out.outputs and "image/png" in a.w_thon_out.outputs[0]["data"]
    # absolute q sliders: maximum = Nyquist frequency (1/nm)
    qn = 0.5 / 0.04
    assert np.isclose(a.w_qmax.max, qn) and np.isclose(a.w_qmax.value, 0.8 * qn, atol=0.05)
    a.w_qmax.value = 3.0                       # live update from the cached spectrum
    assert a._thon_cache is not None
    # FRWR Eq. (1) fill
    a.w_series_start.value, a.w_series_step.value, a.w_series_m.value = 5.0, 3.0, 2.0
    a._fill_linear()
    assert np.allclose(a.params().defoci_nm, [-7, 2, 5, 8, 17])
    # series fit with the power model, then apply -> table and fill widgets updated
    a.w_series_model.value = "power"
    a._fit_global()
    assert a._global_fit["model"] == "power"
    a._apply_fits()
    assert np.isclose(a.w_series_m.value, 2.0)
    # threaded reconstruction stopped early keeps the current wave
    a.w_method.value = "gs"
    a.w_iter.value = 100000
    a._run(block=False)
    for _ in range(600):
        if a.w_prog.value >= 3:
            break
        threading.Event().wait(0.05)
    a._request_stop()
    for _ in range(600):
        if not a._busy:
            break
        threading.Event().wait(0.05)
    assert not a._busy and a.rec is not None
    assert a.rec.wave.shape == I.shape[1:] and 0 < a.rec.options["stopped_after_iterations"] < 100000
    assert "stopped" in a.w_rec_status.value


def test_app_frwr3_reconstruction_and_alignment_stop(tmp_path, monkeypatch):
    App = _app()
    p = sim.hr_parameters(n_images=4, start=-40, step=10, aberrations={"c_3_0": 1e4})
    gt = sim.hr_object(256, 0.04, crystal_radius_nm=2.0)
    I, _ = sim.simulate_focal_series(gt, p, dose_per_image=2000, n_defocus=1, n_tilt=1)
    f = tmp_path / "s.nxs"
    ih.nxem.write_focal_series(f, I, p, is_simulation=True)
    monkeypatch.chdir(tmp_path)
    a = App().load(str(f))
    before = a.stack.copy()
    a.w_fa_bin.value = 1
    # emulate a press of 'stop' right after the start (the first loop then raises StopRequested)
    a._start_job = lambda fn, *args, **kw: (a._stop.set(), fn())
    a._frwr3_align()
    assert np.array_equal(a.stack, before) and "unchanged" in a.w_fa_status.value
    a.w_method.value = "frwr"
    a.w_frwr3_yaml.value = "fit defocus: 0\nmax shift: 0\nintermediate loops: 2\npadding: 0.25"
    a._run()
    assert a.rec is not None and a.rec.options.get("stopped_after_loops") is not None
    assert "stopped" in a.w_rec_status.value


def test_file_browser_and_upload_from_same_computer(tmp_path, monkeypatch):
    App = _app()
    p = sim.hr_parameters(n_images=3, start=-20, step=10)
    I, _ = sim.simulate_focal_series(sim.hr_object(64, 0.04), p, dose_per_image=1000, n_defocus=1, n_tilt=1)
    sub = tmp_path / "sub"
    sub.mkdir()
    f = sub / "series.nxs"
    ih.nxem.write_focal_series(f, I, p, is_simulation=True)
    (sub / "notes.txt").write_text("x")
    monkeypatch.chdir(tmp_path)
    a = App(data_dir=str(tmp_path))
    a.w_br_dir.value = str(sub)
    labels = [o[0] for o in a.w_br_list.options]
    assert any("series.nxs" in l for l in labels) and not any("notes" in l for l in labels)
    a.w_br_list.value = [o[1] for o in a.w_br_list.options if "series.nxs" in o[0]][0]
    assert a.w_path.value == str(f)
    a.w_br_list.value = a.w_br_list.options[0][1]           # '..' -> parent folder
    assert a.w_br_dir.value == str(tmp_path)
    # upload of a file that exists on this computer -> read from the path
    a.w_br_dir.value = str(sub)
    content = f.read_bytes()
    assert a._find_local("series.nxs", len(content)) == str(f)
    a._on_upload({"new": [{"name": "series.nxs", "content": memoryview(content)}]})
    assert a.stack is not None and "found on this computer" in a.w_load_status.value


def test_frwr3_planned_work_and_start_with_all_planes():
    from inlineholo import frwr3
    p = sim.hr_parameters(n_images=7, start=-30, step=10, aberrations={"c_3_0": 1e4})
    gt = sim.hr_object(96, 0.04, crystal_radius_nm=1.0)
    I, _ = sim.simulate_focal_series(gt, p, dose_per_image=None, n_defocus=1, n_tilt=1)
    I = I / I.mean()
    base = {"fit defocus": 0, "max shift": 0, "final loops": 6, "intermediate loops": 2, "padding": 0.25,
            "print level": 0}
    for extra in ({}, {"start with all planes": 1}):
        eng = frwr3.FRWR3(I, p, dict(base, **extra))
        loops, work = eng.planned_work()
        seen = []
        eng.callback = lambda label, j, R, e: seen.append(label)
        res = eng.run()
        assert len(seen) == len(res.rvalues) == loops and eng.work_done == work, (extra, loops, work, eng.work_done)
        if extra:          # all 7 frames from the first loop
            assert all(lab in ("7-Loop", "Final Loop", "Residue Loop") for lab in seen)
    # a start wave (ROI size) is accepted with all planes from the start
    res = frwr3.run_frwr3(I, p, dict(base, **{"start with all planes": 1}), init_wave=gt)
    assert res.wave.shape == gt.shape and np.isfinite(res.rvalues[-1][2])


def test_initial_wave_geometry():
    """FRWR alignment wave (binned, padded) -> full resolution in the aligned-stack geometry."""
    from types import SimpleNamespace
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(1)
    b, N = 2, 160
    full = np.exp(1j * gaussian_filter(rng.standard_normal((N, N)), 4, mode="wrap") * 5)   # periodic, smooth
    # binned wave sampled at the centres of the 2x2 blocks: shift by -0.5 px, then decimate
    lo = np.fft.fft2(full)
    k = np.fft.fftfreq(N)
    m = (np.abs(k)[:, None] < 0.25) & (np.abs(k)[None, :] < 0.25)
    band = np.fft.ifft2(lo * m)
    wb = ih.align.fourier_shift(band, (-0.5, -0.5))[::2, ::2]                                # (80, 80)
    py = px = 6
    fy, fx = 10, 12                                       # ROI top-left (binned) in the reference frame
    sy, sx = 50, 44
    wp = wb[fy - py:fy + sy + py, fx - px:fx + sx + px]
    res = SimpleNamespace(wave_padded=np.pad(wp, 0), pad_offset=(py, px), wave=wp[py:py + sy, px:px + sx],
                          reference_plane=0, frame_offsets=np.array([[fx, fy]], float))
    t, l = 14, 18
    out = ih.align.initial_wave(res, b, (t, l), (120, 110))
    bt, rt, bb, rb = out["wave_box"]
    assert (bt, rt) == (fy * b - t, fx * b - l)
    ref = band[t + bt:t + bb, l + rt:l + rb]
    err = np.abs(out["wave"] - ref)[6:-6, 6:-6]           # away from the edges of the (non-periodic) cut-out
    assert err.max() < 0.05 * np.abs(ref).max(), err.max()
    assert out["wave_start"].shape == (120, 110)


def test_app_alignment_initial_reconstruction_save_and_refine(tmp_path, monkeypatch):
    pytest.importorskip("pynxtools")
    App = _app()
    w = sim.mr_object(n=192, pixel=1.0, seed=3, film_nm=2.0)
    p = sim.mr_parameters(pixel_size_nm=1.0, defoci_nm=np.array([-3000., -1500., -500., 0., 500., 1500., 3000.]))
    I, _ = sim.simulate_focal_series(w, p, None, n_defocus=1, n_tilt=1)
    f = tmp_path / "m.nxs"
    ih.nxem.write_focal_series(f, I + 1e-3, p, is_simulation=True)
    monkeypatch.chdir(tmp_path)
    a = App().load(str(f))
    assert a.tabs.titles[3] == "4 Align/Pre-reconstruct" and a.tabs.titles[4] == "5 Reconstruct/Refine"
    a.w_fa_bin.value, a.w_fa_shift.value, a.w_fa_loops.value = 2, 8, 6
    a._frwr3_align()
    assert a.init_rec is not None and a.init_wave_start.shape == a.stack.shape[1:]
    assert a.w_fa_wave_out.outputs and np.isclose(a.w_fa_prog.value, 1.0)
    a.w_placeholder.value = True
    path = a._save_alignment()
    assert path and ih.nxem.validate(path)[0]
    st, pp, _, recs = ih.nxem.read_focal_series(path)
    assert st.shape == a.stack.shape and "FRWR initial reconstruction" in recs
    assert any(m == "frwr_initial" for m, _, _ in a.recs)          # also part of the session export
    a._use_init_rec()
    assert a.w_init.value == "frwr_align" and a.w_allframes.value
    a.w_method.value = "frwr"
    a.w_iter.value = 5
    a._run()
    assert a.rec is not None and a.rec.extra["frwr3"].config.get("start with all planes") in (1, "1", [1])
    labels = [r[0] for r in a.rec.extra["frwr3"].rvalues]
    assert "2-loop" not in labels and "3-loop" not in labels
    a.stack = a.stack[:, 2:, :]                                     # stack changed -> clear message
    a._run()
    assert "does not match" in a.w_rec_status.value


def test_schedule_eta_predicts_growing_stages():
    """Loop times as in a real FRWR alignment: expensive alignment loops, per-image cost growing."""
    from inlineholo.control import ScheduleETA
    sched = [(("2-loop", j), 2) for j in range(3)] + [(("3-loop", j), 3) for j in range(6)]
    stages = [3, 5, 7, 9, 11, 12, 13, 14, 15]
    cost = {0: 2.5, 1: 1.3, 2: 0.06, 3: 0.4}
    for n in stages:
        sched += [(("inter", j), n) for j in range(4)]
    sched += [(("final", "align" if j == 0 else "proj"), 15) for j in range(10)]

    def true_time(key, n):
        if key[0] == "inter":
            return n * cost[key[1]] * (n / 9) ** 0.3
        if key == ("final", "align"):
            return n * 2.8
        return n * 0.06

    eta = ScheduleETA(sched, t0=0.0)
    t, preds, total = 0.0, [], sum(true_time(k, n) for k, n in sched)
    for i, (k, n) in enumerate(sched):
        t += true_time(k, n)
        eta.update(t)
        preds.append(t + eta.remaining())
    # after the first stages the estimate is within 15 % and does not keep growing
    late = np.array(preds[9 + 4 * 3:])
    assert np.all(np.abs(late - total) < 0.15 * total), (late, total)
    assert np.isclose(preds[-1], total)
