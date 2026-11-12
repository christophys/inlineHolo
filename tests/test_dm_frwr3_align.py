# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2008-2026 Christoph T. Koch
"""DM3 reading (built-in reader, FRWR tags), DM3 -> NXem conversion and FRWR3 distortion alignment."""
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import inlineholo as ih  # noqa: E402
from inlineholo import frwr3  # noqa: E402
from inlineholo.dmreader import read_dm  # noqa: E402
from inlineholo.io_formats import load_series  # noqa: E402


# ------------------------------------------------------------------ minimal DM3 writer
def _tag_data(name, info, payload):
    b = bytes([21]) + struct.pack(">H", len(name)) + name.encode()
    b += b"%%%%" + struct.pack(">I", len(info)) + b"".join(struct.pack(">I", i) for i in info) + payload
    return b


def _scalar(name, v):
    if isinstance(v, float):
        return _tag_data(name, [7], struct.pack("<d", v))
    return _tag_data(name, [3], struct.pack("<i", v))


def _string(name, s):
    a = np.frombuffer(s.encode("utf-16-le"), np.uint16)
    return _tag_data(name, [20, 4, len(a)], a.astype("<u2").tobytes())


def _group(name, entries):
    head = (bytes([20]) + struct.pack(">H", len(name)) + name.encode()) if name is not None else b""
    if name == "":
        head = bytes([20]) + struct.pack(">H", 0)
    return head + bytes([0, 1]) + struct.pack(">I", len(entries)) + b"".join(entries)


def write_dm3(path, data, scale=None, units="nm", tags=None):
    data = np.asarray(data, np.float32)
    nz, ny, nx = data.shape
    dims = _group("Dimensions", [_scalar("", nx), _scalar("", ny), _scalar("", nz)])
    cal = [_group("", [_scalar("Origin", 0.0), _scalar("Scale", float(scale or 1.0)),
                       _string("Units", units if scale else "")]) for _ in range(3)]
    calib = _group("Calibrations", [_group("Dimension", cal)])
    arr = _tag_data("Data", [20, 6, data.size], data.astype("<f4").tobytes())
    idata = _group("ImageData", [arr, dims, calib, _scalar("DataType", 2)])

    def tg(name, d):
        out = []
        for k, v in d.items():
            if isinstance(v, dict):
                out.append(tg(k, v))
            elif isinstance(v, str):
                out.append(_string(k, v))
            else:
                out.append(_scalar(k, v))
        return _group(name, out)

    img = _group("", [idata, tg("ImageTags", tags or {}), _string("Name", "test")])
    root = _group(None, [_group("ImageList", [img])])
    body = root
    with open(path, "wb") as f:
        f.write(struct.pack(">iIi", 3, len(body) + 16, 1) + body + b"\0" * 8)


FRWR_TAGS = {"FRWR": {"FocalSeries": {"DeltaFoc": 800.0, "offsetFoc": -100.0, "stepsFoc": 5.0,
                                      "Acquisition Time": "17.09.2012 23:10:31"},
                      "Info": {"AstigMag": 20.0, "AstigAngle": 30.0, "alpha": 0.2, "Delta": 5.0,
                               "Microscope": "Zeiss SESAM", "Specimen": "CeO2", "Operator": "CTK",
                               "ObjectiveAperture": 5.0}},
             "Microscope Info": {"Cs": 1.2, "Voltage": 200000.0}}


def test_dm3_reader_and_frwr_tags(tmp_path):
    data = np.random.default_rng(0).random((5, 24, 32)).astype(np.float32)
    fn = str(tmp_path / "s.dm3")
    write_dm3(fn, data, 0.25, "nm", FRWR_TAGS)
    ims, tags = read_dm(fn)
    assert np.array_equal(ims[0]["data"], data) and ims[0]["units"][-1] == "nm"
    st, md, p = load_series(fn)
    assert st.shape == (5, 24, 32) and p is not None
    assert np.allclose(p.defoci_nm, [-1700, -900, -100, 700, 1500])
    assert p.voltage_V == 200e3 and np.isclose(p.aberrations["c_3_0"], 1.2e6)
    assert np.isclose(np.hypot(p.aberrations["c_1_2_a"], p.aberrations["c_1_2_b"]), 20.0)
    assert np.isclose(p.convergence_rad, 0.2e-3 / np.sqrt(2)) and np.isclose(p.aperture_rad, 5e-3)
    assert md["nxem_hints"]["specimen"] == "CeO2"
    # uncalibrated file: no pixel size, no parameters
    write_dm3(str(tmp_path / "r.dm3"), data, None)
    st, md, p = load_series(str(tmp_path / "r.dm3"))
    assert p is None and md.get("uncalibrated") and "pixel_size_nm" not in md


def test_app_converts_dm3_to_valid_nxem(tmp_path, monkeypatch):
    pytest.importorskip("pynxtools")
    import matplotlib
    matplotlib.use("Agg")
    from inlineholo.app import App
    fn = str(tmp_path / "s.dm3")
    write_dm3(fn, np.random.default_rng(1).random((5, 24, 32)) + 1, 0.25, "nm", FRWR_TAGS)
    monkeypatch.chdir(tmp_path)
    a = App()
    a._load_path(fn)
    assert a.w_conv_box.layout.display != "none"
    m = a.metadata()
    assert m["sample"]["atom_types"] == "Ce, O" and m["instrument"]["vendor"] == "Carl Zeiss"
    assert m["experiment"]["start_time"].startswith("2012-09-17T23:10")
    assert a._convert() is None                        # preparation date and emitter type missing
    a.w_md[("sample", "preparation_date")].value = "2012-09-01"
    a.w_md[("instrument", "emitter_type")].value = "schottky"
    out = a._convert()
    assert out.endswith("s.nxs") and nxem_valid(out)
    st, p, _, _ = ih.nxem.read_focal_series(out)
    assert st.shape == (5, 24, 32) and np.allclose(p.defoci_nm, [-1700, -900, -100, 700, 1500])


def nxem_valid(path):
    ok, probs, _ = ih.nxem.validate(path)
    return ok


def test_frwr3_recovers_defocus_dependent_distortions():
    """Synthetic series with rotation / magnification linear in defocus plus drift."""
    w = ih.simulate.mr_object(n=256, pixel=2.0, seed=3, film_nm=2.0)
    df = np.array([-6000., -3000., -1000., 0., 1000., 3000., 6000.])
    p = ih.simulate.mr_parameters(pixel_size_nm=2.0, defoci_nm=df)
    I, _ = ih.simulate.simulate_focal_series(w, p, None, n_defocus=1, n_tilt=1)
    rot_mm, zoom_mm = -2.0, 1.5                               # per mm of defocus
    drift = np.random.default_rng(5).integers(-5, 6, (len(df), 2)).astype(float)
    drift[3] = 0
    true = np.zeros((len(df), 6))
    frames = []
    for k, z in enumerate(df):
        true[k, frwr3.DIST_ROT], true[k, frwr3.DIST_ZOOM] = rot_mm * z * 1e-6, zoom_mm * z * 1e-6
        fr, _ = frwr3.warp_frame(I[k], -true[k] / (1 + true[k]))   # recorded = true image seen through the lens
        frames.append(np.roll(fr, tuple(drift[k].astype(int)), axis=(0, 1)))
    frames = np.array(frames) + 1e-3
    for refine in (0, 3):
        out, info = ih.align.frwr3_align(frames, p, binning=1, margin_px=24, max_shift_px=12, final_loops=4,
                                         config={"distortion refine": refine})
        d = info["distortions"]
        # displacement error of rotation + magnification 100 px from the centre (sub-pixel)
        err = np.hypot((d[:, 0] - true[:, 0]) * 100, (d[:, 1] - true[:, 1]) * 100)
        assert err.max() < 0.6, (refine, err)
        # drift (x, y) recovered to better than one pixel
        assert np.abs(info["shifts_px"] - drift[:, ::-1]).max() < 1.0, info["shifts_px"]
        t, l, b, r = info["crop"]
        for k in range(len(df)):
            c1 = np.corrcoef(out[k][8:-8, 8:-8].ravel(), I[k][t:b, l:r][8:-8, 8:-8].ravel())[0, 1]
            c0 = np.corrcoef(frames[k][t:b, l:r][8:-8, 8:-8].ravel(), I[k][t:b, l:r][8:-8, 8:-8].ravel())[0, 1]
            assert c1 > 0.9 and c1 >= c0 - 1e-3, (k, c0, c1)
