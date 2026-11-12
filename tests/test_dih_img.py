# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Many-beam DIH strain analysis (port of the DM DIH-GPA scripts) and .img frame series."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import inlineholo as ih  # noqa: E402
from inlineholo import dih  # noqa: E402
from inlineholo.frwr3_io import write_img, read_img  # noqa: E402
from inlineholo.io_formats import load_series, load_folder, img_series_files, defocus_from_name  # noqa: E402


def test_g_vectors_2d():
    si = dih.LATTICES["Si"]
    g = dih.g_vectors_2d([(2, 0, 0), (0, 2, 0), (2, 2, 0)], si, zone=(0, 0, 1), x_hkl=(1, 0, 0))
    a = si[0]
    assert np.allclose(g, [[2 / a, 0], [0, 2 / a], [2 / a, 2 / a]])
    # [110] zone, x along (1 -1 0): (002) points along y, (1 -1 1) has both components
    g = dih.g_vectors_2d([(0, 0, 2), (2, -2, 0), (1, -1, 1)], si, zone=(1, 1, 0), x_hkl=(1, -1, 0))
    assert np.allclose(g[0], [0, -2 / a]) or np.allclose(g[0], [0, 2 / a])
    assert np.allclose(g[1], [2 * np.sqrt(2) / a, 0])
    # rotation of the crystal by 90 degrees
    g = dih.g_vectors_2d([(2, 0, 0)], si, rotation_deg=90)
    assert np.allclose(g, [[0, 2 / a]])
    # hexagonal GaN [0001]: |g(100)| = 2 / (sqrt(3) a)
    gan = dih.LATTICES["GaN"]
    g = dih.g_vectors_2d([(1, 0, 0), (0, 1, 0)], gan, zone=(0, 0, 1), x_hkl=(1, 0, 0))
    assert np.allclose(np.linalg.norm(g, axis=1), 2 / (np.sqrt(3) * gan[0]))
    assert np.isclose(np.degrees(np.arccos(g[0] @ g[1] / np.linalg.norm(g[0]) / np.linalg.norm(g[1]))), 60)
    assert dih.parse_hkl("[-110]_amplitude") == (-1, 1, 0) and dih.parse_hkl("(2 0 0)") == (2, 0, 0)
    assert dih.parse_hkl("2 -2 0") == (2, -2, 0) and dih.parse_hkl("x") is None


def _synthetic(n=128, pix=0.05):
    """Displacement field with e_xx = 1 % (right half), e_yy = -0.5 %, rotation 2 mrad (smooth)."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    x, y = xx * pix, -yy * pix                      # y upwards
    s = 0.5 * (1 + np.tanh((xx - n / 2) / 4))       # 0 left, 1 right
    from scipy.integrate import cumulative_trapezoid
    exx = 0.01 * s
    eyy = -0.005 * np.ones_like(s)
    w = 0.002
    ux = cumulative_trapezoid(exx, x, axis=1, initial=0) - w * y
    uy = eyy * y + w * x
    return dict(ux=ux, uy=uy, exx=exx, eyy=eyy, w=w, pix=pix)


def test_strain_from_phases_recovers_tensor():
    S = _synthetic()
    lat = dih.LATTICES["test (cubic 0.456 nm)"]
    hkls = [(2, 0, 0), (0, 2, 0), (1, 1, 0), (-1, 1, 0)]
    g = dih.g_vectors_2d(hkls, lat)
    rng = np.random.default_rng(0)
    refls = []
    for h, gv in zip(hkls, g):
        ph = -2 * np.pi * (gv[0] * S["ux"] + gv[1] * S["uy"]) + rng.uniform(-3, 3)     # arbitrary offsets
        amp = 1 + 0.2 * rng.random(ph.shape)
        refls.append(dih.DFReflection(h, amp, np.angle(np.exp(1j * ph)), S["pix"], name=str(h)))
    res = dih.strain_from_phases(refls, lattice=lat, ref_box=(10, 10, 118, 50), reference_correction=False)
    c = (slice(5, -5), slice(5, -5))
    assert np.nanmax(np.abs(res["e_xx"][c] - S["exx"][c])) < 2e-4
    assert np.nanmax(np.abs(res["e_yy"][c] - S["eyy"][c])) < 2e-4
    assert np.nanmax(np.abs(res["e_xy"][c])) < 2e-4
    assert np.nanmax(np.abs(res["rotation"][c] - S["w"])) < 2e-4
    # reference correction: the reference area (left part) becomes unstrained and unrotated
    res = dih.strain_from_phases(refls, lattice=lat, ref_box=(10, 10, 118, 50))
    assert abs(np.nanmedian(res["e_xx"][20:100, 15:45])) < 1e-4 and abs(np.nanmedian(res["e_yy"][20:100, 15:45])) < 1e-4
    assert np.isclose(np.nanmedian(res["e_xx"][20:100, 90:120]), 0.01, atol=3e-4)
    # band limit and a single reflection (strain along g)
    res = dih.strain_from_phases(refls[:1], lattice=lat, reference_correction=False, resolution_nm=0.5, smooth=0.5)
    assert np.isclose(np.nanmedian(res["e_gg"][20:100, 90:120]), 0.01, atol=3e-4)
    # displacement variant (unwrapped phases, plane removed in the reference area)
    du = dih.displacement_from_phases(refls, lattice=lat, ref_box=(10, 10, 118, 50))
    ux = S["ux"] - S["ux"][10:118, 10:50].mean()
    gx = np.gradient(du["u_x"], S["pix"], axis=1)
    assert np.isclose(np.median(gx[20:100, 90:120]) - np.median(gx[20:100, 15:45]), 0.01, atol=5e-4)
    assert np.isfinite(ux).all()


def test_load_reflections_pairs_waves_and_shifts(tmp_path):
    S = _synthetic(64)
    lat = dih.LATTICES["test (cubic 0.456 nm)"]
    for h in [(2, 0, 0), (0, 2, 0)]:
        gv = dih.g_vectors_2d([h], lat)[0]
        ph = -2 * np.pi * (gv[0] * S["ux"] + gv[1] * S["uy"])
        tag = "[" + "".join(str(v) for v in h) + "]"
        write_img(str(tmp_path / f"{tag}_amplitude.img"), np.ones_like(ph), dx=10 * S["pix"], dy=10 * S["pix"])
        write_img(str(tmp_path / f"{tag}_phase.img"), ph, dx=10 * S["pix"], dy=10 * S["pix"])
    np.save(tmp_path / "wave_[110].npy", np.exp(1j * ph))
    files = sorted(str(p) for p in tmp_path.iterdir())
    refls, _ = dih.load_reflections(files)
    assert sorted(r.hkl for r in refls) == [(0, 2, 0), (1, 1, 0), (2, 0, 0)]
    assert all(np.isclose(r.pixel_nm, S["pix"]) for r in refls if r.source.endswith(".img"))
    # shifts are applied before the analysis
    r = refls[0]
    r.shift_px = (3, -2)
    use, A, P = dih.common_frame([r])
    assert np.allclose(P[0], np.roll(r.phase, (-2, 3), (0, 1)))


def test_img_series_and_folder_loading(tmp_path):
    rng = np.random.default_rng(1)
    frames = rng.random((12, 20, 24)).astype(np.float32)
    for i, f in enumerate(frames):
        write_img(str(tmp_path / f"image_{i}.img"), f, dx=2.5, dy=2.5, t=-1000.0 + 100 * i)
        write_img(str(tmp_path / f"imageRef_{i}.img"), f * 0, dx=2.5, dy=2.5)
    a, h = read_img(str(tmp_path / "image_3.img"))
    assert np.allclose(a, frames[3]) and h["dx"] == 2.5
    fs = img_series_files(str(tmp_path / "image_7.img"))
    assert [os.path.basename(f) for f in fs] == [f"image_{i}.img" for i in range(12)]
    st, md, p = load_series(str(tmp_path / "image_10.img"))
    assert st.shape == (12, 20, 24) and np.allclose(st[11], frames[11])
    assert np.isclose(md["pixel_size_nm"], 0.25) and np.allclose(md["defoci_nm"], -100 + 10 * np.arange(12))
    # folder of single frames named by their defocus
    d = tmp_path / "df"
    d.mkdir()
    dfs = [-1000, -250, 0, 250, 1000, 500, -500]
    for v in dfs:
        np.save(d / f"F{v}nm.npy", np.full((8, 8), float(v)))
    st, md = load_folder(str(d))
    assert np.allclose(md["defoci_nm"], sorted(dfs)) and np.allclose(st[:, 0, 0], sorted(dfs))
    assert defocus_from_name("F-1000nm.dm3") == -1000 and defocus_from_name("df_+2.5um.tif") == 2500


def test_app_tab7_many_beam_dih(tmp_path, monkeypatch):
    pytest.importorskip("pynxtools")
    import matplotlib
    matplotlib.use("Agg")
    from inlineholo.app import App
    S = _synthetic(96)
    lat = dih.LATTICES["Si"]
    for h in [(2, 0, 0), (0, 2, 0), (2, 2, 0)]:
        gv = dih.g_vectors_2d([h], lat)[0]
        w = np.exp(-2j * np.pi * (gv[0] * S["ux"] + gv[1] * S["uy"]))
        np.save(tmp_path / f"wave_{''.join(str(v) for v in h)}.npy", w)
    monkeypatch.chdir(tmp_path)
    a = App()
    assert a.tabs.titles[6] == "7 DIH strain"
    a._dih_add_paths("; ".join(str(tmp_path / f) for f in sorted(os.listdir(tmp_path))))
    assert [r.hkl for r in a.dih_refls] == [(0, 2, 0), (2, 0, 0), (2, 2, 0)]
    for r in a.dih_refls:
        r.pixel_nm = S["pix"]
    for w, v in zip(a.w_dih_box, (10, 5, 86, 35)):
        w.value = v
    a.w_dih_res.value = 0
    res = a._dih_compute()
    assert res is not None and np.isclose(np.nanmedian(res["e_xx"][20:80, 70:90]), 0.01, atol=5e-4)
    assert a.w_dih_out.outputs and sum(m["kind"] == "dih" for m in a.maps) == 4
    a._dih_overlay()
    # with a focal series loaded, the maps are part of the validated NXem export
    p = ih.simulate.hr_parameters(n_images=3, start=-20, step=10)
    I, _ = ih.simulate.simulate_focal_series(ih.simulate.hr_object(96, S["pix"]), p, dose_per_image=None,
                                             n_defocus=1, n_tilt=1)
    f = tmp_path / "s.nxs"
    ih.nxem.write_focal_series(f, I, p, is_simulation=True)
    a.load(str(f))
    a._dih_compute()
    a.w_placeholder.value = True
    a.w_fname.value = "dih.nxs"
    a._save()
    out = tmp_path / "output" / "dih.nxs"
    assert ih.nxem.validate(str(out))[0]
    kinds = [k for k, _ in ih.nxem.read_maps(str(out)).values()]
    assert {"dih_e_xx", "dih_e_yy", "dih_e_xy", "dih_rotation"} <= set(kinds)
