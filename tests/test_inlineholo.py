# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import inlineholo as ih  # noqa: E402
from inlineholo import simulate as sim  # noqa: E402


def small_params(**kw):
    d = dict(voltage_V=300e3, pixel_size_nm=0.05, defoci_nm=np.array([-8.0, -2.0, 3.0, 9.0]),
             aberrations={"c_3_0": 2e4, "c_1_2_a": 0.8, "c_2_1_b": 30.0},
             focal_spread_nm=3.0, convergence_rad=0.15e-3, mtf={"type": "sinc"})
    d.update(kw)
    return ih.ImagingParameters(**d)


def random_wave(n=32, seed=0, strength=0.3):
    rng = np.random.default_rng(seed)
    ph = strength * rng.standard_normal((n, n))
    am = 1 + 0.05 * rng.standard_normal((n, n))
    return am * np.exp(1j * ph)


@pytest.mark.parametrize("kind", ih.ForwardModel.KINDS)
@pytest.mark.parametrize("loss", ["lsq", "amplitude", "poisson"])
def test_gradient(kind, loss):
    p = small_params()
    psi = random_wave()
    fm = ih.ForwardModel(p, psi.shape, kind, n_defocus=3, n_tilt=2)
    data = fm.forward(random_wave(seed=5)) + 0.01
    L, g = fm.loss_and_grad(psi, data, loss)
    rng = np.random.default_rng(1)
    for _ in range(3):
        d = rng.standard_normal(psi.shape) + 1j * rng.standard_normal(psi.shape)
        h = 1e-6
        Lp, _ = fm.loss_and_grad(psi + h * d, data, loss)
        Lm, _ = fm.loss_and_grad(psi - h * d, data, loss)
        num = (Lp - Lm) / (2 * h)
        ana = np.sum(g.real * d.real + g.imag * d.imag)
        assert np.isclose(num, ana, rtol=1e-4, atol=1e-12), (num, ana)


def test_flux_conservation():
    psi = random_wave(64)
    for kind in ih.ForwardModel.KINDS:
        if kind == "envelope":
            continue  # envelope damping does not conserve flux
        # temporal envelope on the wave (blur model) is not flux preserving -> switch off
        p = small_params(focal_spread_nm=0.0 if kind == "blur" else 3.0)
        I = ih.ForwardModel(p, psi.shape, kind, use_mtf=False).forward(psi)
        assert np.allclose(I.mean(axis=(1, 2)), np.mean(np.abs(psi) ** 2), rtol=1e-6)


def test_envelope_matches_incoherent_for_weak_object():
    """Linear (weak-object) imaging: envelope model == incoherent sum to first order."""
    p = small_params(mtf=None)
    rng = np.random.default_rng(3)
    psi = 1 + 1j * 0.002 * rng.standard_normal((64, 64))
    I_env = ih.ForwardModel(p, psi.shape, "envelope").forward(psi)
    I_inc = ih.ForwardModel(p, psi.shape, "incoherent", n_defocus=9, n_tilt=7).forward(psi)
    c_env = I_env - I_env.mean(axis=(1, 2), keepdims=True)
    c_inc = I_inc - I_inc.mean(axis=(1, 2), keepdims=True)
    assert np.linalg.norm(c_env - c_inc) / np.linalg.norm(c_inc) < 0.03


def test_tie_sign_and_accuracy():
    n, px = 128, 1.0
    y, x = np.mgrid[0:n, 0:n]
    phase = 0.5 * np.exp(-((x - 64) ** 2 + (y - 60) ** 2) / (2 * 8 ** 2))
    psi = np.exp(1j * phase)
    p = ih.ImagingParameters(voltage_V=300e3, pixel_size_nm=px, defoci_nm=[-200, 0, 200])
    I = ih.ForwardModel(p, psi.shape, "coherent").forward(psi)
    r = ih.reconstruct(I, p, "tie", q_reg=1e-4)
    rph = r.phase - r.phase.mean()
    tph = phase - phase.mean()
    assert np.corrcoef(rph.ravel(), tph.ravel())[0, 1] > 0.99


@pytest.mark.parametrize("method", ["pam", "schiske"])
def test_linear_methods_weak_object(method):
    p = sim.hr_parameters(n_images=10, start=-10, step=2.0, mtf=None)
    gt = 1 + 1j * 0.02 * np.random.default_rng(0).standard_normal((64, 64))
    from scipy.ndimage import gaussian_filter
    gt = np.exp(1j * gaussian_filter(np.angle(gt), 1.0, mode="wrap") * 5)
    I = ih.ForwardModel(p, gt.shape, "envelope").forward(gt)
    r = ih.reconstruct(I, p, method, reg=1e-3)
    res = ih.compare_to_truth(r.wave, gt, p.pixel_size_nm)
    assert res["mean_frc"] > 0.8


@pytest.mark.parametrize("method", ["gs", "iwfr", "frwr", "frwr_gf", "frwr_gd", "gpfrwr_gd", "mal", "ml_poisson"])
def test_iterative_methods_reduce_error(method):
    p = small_params(defoci_nm=np.array([-12., -6., -2., 2., 6., 12.]), mtf=None,
                     focal_spread_nm=1.0, convergence_rad=0.05e-3)
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(2)
    ph = gaussian_filter(rng.standard_normal((48, 48)), 1.5, mode="wrap")
    gt = np.exp(1j * 0.6 * ph / ph.std())
    I = ih.ForwardModel(p, gt.shape, "incoherent", n_defocus=3, n_tilt=2).forward(gt)
    opts = dict(n_iter=40)
    if method == "ml_poisson":
        opts.update(n_defocus=3, n_tilt=2)
    r = ih.reconstruct(I, p, method, **opts)
    res = ih.compare_to_truth(r.wave, gt, p.pixel_size_nm)
    flat = ih.compare_to_truth(np.ones_like(gt) + 0.01 * rng.standard_normal(gt.shape), gt,
                               p.pixel_size_nm)
    assert res["nrmse"] < 0.9 * flat["nrmse"], (res["nrmse"], flat["nrmse"])


def test_nxem_roundtrip(tmp_path):
    p = small_params()
    gt = random_wave(32)
    I = ih.ForwardModel(p, gt.shape, "envelope").forward(gt) * 100
    f = tmp_path / "series.nxs"
    ih.nxem.write_focal_series(f, I, p, ground_truth=gt, is_simulation=True)
    r = ih.reconstruct(I, p, "pam")
    path = ih.nxem.write_reconstruction(f, r, p)
    assert path.endswith("image1")
    stack, p2, gt2, recs = ih.nxem.read_focal_series(f)
    assert np.allclose(stack, I)
    assert np.allclose(p2.defoci_nm, p.defoci_nm)
    assert np.isclose(p2.aberrations["c_3_0"], p.aberrations["c_3_0"])
    assert np.isclose(p2.focal_spread_nm, p.focal_spread_nm)
    assert np.isclose(p2.convergence_rad, p.convergence_rad)
    assert np.isclose(p2.pixel_size_nm, p.pixel_size_nm)
    assert np.allclose(p2.astig_a_nm, p.astig_a_nm)
    assert np.allclose(gt2, gt)
    assert np.allclose(recs["pam"], r.wave)


def test_frc_identity_and_noise():
    a = random_wave(64)
    q, f, n = ih.frc(a, a, 1.0)
    assert np.allclose(f[1:], 1)
    b = np.random.default_rng(9).standard_normal((64, 64))
    c = np.random.default_rng(10).standard_normal((64, 64))
    q, f, n = ih.frc(b, c, 1.0)
    assert np.abs(f[5:]).mean() < 0.1


def test_thon_fit_recovers_defocus_and_astigmatism():
    for c1, a, b in [(-35.0, 2.0, -1.5), (-80.0, -1.0, 1.0)]:
        p = sim.hr_parameters(n_images=1, start=c1, step=1,
                              aberrations={"c_3_0": 1e4, "c_1_2_a": a, "c_1_2_b": b})
        gt = sim.hr_object(256, 0.04, crystal_radius_nm=0.1)   # mostly amorphous carbon
        I, _ = sim.simulate_focal_series(gt, p, dose_per_image=20000, n_defocus=3, n_tilt=2)
        r = ih.ctffit.fit_thon(I[0], 0.04, 300e3, (c1 - 20, c1 + 20), 1e4)
        assert abs(r["c_1_0"] - c1) < 1.0
        assert abs(r["c_1_2_a"] - a) < 1.0 and abs(r["c_1_2_b"] - b) < 1.0


def test_register_stack():
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(rng.standard_normal((96, 96)), 2, mode="wrap")
    sh = np.array([[0, 0], [1.5, -2.25], [-3.0, 0.5]])
    stack = np.stack([ih.align.fourier_shift(img, -s) for s in sh])
    out, est = ih.align.register_stack(stack, defoci=np.array([0, 1, 2.]), reference=0)
    assert np.allclose(est, sh, atol=0.1)
    assert np.allclose(out[1], img, atol=1e-2 * np.abs(img).max())


def test_app_headless(tmp_path, monkeypatch):
    """Smoke test of the ipywidgets app: load NXem, Thon fit, align, reconstruct, export."""
    import matplotlib
    matplotlib.use("Agg")
    from inlineholo.app import App
    p = sim.hr_parameters(n_images=4, start=-40, step=10, aberrations={"c_3_0": 1e4})
    gt = sim.hr_object(128, 0.04, crystal_radius_nm=1.0)
    I, _ = sim.simulate_focal_series(gt, p, dose_per_image=2000, n_defocus=3, n_tilt=2)
    f = tmp_path / "s.nxs"
    ih.nxem.write_focal_series(f, I, p, ground_truth=gt, is_simulation=True)
    monkeypatch.chdir(tmp_path)
    a = App().load(str(f))
    assert np.allclose(a.params().defoci_nm, p.defoci_nm)
    a._fit(single=True)
    a._fit_global()
    a._apply_fits()
    a._align()
    a.w_method.value = "frwr"
    a.w_iter.value = 5
    a.w_frwr3_yaml.value = "fit defocus: 0\nmax shift: 0\nintermediate loops: 2\npadding: 0.25"
    a._run()
    assert a.rec is not None
    assert a.rec.extra["frwr3"].config["intermediate loops"] == 2
    a._save()                                   # refused: required NXem metadata missing
    assert not (tmp_path / "output" / "focal_series_reconstruction.nxs").exists()
    a.set_metadata({"sample": {"atom_types": "Sr, Ti, O, C", "preparation_date": "2026-01-01"},
                    "experiment": {"start_time": "2026-11-01T10:00:00+01:00"}})
    assert not ih.nxem.missing_required(a.metadata())   # instrument prefilled for simulated data
    a._strain()
    a._save()
    out = tmp_path / "output" / "focal_series_reconstruction.nxs"
    _, _, _, recs = ih.nxem.read_focal_series(out)
    assert "frwr" in recs
    assert any(k == "strain_gg" for k, _ in ih.nxem.read_maps(out).values())
    assert ih.nxem.read_metadata(out)["sample"]["atom_types"] == "Sr, Ti, O, C"


def test_mftie_and_gftie_recover_smooth_phase():
    n = 128
    y, x = np.mgrid[0:n, 0:n]
    phase = 1.5 * np.exp(-((x - 64) ** 2 + (y - 60) ** 2) / (2 * 12 ** 2))
    psi = np.exp(1j * phase)
    p = ih.ImagingParameters(voltage_V=300e3, pixel_size_nm=1.0,
                             defoci_nm=[-2000, -500, -100, 0, 100, 500, 2000])
    I = ih.ForwardModel(p, psi.shape, "coherent").forward(psi)
    for m, o in (("mftie", {}), ("gftie", dict(n_iter=10))):
        r = ih.reconstruct(I, p, m, **o)
        rph = r.phase - r.phase.mean()
        assert np.corrcoef(rph.ravel(), (phase - phase.mean()).ravel())[0, 1] > 0.97, m


def test_offaxis_roundtrip():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    ph = gaussian_filter(rng.standard_normal((128, 128)), 6, mode="wrap")
    psi = np.exp(1j * ph / ph.std())
    H, qc = ih.offaxis.simulate_offaxis(psi, 1.0, 0.25, 0, 0.5, None, mtf=False)
    w = ih.offaxis.reconstruct_offaxis(H, 1.0, qc)
    assert ih.compare_to_truth(w, psi, 1.0)["frc_low"] > 0.95


def test_strain_from_geometric_phase():
    w, eps, bar, g = sim.dih_object(128, 0.25, mip_step_rad=0.0)
    e, _ = ih.strain.strain_1d(w, g, 0.25, ref_mask=bar)
    assert np.abs(e - eps).max() < 1e-3
    # 2D: two reflections, uniform strain
    n, px = 64, 0.25
    yy, xx = np.mgrid[0:n, 0:n] * px
    exx, eyy = 0.01, -0.005
    g1, g2 = (0.0, 4.0), (4.0, 0.0)            # (g_y, g_x)
    # displacement u = (exx x, eyy y) -> phase -2 pi g.u (non periodic is fine for interior)
    w1 = np.exp(-2j * np.pi * (g1[1] * exx * xx + g1[0] * eyy * yy))
    w2 = np.exp(-2j * np.pi * (g2[1] * exx * xx + g2[0] * eyy * yy))
    a, b, c, r = ih.strain.strain_2d(w1, w2, g1, g2, px)
    sl = slice(16, 48)
    assert abs(np.median(a[sl, sl]) - exx) < 2e-3 and abs(np.median(b[sl, sl]) - eyy) < 2e-3


def test_hybrid_and_padding_run():
    p = small_params(defoci_nm=np.array([-12., -6., 0., 6., 12.]), mtf=None)
    gt = random_wave(32, strength=0.2)
    I = ih.ForwardModel(p, gt.shape, "blur").forward(gt)
    r = ih.reconstruct(I, p, "hybrid", offaxis_wave=gt * 1.0, refine="frwr", n_iter=5)
    assert r.wave.shape == gt.shape
    r = ih.reconstruct(I, p, "gpfrwr_gd", n_iter=12, pad=1.5, n_start=1, every=2, n_final=2)
    assert r.wave.shape == gt.shape
    r = ih.reconstruct(I, p, "gpfrwr", n_iter=8, pad=1.5, intermediate_loops=2)
    assert r.wave.shape == gt.shape
