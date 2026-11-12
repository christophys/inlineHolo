# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2008-2026 Christoph T. Koch
"""Tests of the FRWR3 port (inlineholo.frwr3) incl. a regression test against the C++ code."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import inlineholo as ih  # noqa: E402
from inlineholo import frwr3, frwr3_io  # noqa: E402

HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "data", "frwr3_ref")
EX = os.path.join(HERE, "..", "examples")


def test_matches_cpp_frwr3_reference():
    """R-value trace and final wave agree with the (patched) C++ frwr3 on the same input.

    Reference generated with frwr3 (2019 source, FFTW 3.3.10, see docs) on tests/data/frwr3_ref:
    7 planes, padding 0.25, MTF, alpha, objective aperture with edge, incoherent background
    and gradient flipping, 60 loops in total.
    """
    res = ih.frwr3_from_cdf(os.path.join(REF, "series.cdf"), os.path.join(REF, "run.conf"),
                            **{"print level": 0, "backend": "numpy"})
    cr = np.array([c[2] for c in json.load(open(os.path.join(REF, "rtrace_cpp.json")))])
    pr = np.array([r[2] for r in res.rvalues])
    assert len(cr) == len(pr)
    assert np.max(np.abs(pr - cr) / cr) < 1e-3
    wc = np.load(os.path.join(REF, "wave_cpp.npy"))
    c = ih.compare_to_truth(res.wave, wc, 2.0)
    assert min(c["frc_low"], c["frc_mid"], c["frc_high"]) > 0.999
    assert np.max(np.abs(res.phase - np.angle(wc))) < 1e-2


def test_conf_and_yaml_configs_identical(tmp_path):
    conf = os.path.join(EX, "frwr3_lowres2.conf")
    y = tmp_path / "c.yaml"
    frwr3.conf_to_yaml(conf, str(y))
    a = frwr3.load_config(conf)
    b = frwr3.load_config(str(y))
    for k in a:
        assert str(frwr3._vals(a[k])) == str(frwr3._vals(b[k])), k
    assert a["GF beta"] == 0.85 and frwr3._vals(a["zoomFactor_mm"])[1] == pytest.approx(39.8842)
    # full key listing parses back to the defaults
    t = tmp_path / "all.yaml"
    frwr3.default_yaml(str(t))
    c = frwr3.load_config(str(t))
    d = frwr3.load_config(None)
    for k in d:
        assert str(frwr3._vals(c[k])) == str(frwr3._vals(d[k])), k


def test_img_and_cdf_roundtrip(tmp_path):
    a = np.random.default_rng(0).standard_normal((5, 7)).astype(np.float32)
    frwr3_io.write_img(str(tmp_path / "a.img"), a, dx=2.0)
    b, h = frwr3_io.read_img(str(tmp_path / "a.img"))
    assert np.allclose(a, b) and h["dx"] == 2.0
    p = ih.ImagingParameters(voltage_V=200e3, pixel_size_nm=0.5, defoci_nm=[-100, 0, 100.],
                             aberrations={"c_3_0": 1.2e6, "c_1_2_a": 3.0, "c_1_2_b": -4.0},
                             convergence_rad=1e-4)
    s = np.random.default_rng(1).random((3, 16, 20)).astype(np.float32)
    frwr3_io.write_cdf(str(tmp_path / "s.cdf"), s, p, roi=(2, 3, 14, 19))
    s2, p2, roi, info = frwr3_io.read_cdf(str(tmp_path / "s.cdf"))
    assert np.allclose(s, s2) and roi == (2, 3, 14, 19)
    assert np.allclose(p2.defoci_nm, p.defoci_nm)
    assert p2.aberrations["c_1_2_a"] == pytest.approx(3.0, rel=1e-4)
    assert p2.aberrations["c_1_2_b"] == pytest.approx(-4.0, rel=1e-4)
    assert p2.convergence_rad == pytest.approx(1e-4, rel=1e-5)


def _small_series(drift=None):
    w = ih.simulate.mr_object(n=96, pixel=2.0, seed=4, film_nm=2.0)
    p = ih.simulate.mr_parameters(pixel_size_nm=2.0, defoci_nm=np.array([-3000., -800., 0., 800., 3000.]))
    I, _ = ih.simulate.simulate_focal_series(w, p, None, n_defocus=1, n_tilt=1)
    if drift is not None:
        I = np.stack([np.roll(I[k], tuple(drift[k]), axis=(0, 1)) for k in range(len(I))])
    return w, p, I


def test_backends_agree():
    w, p, I = _small_series()
    cfg = {"print level": 0, "max shift": 0, "final loops": 6, "intermediate loops": 3, "residue loops": 0,
           "padding": 0.25, "Nedge": 0, "Nedge for smoothing": 0}
    r1 = ih.run_frwr3(I, p, cfg, backend="numpy", precision="float64")
    r2 = ih.run_frwr3(I, p, cfg, backend="numpy", precision="float32")
    assert np.abs(r1.wave - r2.wave).max() < 1e-3
    try:
        import torch  # noqa
    except ImportError:
        pytest.skip("torch not installed")
    r3 = ih.run_frwr3(I, p, cfg, backend="torch", precision="float64")
    assert np.abs(r1.wave - r3.wave).max() < 1e-6


def test_drift_is_recovered():
    rng = np.random.default_rng(3)
    drift = rng.integers(-4, 5, (5, 2))
    drift[2] = 0
    w, p, I = _small_series(drift)
    roi = (12, 12, 84, 84)
    cfg = {"print level": 0, "max shift": 6, "final loops": 4, "intermediate loops": 3, "residue loops": 0,
           "padding": 0.25, "Nedge": 5, "Nedge for smoothing": 5, "bandpass filter high": 0.3}
    r = ih.run_frwr3(I, p, cfg, roi=roi)
    expected = drift[:, ::-1] + np.array([roi[1], roi[0]])
    err = np.abs(r.frame_offsets - expected)
    assert err[1:4].max() <= 1          # planes up to |df| = 800 nm (integer-pixel search)
    assert err.max() <= 3               # large-defocus planes: FRWR3 chi^2 registration within 3 px


def test_reconstruct_registry_frwr3_runs():
    w, p, I = _small_series()
    for m in ("frwr", "frwr_gf"):
        r = ih.reconstruct(I, p, m, n_iter=5, intermediate_loops=2)
        assert r.wave.shape == I.shape[1:]
        assert np.isfinite(r.wave).all()
        assert "frwr3" in r.extra


def test_fokker_planck_diffusion_and_tie_fp():
    from scipy.ndimage import gaussian_filter
    from inlineholo import fokker_planck as fp
    rng = np.random.default_rng(0)
    n, px = 96, 1.0
    ph = gaussian_filter(rng.standard_normal((n, n)), 5, mode="wrap")
    w = (1 - 0.1 * gaussian_filter(rng.standard_normal((n, n)), 4, mode="wrap")) * np.exp(1j * ph / ph.std())
    y, x = np.mgrid[0:n, 0:n]
    D = 1e-6 * np.exp(-((x - 35) ** 2 + (y - 45) ** 2) / (2 * 10 ** 2))
    grid = ih.FourierGrid((n, n), px)
    I0 = np.abs(w) ** 2
    lapDI = np.real(np.fft.ifft2(-4 * np.pi ** 2 * grid.q2 * np.fft.fft2(D * I0)))
    for zs in ([-400., -200., 0., 200., 400.], [0., 200., 400., 600.]):
        p = ih.ImagingParameters(voltage_V=300e3, pixel_size_nm=px, defoci_nm=np.array(zs))
        I = ih.ForwardModel(p, (n, n), "coherent", use_mtf=False).forward(w)
        Id = I + np.array(zs)[:, None, None] ** 2 * lapDI
        r = fp.diffusion_field(Id + 0.03, p, w, model="coherent", background=True)
        assert np.corrcoef(r["D"].ravel(), D.ravel())[0, 1] > 0.98
        assert abs(r["B"].mean() - 0.03) < 1e-3
        t = fp.tie_fp(Id, p)
        t0 = fp.tie_fp(I, p)
        e = ih.compare_to_truth(t.wave, w, px)["rms_phase"]
        e0 = ih.compare_to_truth(t0.wave, w, px)["rms_phase"]
        assert abs(e - e0) < 1e-3              # dz^2 diffusion terms do not bias tie_fp
        assert np.corrcoef(t.extra["D"].ravel(), D.ravel())[0, 1] > 0.95
    # first-order TIE on the one-sided series is biased by the diffusion term
    e1 = ih.compare_to_truth(ih.reconstruct(Id, p, "tie_full", normalize=False).wave, w, px)["rms_phase"]
    assert e1 > e + 0.01
