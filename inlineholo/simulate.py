# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Test objects and focal-series simulation (thin-object approximation).

* :func:`hr_object` - high-resolution test object: SrTiO3-like [001] projected crystal
  (wedge-shaped, rounded particle with an O-vacancy column and a displaced column) on an
  amorphous carbon film; atom/column potentials modelled as Gaussians.
* :func:`mr_object` - medium-resolution test object: MgO-like cube, carbon sphere with a
  metal core and small particles on a weak amorphous film, phases from the mean inner
  potential (strong, wrapping phase), weak absorption.
* :func:`simulate_focal_series` - image formation with the rigorous incoherent model
  (more quadrature points than used for reconstruction, detector MTF) plus Poisson noise.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .forward import ForwardModel, fft2, ifft2
from .optics import interaction_constant
from .params import ImagingParameters


def _splat(shape, pixel, pos_nm, weights):
    """Bilinear periodic splatting of point weights onto a grid."""
    ny, nx = shape
    img = np.zeros(shape)
    x = pos_nm[:, 0] / pixel
    y = pos_nm[:, 1] / pixel
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(img, ((y0 + dy) % ny, (x0 + dx) % nx), weights * w)
    return img


def _gauss_blur_fourier(img, pixel, sigma_nm):
    ny, nx = img.shape
    qy = np.fft.fftfreq(ny, d=pixel)[:, None]
    qx = np.fft.fftfreq(nx, d=pixel)[None, :]
    return np.real(ifft2(fft2(img) * np.exp(-2 * np.pi ** 2 * sigma_nm ** 2 * (qx ** 2 + qy ** 2))))


def hr_object(n=256, pixel=0.04, seed=1, a=0.3905, crystal_radius_nm=None,
              phases=(0.55, 0.45, 0.12), wedge=0.6, carbon_density=60.0,
              carbon_phase=0.04, sigma_col=0.035, absorption=0.08):
    """High-resolution test object (complex exit wave, thin-object approximation).

    phases: peak-integrated column phase scale for (Sr, TiO, O) columns in rad at the thick
    end of the wedge; carbon film: atoms per nm^2 and integrated phase per atom.
    """
    rng = np.random.default_rng(seed)
    L = n * pixel
    if crystal_radius_nm is None:
        crystal_radius_nm = 0.32 * L
    cx, cy = 0.55 * L, 0.48 * L
    pos, w = [], []
    nc = int(np.ceil(L / a)) + 1
    for i in range(-1, nc):
        for j in range(-1, nc):
            for (ox, oy), ph in (((0, 0), phases[0]), ((0.5, 0.5), phases[1]),
                                 ((0.5, 0), phases[2]), ((0, 0.5), phases[2])):
                x, y = (i + ox) * a + 0.13, (j + oy) * a + 0.21
                r = np.hypot(x - cx, y - cy)
                if r > crystal_radius_nm:
                    continue
                t = (1 - wedge) + wedge * (x - (cx - crystal_radius_nm)) / (2 * crystal_radius_nm)
                pos.append((x, y))
                w.append(ph * t)
    pos = np.array(pos)
    w = np.array(w)
    # defects: one missing O column, one displaced Sr column
    d = np.hypot(pos[:, 0] - (cx + 0.5 * a), pos[:, 1] - cy)
    w[np.argmin(d + 10 * (w > phases[2] * 1.01))] = 0.0
    k = np.argmin(np.hypot(pos[:, 0] - (cx - 2 * a), pos[:, 1] - (cy + 2 * a)))
    pos[k] += (0.04, 0.03)
    crys = _splat((n, n), pixel, pos, w)
    crys = _gauss_blur_fourier(crys, pixel, sigma_col) * (2 * np.pi * sigma_col ** 2) / pixel ** 2
    # amorphous carbon film
    na = int(carbon_density * L * L)
    cpos = rng.uniform(0, L, size=(na, 2))
    carb = _splat((n, n), pixel, cpos, np.full(na, carbon_phase))
    sig_c = 0.05
    carb = _gauss_blur_fourier(carb, pixel, sig_c) * (2 * np.pi * sig_c ** 2) / pixel ** 2
    phase = crys + carb
    amp = np.exp(-absorption * phase)
    return amp * np.exp(1j * phase)


def mr_object(n=256, pixel=1.0, voltage_V=300e3, seed=2, V_cube=13.0, V_sphere=9.0,
              V_core=20.0, V_film=10.0, film_nm=3.0, imfp_nm=150.0):
    """Medium-resolution test object; phases from the mean inner potential."""
    rng = np.random.default_rng(seed)
    CE = interaction_constant(voltage_V)
    y, x = (np.mgrid[0:n, 0:n] + 0.5) * pixel
    L = n * pixel
    t_V = np.zeros((n, n))       # integral V dz  (V nm)
    t = np.zeros((n, n))         # thickness (nm)
    # cube (edge e), viewed along <100>, slightly rotated
    e = 0.24 * L
    th = np.deg2rad(12)
    xc, yc = 0.30 * L, 0.36 * L
    xr = (x - xc) * np.cos(th) + (y - yc) * np.sin(th)
    yr = -(x - xc) * np.sin(th) + (y - yc) * np.cos(th)
    cube = (np.abs(xr) < e / 2) & (np.abs(yr) < e / 2)
    t_V += cube * e * V_cube
    t += cube * e
    # carbon sphere with metal core
    R, Rc = 0.13 * L, 0.05 * L
    xs, ys = 0.68 * L, 0.62 * L
    r2 = (x - xs) ** 2 + (y - ys) ** 2
    ts = 2 * np.sqrt(np.maximum(R ** 2 - r2, 0))
    tc = 2 * np.sqrt(np.maximum(Rc ** 2 - ((x - xs - 0.02 * L) ** 2 + (y - ys + 0.01 * L) ** 2), 0))
    t_V += (ts - tc) * V_sphere + tc * V_core
    t += ts
    # small particles
    for _ in range(6):
        r0 = rng.uniform(0.012, 0.025) * L
        x0, y0 = rng.uniform(0.15, 0.85, 2) * L
        tt = 2 * np.sqrt(np.maximum(r0 ** 2 - ((x - x0) ** 2 + (y - y0) ** 2), 0))
        t_V += tt * 15.0
        t += tt
    # weak amorphous film: correlated noise
    film = gaussian_filter(rng.standard_normal((n, n)), 2.0 / pixel, mode="wrap")
    film = film / film.std() * 0.3 + 1
    t_V += film * film_nm * V_film
    t += film * film_nm
    # band-limit edges
    t_V = gaussian_filter(t_V, 0.8, mode="wrap")
    t = gaussian_filter(t, 0.8, mode="wrap")
    phase = CE * t_V
    amp = np.exp(-t / (2 * imfp_nm))
    return amp * np.exp(1j * phase)


def simulate_focal_series(wave, params: ImagingParameters, dose_per_image=None, seed=0,
                          n_defocus=9, n_tilt=5, mtf=True, background=0.0):
    """Simulate a partially coherent focal series.

    dose_per_image : electrons per pixel per image (mean over the vacuum-normalized
        intensity); ``None`` returns noise-free intensities (normalized, mean ~1).
    background : fraction of an incoherent, defocus-independent background (e.g. inelastic)
    Returns (stack, noise_free_stack).
    """
    p = params.copy()
    if mtf and p.mtf is None:
        p.mtf = {"type": "sinc"}
    fm = ForwardModel(p, wave.shape, "incoherent", n_defocus=n_defocus, n_tilt=n_tilt)
    I = fm.forward(wave)
    if background:
        I = (1 - background) * I + background * np.mean(np.abs(wave) ** 2)
    if dose_per_image is None:
        return I, I
    rng = np.random.default_rng(seed)
    counts = rng.poisson(np.maximum(I, 0) * dose_per_image).astype(float)
    return counts, I


def dose_per_pixel(total_dose_e_per_A2, pixel_nm, n_images):
    """Electrons per pixel per image for a total dose (e/A^2) split over the series."""
    return total_dose_e_per_A2 * (pixel_nm * 10) ** 2 / n_images


def hr_parameters(n_images=16, step=2.0, start=-16.0, **kw):
    d = dict(voltage_V=300e3, pixel_size_nm=0.04,
             defoci_nm=start + step * np.arange(n_images),
             aberrations={"c_3_0": 1.0e4, "c_1_2_a": 1.0, "c_1_2_b": 0.6, "c_2_1_a": 40.0},
             focal_spread_nm=2.5, convergence_rad=0.1e-3)
    d.update(kw)
    return ImagingParameters(**d)


def mr_parameters(**kw):
    df = np.array([-8000, -4000, -2000, -1000, -500, -250, 0, 250, 500, 1000, 2000, 4000, 8000.])
    d = dict(voltage_V=300e3, pixel_size_nm=1.0, defoci_nm=df,
             aberrations={"c_3_0": 1.2e6}, focal_spread_nm=5.0, convergence_rad=5e-5)
    d.update(kw)
    return ImagingParameters(**d)


# ----------------------------------------------------------------------------- DIH test object
def dih_object(n=256, pixel=0.25, g_nm=1 / 0.2593, n_wells=4, well_nm=2.7, period_nm=12.0,
               eps_well=0.025, fluct=0.15, interface_nm=0.3, amp=0.45, mip_step_rad=0.10,
               seed=3, voltage_V=200e3):
    """Dark-field exit wave of a (0002) reflection for an InGaN/GaN-like multi-quantum-well stack.

    Growth direction along x (= direction of g).  The wells have a tetragonal strain eps_well
    relative to GaN with laterally fluctuating composition (relative rms ``fluct``) and a
    smooth strain relaxation towards a free surface on one side.  The geometric phase is
    computed relative to the *average* lattice so that the wave is periodic; the strain relative
    to the GaN barriers is recovered by subtracting the barrier (reference) mean.
    Returns (wave, eps_xx_true_relative_to_GaN, barrier_mask, g_vec).
    """
    rng = np.random.default_rng(seed)
    y, x = (np.mgrid[0:n, 0:n] + 0.5) * pixel
    L = n * pixel
    prof = np.zeros((n, n))
    x0 = 0.5 * L - 0.5 * (n_wells - 1) * period_nm
    for k in range(n_wells):
        xc = x0 + k * period_nm
        prof += 0.5 * (np.tanh((x - xc + well_nm / 2) / interface_nm) - np.tanh((x - xc - well_nm / 2) / interface_nm))
    comp = gaussian_filter(rng.standard_normal((n, n)), 1.5 / pixel, mode="wrap")
    comp = 1 + fluct * comp / comp.std()
    relax = 1 - 0.5 * np.exp(-((y - 0.0) % L) / (0.15 * L)) - 0.5 * np.exp(-((L - y) % L) / (0.15 * L))
    eps = eps_well * prof * comp * relax                      # strain relative to GaN
    barrier = prof < 0.02
    eps_rel_mean = eps - eps.mean(axis=1, keepdims=True)      # relative to the average lattice (per row)
    # displacement u_x = integral of eps_rel_mean dx (periodic; spectral integration along x)
    F = np.fft.fft(eps_rel_mean, axis=1)
    kx = np.fft.fftfreq(n, d=pixel)
    with np.errstate(divide="ignore", invalid="ignore"):
        U = np.where(kx[None, :] != 0, F / (2j * np.pi * kx[None, :]), 0)
    u = np.real(np.fft.ifft(U, axis=1))
    phase = -2 * np.pi * g_nm * u + mip_step_rad * prof
    amplitude = amp * (1 - 0.15 * prof) * (1 + 0.03 * gaussian_filter(rng.standard_normal((n, n)), 3, mode="wrap"))
    wave = amplitude * np.exp(1j * phase)
    eps_true = eps_rel_mean - eps_rel_mean[barrier].mean()   # = d u_x/dx, referenced to the GaN barriers
    return wave, eps_true, barrier, (0.0, g_nm)


def dih_parameters(**kw):
    """Dark-field focal series: 200 kV, 0.25 nm pixels, 10 um objective aperture (~0.8 nm
    resolution), geometric defocus series up to +-4 um, illumination 0.04 mrad."""
    df = np.array([-4000, -2000, -1000, -500, -200, -100, 0, 100, 200, 500, 1000, 2000, 4000.])
    d = dict(voltage_V=200e3, pixel_size_nm=0.25, defoci_nm=df, aberrations={"c_3_0": 1.0e6},
             focal_spread_nm=4.0, convergence_rad=4e-5, aperture_rad=3.1e-3)
    d.update(kw)
    return ImagingParameters(**d)


# ------------------------------------------------------------------- random MR objects (training)
def random_mr_object(n=128, pixel=1.0, voltage_V=300e3, rng=None, max_objects=6):
    """Random medium-resolution test object (cubes/plates, spheres with cores, small particles
    on a thin amorphous film) for training learned reconstruction methods.
    Returns (wave, unwrapped_phase)."""
    rng = np.random.default_rng(rng)
    CE = interaction_constant(voltage_V)
    y, x = (np.mgrid[0:n, 0:n] + 0.5) * pixel
    L = n * pixel
    tV = np.zeros((n, n))
    t = np.zeros((n, n))
    for _ in range(rng.integers(1, max_objects + 1)):
        kind = rng.choice(["cube", "sphere", "particle"], p=[0.35, 0.35, 0.3])
        xc, yc = rng.uniform(0.1, 0.9, 2) * L
        V = rng.uniform(8, 20)
        if kind == "cube":
            e = rng.uniform(0.08, 0.35) * L
            th = rng.uniform(0, np.pi)
            xr = (x - xc) * np.cos(th) + (y - yc) * np.sin(th)
            yr = -(x - xc) * np.sin(th) + (y - yc) * np.cos(th)
            m = (np.abs(xr) < e / 2) & (np.abs(yr) < e * rng.uniform(0.3, 0.5))
            depth = e * rng.uniform(0.3, 1.0)
            tV += m * depth * V
            t += m * depth
        else:
            R = rng.uniform(0.03, 0.15) * L if kind == "sphere" else rng.uniform(0.01, 0.03) * L
            tt = 2 * np.sqrt(np.maximum(R ** 2 - ((x - xc) ** 2 + (y - yc) ** 2), 0))
            tV += tt * V
            t += tt
            if kind == "sphere" and rng.random() < 0.5:
                Rc = R * rng.uniform(0.2, 0.5)
                tc = 2 * np.sqrt(np.maximum(Rc ** 2 - ((x - xc) ** 2 + (y - yc) ** 2), 0))
                tV += tc * rng.uniform(5, 15)
    film = gaussian_filter(rng.standard_normal((n, n)), rng.uniform(1, 4) / pixel, mode="wrap")
    fthick = rng.uniform(0, 5)
    film = (film / (film.std() + 1e-9) * 0.3 + 1) * fthick
    tV += film * 10
    t += film
    tV = gaussian_filter(tV, 0.8, mode="wrap")
    t = gaussian_filter(t, 0.8, mode="wrap")
    phase = CE * tV
    amp = np.exp(-t / (2 * rng.uniform(100, 250)))
    return amp * np.exp(1j * phase), phase
