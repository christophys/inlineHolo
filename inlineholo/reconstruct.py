# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Focal-series (inline holography) reconstruction algorithms.

Use :func:`reconstruct` with ``method=`` one of the keys of :data:`METHODS`.
All methods return the complex exit wave in the object plane (C1 = 0, all
aberrations given in the parameters removed), normalized such that the vacuum
(or the mean) intensity is ~1.

Categories (see METHODS[...]["category"]):

* ``direct-linear``  : TIE (uniform / full / higher-order), parabola method (PAM),
                        Schiske-type linear Wiener filter.
* ``projection``      : Gerchberg-Saxton/Misell sequential projections, IWFR
                        (iterative wave-function reconstruction, Allen et al.).
* ``projection``      : FRWR (full port of C. T. Koch's FRWR3 program, see
                        :mod:`inlineholo.frwr3`; flux-preserving projections with
                        progressive plane inclusion, alignment, distortion/defocus/alpha
                        fitting, gradient flipping and phase prediction = GP-FRWR)
* ``optimization``    : flux-preserving least-squares gradient descent (``frwr_gd``,
                        ``gpfrwr_gd``), MAL-type maximum-likelihood
                        (Gaussian noise, quasi-coherent TCC approximation),
                        Poisson maximum-likelihood with rigorous partial coherence
                        (incoherent sum, "full TCC"), and a generic gradient solver.

Except for FRWR (a loop-by-loop port of the FRWR3 C++ code, validated against it),
the implementations follow the published principles; they are compact
re-implementations for didactic and benchmarking purposes, not the original
programs (e.g. TrueImage/MAL, IWFR).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .forward import ForwardModel, fft2, ifft2, normalize_stack
from .optics import FourierGrid
from .params import ImagingParameters


@dataclass
class Reconstruction:
    wave: np.ndarray
    method: str
    options: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    runtime_s: float = 0.0
    background: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def phase(self):
        return np.angle(self.wave)

    @property
    def amplitude(self):
        return np.abs(self.wave)


# ----------------------------------------------------------------------------- helpers
def _flip(a):
    """a(-q) on an FFT grid."""
    return np.roll(a[::-1, ::-1], 1, axis=(0, 1))


def _inv_laplace(f, grid: FourierGrid, q_reg: float):
    F = fft2(f)
    den = 4 * np.pi ** 2 * (grid.q2 + q_reg ** 2)
    den[0, 0] = np.inf
    return np.real(ifft2(-F / den))


def _grad(f, grid):
    F = fft2(f)
    return (np.real(ifft2(2j * np.pi * grid.qx * F)), np.real(ifft2(2j * np.pi * grid.qy * F)))


def _div(fx, fy, grid):
    return np.real(ifft2(2j * np.pi * (grid.qx * fft2(fx) + grid.qy * fft2(fy))))


def _initial_wave(data, params, init, shape, **kw):
    if isinstance(init, np.ndarray):
        return init.astype(complex)
    i0 = int(np.argmin(np.abs(params.defoci_nm)))
    if init == "flat":
        return np.sqrt(np.mean(data[i0])) * np.ones(shape, complex)
    if init == "backprop":
        fm = ForwardModel(params.subset([i0]), shape, "coherent", use_mtf=False)
        T, _ = fm.nodes[0][0]
        return ifft2(fft2(np.sqrt(np.maximum(data[i0], 0))) * np.conj(T))
    if init in METHODS:
        allowed = ("q_reg", "pad", "max_dz", "order") if init.startswith("tie") else ("reg", "counts")
        sub = {k: v for k, v in kw.items() if k in allowed}
        return METHODS[init]["func"](data, params, **sub).wave
    raise ValueError(f"unknown init {init!r}")


# ------------------------------------------------------------------------------- TIE
def _tie_derivative(data, z, z0=0.0, order=1, max_dz=None, nearest=True):
    z = np.asarray(z, float) - z0
    idx = np.arange(len(z))
    if max_dz is not None:
        idx = idx[np.abs(z) <= max_dz]
    if nearest:
        neg = idx[z[idx] < 0]
        pos = idx[z[idx] > 0]
        sel = []
        if len(neg):
            sel.append(neg[np.argmax(z[neg])])
        sel += list(idx[z[idx] == 0])
        if len(pos):
            sel.append(pos[np.argmin(z[pos])])
        if len(sel) < 2:
            sel = list(idx[np.argsort(np.abs(z[idx]))[:2]])
        idx = np.array(sorted(sel))
        order = min(len(idx) - 1, 2)
    order = int(min(order, len(idx) - 1))
    V = np.vander(z[idx], order + 1, increasing=True)
    coef = np.linalg.pinv(V) @ data[idx].reshape(len(idx), -1)
    shp = data.shape[1:]
    return coef[0].reshape(shp), coef[1].reshape(shp), idx


def _mirror(a):
    return np.block([[a, a[:, ::-1]], [a[::-1, :], a[::-1, ::-1]]])


def tie(data, params, full=False, order=1, nearest=True, max_dz=None, q_reg=None, pad=False):
    """Transport-of-intensity equation (Teague 1983; Gureyev & Nugent 1997).

    I0 and dI/dz at z=0 from a polynomial fit through the selected planes
    (nearest planes on either side for ``nearest=True``; all planes up to |z|<=max_dz
    with a polynomial of degree ``order`` otherwise, cf. Waller et al. 2010).
    ``full=True`` solves div(I grad phi) = -k dI/dz with Teague's auxiliary function,
    otherwise uniform intensity is assumed.  ``q_reg`` (1/nm) is a Tikhonov
    regularization of the inverse Laplacian (low-frequency cut-off).
    """
    lam = params.wavelength_nm
    I0, dIdz, used = _tie_derivative(data, params.defoci_nm, 0.0, order, max_dz, nearest)
    ny, nx = I0.shape
    if pad:
        I0p, dIp = _mirror(I0), _mirror(dIdz)
    else:
        I0p, dIp = I0, dIdz
    grid = FourierGrid(I0p.shape, params.pixel_size_nm)
    if q_reg is None:
        q_reg = 0.5 / (max(I0p.shape) * params.pixel_size_nm)
    k = 2 * np.pi / lam
    Im = max(float(np.mean(I0p)), 1e-6)
    if not full:
        phi = -k * _inv_laplace(dIp / Im, grid, q_reg)
    else:
        aux = -k * _inv_laplace(dIp, grid, q_reg)
        gx, gy = _grad(aux, grid)
        Ireg = np.maximum(I0p, 0.05 * Im)
        phi = _inv_laplace(_div(gx / Ireg, gy / Ireg, grid), grid, q_reg)
    phi = phi[:ny, :nx]
    wave = np.sqrt(np.maximum(I0, 0)) * np.exp(1j * phi)
    return Reconstruction(wave, "tie_full" if full else "tie",
                          dict(full=full, order=order, planes=used.tolist(), q_reg=q_reg, pad=pad))


# -------------------------------------------------------------------- linear methods
def _linear_transfer(params, shape, coherence=True):
    fm = ForwardModel(params, shape, "envelope" if coherence else "coherent", use_mtf=True)
    T = np.stack([n[0][0] for n in fm.nodes])
    if fm.mtf is not None:
        T = T * fm.mtf[None]
    return T


def _auto_reg(reg, counts, signal_var=0.01):
    """Wiener regularization: noise/signal power ratio ~ 1 / (counts * signal variance)."""
    if reg != "auto":
        return float(reg)
    if not counts:
        return 1e-3
    return float(np.clip(1.0 / (counts * signal_var), 1e-4, 10.0))


def pam(data, params, reg="auto", coherence=True, counts=None):
    """Parabola method / focus-variation linear restoration (Op de Beeck et al. 1996).

    Linear-imaging estimate of the scattered wave, ignoring the conjugate ('twin')
    term which averages out over a sufficiently large focus range:
    S(q) = sum_n conj(T_n) J_n / (sum_n |T_n|^2 + N reg).
    """
    N = len(data)
    reg = _auto_reg(reg, counts)
    J = np.stack([fft2(d - d.mean()) for d in data])
    T = _linear_transfer(params, data.shape[1:], coherence)
    S = np.sum(np.conj(T) * J, 0) / (np.sum(np.abs(T) ** 2, 0) + N * reg)
    S[0, 0] = 0
    return Reconstruction(1 + ifft2(S), "pam", dict(reg=reg))


def schiske(data, params, reg="auto", coherence=True, counts=None):
    """Schiske-type linear least-squares (Wiener) focal-series restoration.

    Solves, for every pair (q, -q), the 2x2 normal equations of
    J_n(q) = T_n(q) S(q) + conj(T_n(-q)) conj(S(-q))   (Schiske 1968/1973; Saxton 1994),
    i.e. it explicitly separates wave and twin image.
    """
    N = len(data)
    reg = _auto_reg(reg, counts)
    J = np.stack([fft2(d - d.mean()) for d in data])
    T = _linear_transfer(params, data.shape[1:], coherence)
    Tm = np.stack([_flip(t) for t in T])
    C = np.conj(Tm)
    A11 = np.sum(np.abs(T) ** 2, 0) + N * reg
    A22 = np.sum(np.abs(C) ** 2, 0) + N * reg
    A12 = np.sum(np.conj(T) * C, 0)
    r1 = np.sum(np.conj(T) * J, 0)
    r2 = np.sum(np.conj(C) * J, 0)
    det = A11 * A22 - np.abs(A12) ** 2
    S = (A22 * r1 - A12 * r2) / det
    S[0, 0] = 0
    return Reconstruction(1 + ifft2(S), "schiske", dict(reg=reg))


# --------------------------------------------------------------- projection methods
def _projection(data, params, n_iter, sequential, init, shape, beta=1.0, callback=None, **kw):
    fm = ForwardModel(params, shape, "coherent", use_mtf=False)
    T = [n[0][0] for n in fm.nodes]
    amp = [np.sqrt(np.maximum(d, 0)) for d in data]
    psi = _initial_wave(data, params, init, shape, **kw)
    Psi = fft2(psi)
    hist = []
    order = np.argsort(np.abs(params.defoci_nm))
    for it in range(n_iter):
        err = 0.0
        if sequential:
            for n in order:
                f = ifft2(Psi * T[n])
                a = np.abs(f)
                err += np.mean((a - amp[n]) ** 2)
                f2 = amp[n] * np.exp(1j * np.angle(f))
                Psi = Psi + beta * (fft2(f2) * np.conj(T[n]) - Psi)
        else:
            acc = np.zeros_like(Psi)
            for n in range(len(T)):
                f = ifft2(Psi * T[n])
                a = np.abs(f)
                err += np.mean((a - amp[n]) ** 2)
                f2 = amp[n] * np.exp(1j * np.angle(f))
                acc += fft2(f2) * np.conj(T[n])
            Psi = Psi + beta * (acc / len(T) - Psi)
        hist.append(err / len(T))
        if callback:
            callback(it, ifft2(Psi), hist[-1])
    return ifft2(Psi), hist


def gerchberg_saxton(data, params, n_iter=100, init="backprop", callback=None, **kw):
    """Sequential Gerchberg-Saxton / Misell projections through all planes (coherent)."""
    t = time.time()
    psi, h = _projection(data, params, n_iter, True, init, data.shape[1:], callback=callback, **kw)
    return Reconstruction(psi, "gs", dict(n_iter=n_iter, init=str(init)), h, time.time() - t)


def iwfr(data, params, n_iter=100, init="backprop", callback=None, **kw):
    """Iterative wave-function reconstruction (Allen & Oxley 2001; Allen et al. 2004):
    parallel amplitude projections in all planes, back-propagated and averaged."""
    t = time.time()
    psi, h = _projection(data, params, n_iter, False, init, data.shape[1:], callback=callback, **kw)
    return Reconstruction(psi, "iwfr", dict(n_iter=n_iter, init=str(init)), h, time.time() - t)


# ------------------------------------------------------------- optimization methods
def _tv_grad(psi, eps):
    dx = np.roll(psi, -1, 1) - psi
    dy = np.roll(psi, -1, 0) - psi
    mag = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2 + eps ** 2)
    R = np.mean(mag)
    px, py = dx / mag, dy / mag
    div = (px - np.roll(px, 1, 1)) + (py - np.roll(py, 1, 0))
    return R, -2 * div / psi.size


def _regularizer(psi, tv=0.0, tikhonov=0.0, tv_eps=1e-3):
    R, g = 0.0, np.zeros_like(psi)
    if tv:
        r, gg = _tv_grad(psi, tv_eps)
        R += tv * r
        g += tv * gg
    if tikhonov:
        dx = np.roll(psi, -1, 1) - psi
        dy = np.roll(psi, -1, 0) - psi
        R += tikhonov * np.mean(np.abs(dx) ** 2 + np.abs(dy) ** 2)
        lap = (np.roll(psi, 1, 1) + np.roll(psi, -1, 1) + np.roll(psi, 1, 0) + np.roll(psi, -1, 0) - 4 * psi)
        g += tikhonov * (-2 * lap) * 2 / psi.size
    return R, g


def gradient_solver(data, params, model="envelope", loss="lsq", n_iter=100, init="flat",
                    optimizer="lbfgs", tv=0.0, tikhonov=0.0, n_defocus=5, n_tilt=3,
                    weights=None, callback=None, name="gd", **kw):
    """Generic gradient-based (maximum-likelihood / least-squares) reconstruction.

    model: 'coherent' | 'envelope' | 'blur' | 'incoherent'  (see forward.ForwardModel)
    loss : 'lsq' (Gaussian) | 'amplitude' | 'poisson'
    optimizer: 'lbfgs' (scipy L-BFGS-B) or 'gd' (steepest descent with adaptive step)
    """
    t0 = time.time()
    shape = data.shape[1:]
    fm = ForwardModel(params, shape, model, n_defocus=n_defocus, n_tilt=n_tilt)
    psi0 = _initial_wave(data, params, init, shape, **kw)
    npix = psi0.size
    hist = []

    def fg(psi):
        L, g = fm.loss_and_grad(psi, data, loss, weights)
        R, gr = _regularizer(psi, tv, tikhonov)
        return (L + R) * npix, (g + gr) * npix

    if optimizer == "lbfgs":
        def fun(x):
            psi = x[:npix].reshape(shape) + 1j * x[npix:].reshape(shape)
            f, g = fg(psi)
            return f, np.concatenate([g.real.ravel(), g.imag.ravel()])

        def cb(xk):
            psi = xk[:npix].reshape(shape) + 1j * xk[npix:].reshape(shape)
            hist.append(fg(psi)[0] / npix)
            if callback:
                callback(len(hist), psi, hist[-1])

        x0 = np.concatenate([psi0.real.ravel(), psi0.imag.ravel()])
        res = minimize(fun, x0, jac=True, method="L-BFGS-B", callback=cb,
                       options=dict(maxiter=n_iter, maxfun=int(n_iter * 1.5) + 10,
                                    ftol=1e-15, gtol=1e-12, maxcor=10))
        psi = res.x[:npix].reshape(shape) + 1j * res.x[npix:].reshape(shape)
    else:
        psi, hist = _gd_loop(fg, psi0, n_iter, params.n_images, callback)
    opts = dict(model=model, loss=loss, n_iter=n_iter, init=str(init), optimizer=optimizer,
                tv=tv, tikhonov=tikhonov)
    return Reconstruction(psi, name, opts, hist, time.time() - t0)


def _gd_loop(fg, psi, n_iter, n_img, callback=None, step=0.5, hook=None):
    """Steepest descent with adaptive step size (increase on success, halve on failure)."""
    npix = psi.size
    f, g = fg(psi)
    hist = [f / npix]
    for it in range(n_iter):
        d = -g / 2 / n_img                      # ~ mean over planes of back-propagated residuals
        for _ in range(8):
            trial = psi + step * d
            f2, g2 = fg(trial)
            if f2 <= f:
                step = min(step * 1.25, 4.0)
                break
            step *= 0.5
        else:
            break
        psi, f, g = trial, f2, g2
        if hook is not None:
            psi2 = hook(it, psi)
            if psi2 is not None:
                psi = psi2
                f, g = fg(psi)
        hist.append(f / npix)
        if callback:
            callback(it, psi, hist[-1])
    return psi, hist


def _pad_series(data, factor):
    """Pad a focal series (centred) with the mean of each image; returns (padded, mask, slices)."""
    n, ny, nx = data.shape
    Py, Px = int(round(ny * factor)), int(round(nx * factor))
    Py += Py % 2
    Px += Px % 2
    y0, x0 = (Py - ny) // 2, (Px - nx) // 2
    out = np.empty((n, Py, Px))
    for k in range(n):
        out[k] = data[k].mean()
        out[k, y0:y0 + ny, x0:x0 + nx] = data[k]
    mask = np.zeros((Py, Px))
    mask[y0:y0 + ny, x0:x0 + nx] = 1.0
    return out, mask, (slice(y0, y0 + ny), slice(x0, x0 + nx))


def frwr(data, params, n_iter=150, init="flat", background=False, bg_damping=0.98,
         pad=1.0, callback=None, hook=None, name="frwr", **kw):
    """FRWR-type flux-preserving non-linear reconstruction (Koch 2008, 2014).

    Least-squares fit of I_n = G_n (x) |psi_n|^2 (+ I_bg) where partial spatial coherence
    is a defocus-dependent Gaussian blur of the intensity and temporal coherence a
    damping envelope on the wave; steepest descent with adaptive step size.
    ``background=True`` additionally estimates a shared, positive incoherent background
    image: after every iteration the mean residual is added to it and it is multiplied by
    ``bg_damping`` (Ozsoy-Keskinbora et al. 2022).
    ``pad`` > 1 reconstructs the wave on an array ``pad`` times larger than the field of
    view, with the data constraint applied only inside the field of view, so that the
    measured area need not obey periodic boundary conditions.
    """
    t0 = time.time()
    fov = None
    if pad and pad > 1.0:
        data_p, mask, fov = _pad_series(data, pad)
    else:
        data_p, mask = data, None
    shape = data_p.shape[1:]
    fm = ForwardModel(params, shape, "blur")
    if isinstance(init, np.ndarray) and init.shape != shape and fov is not None:
        w = np.full(shape, np.mean(np.abs(init)), complex)
        w[fov] = init
        init = w
    psi0 = _initial_wave(data_p, params, init, shape, **kw)
    state = dict(bg=np.zeros(shape) if background else None)
    npix = psi0.size if mask is None else mask.sum()

    def fg(psi):
        L, g = fm.loss_and_grad(psi, data_p, "lsq", background=state["bg"], mask=mask)
        return L * npix, g * npix

    def _hook(it, psi):
        if background:
            I = fm.forward(psi) + state["bg"]
            state["bg"] = bg_damping * np.maximum(state["bg"] + np.mean(data_p - I, 0), 0)
        if hook is not None:
            return hook(it, psi, fm, data_p, mask)
        return None

    psi, hist = _gd_loop(fg, psi0, n_iter, params.n_images, callback, hook=_hook)
    bg = state["bg"]
    if fov is not None:
        psi = psi[fov]
        bg = bg[fov] if bg is not None else None
    return Reconstruction(psi, name, dict(n_iter=n_iter, init=str(init)[:20], background=background, pad=pad),
                          hist, time.time() - t0, bg)


def _soft_limit(d, phi_max):
    """Exponential compression of |d| above phi_max/2 towards phi_max (Ozsoy-Keskinbora 2022)."""
    h = phi_max / 2
    a = np.abs(d)
    return np.sign(d) * np.where(a >= h, phi_max - h * np.exp(-(a - h) / h), a)


def _phase_gradient(psi, grid):
    """Phase gradient (rad/nm) computed from the complex wave -> free of phase wrapping."""
    F = fft2(psi)
    dx = ifft2(2j * np.pi * grid.qx * F)
    dy = ifft2(2j * np.pi * grid.qy * F)
    a2 = np.maximum(np.abs(psi) ** 2, 1e-8 * np.max(np.abs(psi)) ** 2 + 1e-12)
    return np.imag(np.conj(psi) * dx) / a2, np.imag(np.conj(psi) * dy) / a2


def _tie_full_inverse(dIdz, I0, grid, k, q_reg, Ifloor=0.05):
    """phi = -k lap^-1 div[ I0^-1 grad lap^-1 dI/dz ]  (Teague's solution)."""
    aux = _inv_laplace(dIdz, grid, q_reg)
    gx, gy = _grad(aux, grid)
    Ir = np.maximum(I0, Ifloor * np.mean(I0))
    return -k * _inv_laplace(_div(gx / Ir, gy / Ir, grid), grid, q_reg)


def gpfrwr(data, params, n_iter=150, init="flat", every=3, beta=-0.97, zeta=1.0,
           flip_fraction=None, r_c_nm=None, r_c_px=None, phi_max=0.2, prediction="inverse_defocus",
           damping=1.0, gradient_flipping=True, phase_prediction=True, n_start=5, n_final=10,
           background=False, pad=1.0, callback=None, **kw):
    """FRWR with gradient flipping and phase prediction (GP-FRWR; Ozsoy-Keskinbora et al. 2022).

    Every ``every`` iterations (after ``n_start`` and except for the last ``n_final``):

    * gradient flipping: inside the field of view, the components of the phase gradient
      whose magnitude is below delta = zeta * std(G) (or the smallest ``flip_fraction``)
      are multiplied by ``beta`` (= -0.97), and the phase is updated as
      phi'(q) = phi(q) [1 - exp(-r_c^2 q^2)] + FT[div G'] exp(-r_c^2 q^2) / (-q^2)
      (q angular), i.e. only length scales larger than r_c are affected;
    * phase prediction: TIE-type update from the intensity residuals of all planes,
      dphi = (1/N) sum_n TIE^-1[(I_exp,n - I_sim,n)] / C1_n  (``prediction='inverse_defocus'``,
      as published) or the least-squares combination sum_n C1_n dI_n / sum C1_n^2
      (``prediction='lsq'``), soft-limited to ``phi_max`` by exponential compression.
    """
    lam = params.wavelength_nm
    k = 2 * np.pi / lam
    z = np.asarray(params.defoci_nm, float)
    px = params.pixel_size_nm
    if r_c_nm is not None:
        rc = r_c_nm
    elif r_c_px is not None:
        rc = r_c_px * px
    else:   # default: 10 % of the field of view
        rc = 0.1 * max(data.shape[1:]) * px
    cache = {}

    def hook(it, psi, fm, data_p, mask):
        if it < n_start or it >= n_iter - n_final or (it - n_start) % every:
            return None
        shape = psi.shape
        if "grid" not in cache:
            g = FourierGrid(shape, px)
            cache["grid"] = g
            cache["taper"] = np.exp(-(2 * np.pi * rc) ** 2 * g.q2)
            cache["q_reg"] = 0.5 / (max(shape) * px)
        grid, taper, q_reg = cache["grid"], cache["taper"], cache["q_reg"]
        m = np.ones(shape, bool) if mask is None else mask > 0
        dphi = np.zeros(shape)
        if gradient_flipping:
            gx, gy = _phase_gradient(psi, grid)
            comps = np.concatenate([gx[m], gy[m]])
            if flip_fraction is not None:
                thr = np.quantile(np.abs(comps), flip_fraction)
            else:
                thr = zeta * np.std(comps)
            fx = m & (np.abs(gx) <= thr)
            fy = m & (np.abs(gy) <= thr)
            gx2 = np.where(fx, beta * gx, gx)
            gy2 = np.where(fy, beta * gy, gy)
            d = _inv_laplace(_div(gx2 - gx, gy2 - gy, grid), grid, q_reg)
            dphi += np.real(ifft2(fft2(d) * taper))
        if phase_prediction:
            I = fm.forward(psi)
            res = (data_p - I) * (m[None] if mask is not None else 1.0)
            I0 = np.abs(psi) ** 2
            nz = np.abs(z) > 1e-9
            if prediction == "lsq":
                num = np.tensordot(z[nz], res[nz], axes=(0, 0)) / np.sum(z[nz] ** 2)
                pred = _tie_full_inverse(num, I0, grid, k, q_reg)
            else:
                pred = np.zeros(shape)
                for n in np.where(nz)[0]:
                    pred += _tie_full_inverse(res[n] / z[n], I0, grid, k, q_reg)
                pred /= max(nz.sum(), 1)
            dphi += _soft_limit(damping * pred, phi_max)
        return psi * np.exp(1j * dphi)

    r = frwr(data, params, n_iter=n_iter, init=init, background=background, pad=pad,
             callback=callback, hook=hook, name="gpfrwr", **kw)
    r.options.update(every=every, beta=beta, zeta=zeta, flip_fraction=flip_fraction, r_c_nm=rc,
                     phi_max=phi_max, prediction=prediction, damping=damping,
                     gradient_flipping=gradient_flipping, phase_prediction=phase_prediction)
    return r


def _mtf_to_frwr3(mtf):
    """Translate ImagingParameters.mtf into the frwr3 'MTF:' line (Gaussian approximation)."""
    if not mtf:
        return [0, 0, 0, 0, 0]
    kind = mtf.get("type", "sinc")
    if kind == "sinc":                      # sinc(u) ~ exp(-pi^2 u^2 / 6)
        return [1, 0.0, 0.0, 1.0, np.pi ** 2 / 6]
    if kind == "gauss":
        s = mtf.get("sigma_px", 0.5)
        return [1, 0.0, 0.0, 1.0, 2 * np.pi ** 2 * s ** 2]
    return [0, 0, 0, 0, 0]


def frwr3_method(data, params, n_iter=150, pad=1.25, init="flat", background=False,
                 gradient_flipping=False, phase_prediction=False, config=None, backend="auto",
                 precision="auto", callback=None, name="frwr", intermediate_loops=9, max_shift=0,
                 fit_defocus=0, full_aberrations=True, temporal_envelope=True, r_c_nm=None,
                 phi_max=0.2, gf_beta=0.97, gf_fraction=0.72, gf_every=3, frames=None, roi=None, **kw):
    """FRWR = in-memory (GPU-capable) port of the FRWR3 program (:mod:`inlineholo.frwr3`).

    The keyword arguments set the most important FRWR3 keys for benchmarking on
    pre-aligned data; ``config`` (YAML/.conf path, text or dict with FRWR3 key names)
    overrides them.  ``n_iter`` = 'final loops'.  With ``frames``/``roi`` the full
    (larger) frames are used so that drift and distortions can be corrected.
    """
    from . import frwr3 as F
    base = {"print level": 0, "final loops": int(n_iter), "intermediate loops": int(intermediate_loops),
            "residue loops": 0, "max shift": int(max_shift), "fit defocus": int(fit_defocus),
            "padding": max(float(pad or 1.0) - 1.0, 0.0), "shift trials": 1, "backend": backend,
            "precision": precision, "aberrations": "full" if full_aberrations else "frwr3",
            "focal spread envelope": int(bool(temporal_envelope)), "MTF": _mtf_to_frwr3(params.mtf),
            "Nedge": 0 if not max_shift else 20, "Nedge for smoothing": 0 if not max_shift else 20}
    if background:
        base["fit incoherent background"] = [1, 0.1, 0.98]
    if gradient_flipping:
        rc = r_c_nm if r_c_nm is not None else 0.1 * max(data.shape[1:]) * params.pixel_size_nm
        base.update({"apply GF": 1, "GF flipping frequency": int(gf_every), "GF beta": float(gf_beta),
                     "GF signal fraction": float(gf_fraction), "GF threshold": 0.01,
                     "GF length scale (nm)": float(rc), "GF adjust fraction": 1, "GF 3rd order fraction": 0.0})
    if phase_prediction:
        base.update({"phase predict threshold": float(phi_max), "phase predict loops": 2,
                     "low intensity threshold": 0.05})
    if config is not None:
        base.update(F.load_config(config, raw_only=True))
    init_wave = None
    if isinstance(init, np.ndarray) or (isinstance(init, str) and init not in ("flat", "")):
        init_wave = _initial_wave(data, params, init, data.shape[1:], **kw)
    t0 = time.time()
    hist = []

    from .control import StopRequested

    def cb(label, j, R, eng):
        hist.append(R)
        if callback is not None and getattr(callback, "planned", None) is None:
            try:
                callback.planned = eng.planned_loops()      # total number of loops (progress bars)
                callback.engine = eng                       # work_done / planned_work() for time estimates
            except Exception:
                pass
        if callback is not None:
            try:
                callback.label = label
            except Exception:
                pass
            try:
                callback(len(hist) - 1, None, R)
            except StopRequested:
                raise
            except Exception:
                pass

    src = data if frames is None else frames
    res = F.run_frwr3(src, params, base, roi=roi, callback=cb, init_wave=init_wave)
    # global scaling: frwr3 keeps the data scale (normalized data -> |psi|^2 ~ 1)
    opts = dict(n_iter=n_iter, pad=pad, background=background, gradient_flipping=gradient_flipping,
                phase_prediction=phase_prediction, init=str(init)[:20], backend=res.backend,
                defoci_nm=res.defoci_nm.tolist())
    if getattr(res, "stopped", False):
        opts["stopped_after_loops"] = len(res.rvalues)
    r = Reconstruction(res.wave, name, opts, [x[2] for x in res.rvalues], time.time() - t0, res.background)
    r.extra = {"frwr3": res}
    return r


def _symmetric_pairs(z, tol=1e-6):
    """Indices (i_minus, i_plus, |dz|) of defocus pairs symmetric about C1 = 0."""
    pairs = []
    for i, zi in enumerate(z):
        if zi <= 0:
            continue
        j = np.where(np.abs(z + zi) <= tol * max(1.0, abs(zi)))[0]
        if len(j):
            pairs.append((int(j[0]), i, float(zi)))
    return sorted(pairs, key=lambda p: p[2])


def _butterworth_band(q, q_lo, q_hi, order=4):
    hp = 1.0 - 1.0 / (1.0 + (q / max(q_lo, 1e-12)) ** (2 * order)) if q_lo > 0 else np.ones_like(q)
    lp = 1.0 / (1.0 + (q / q_hi) ** (2 * order))
    return hp * lp


def mftie(data, params, chi_max=1.0, rho=1.0, full=True, coherence=True, order=4, pad=False,
          eps=1e-3, return_bands=False):
    """Multi-focus TIE (MFTIE) after Eljarrat, Mueller, Huang & Koch, Opt. Express 26, 11819 (2018).

    For every pair of images at +-dz a three-plane TIE partial phase is computed. The partial
    phases are combined with smooth, overlapping band-pass filters H_dz (Butterworth) that are
    normalized over the series and include the partial-spatial-coherence envelope
    S_dz(q) = exp(-2 pi^2 (alpha dz)^2 q^2) of the flux-preserving image model:

        phi(q) = sum_dz H_dz S_dz phi_dz(q) / (sum_dz H_dz S_dz^2 + eps)

    Cut-offs: q_H(dz) from the linearity of the defocus transfer, pi lambda dz q_H^2 = chi_max
    (Fresnel-Kirchhoff criterion); q_L(dz) from homogeneous noise distribution sampling (HNDS):
    q_L^-2 = rho q_m^-2 (dz/dz_max)^2 + q_H^-2 with q_m the lowest frequency of the field of
    view, i.e. every partial phase contributes the same noise-induced error. Requires image
    pairs symmetric about focus and (preferably) the in-focus image.
    """
    lam = params.wavelength_nm
    k = 2 * np.pi / lam
    z = np.asarray(params.defoci_nm, float)
    pairs = _symmetric_pairs(z)
    if not pairs:
        raise ValueError("MFTIE needs image pairs at symmetric defocus +-dz")
    i0 = np.where(np.abs(z) < 1e-9)[0]
    I0 = data[i0[0]] if len(i0) else 0.5 * (data[pairs[0][0]] + data[pairs[0][1]])
    ny, nx = I0.shape
    D = data
    if pad:
        D = np.stack([_mirror(d) for d in data])
        I0 = _mirror(I0)
    grid = FourierGrid(I0.shape, params.pixel_size_nm)
    q = grid.q
    q_m = 1.0 / (max(ny, nx) * params.pixel_size_nm)
    dz_max = pairs[-1][2]
    alpha = params.convergence_rad if coherence else 0.0
    num = np.zeros(I0.shape, complex)
    den = np.zeros(I0.shape)
    bands = []
    for im, ip, dz in pairs:
        dIdz = (D[ip] - D[im]) / (2 * dz)
        if full:
            phi = _tie_full_inverse(dIdz, I0, grid, k, q_reg=0.0)
        else:
            phi = -k * _inv_laplace(dIdz / max(I0.mean(), 1e-9), grid, 0.0)
        q_hi = np.sqrt(chi_max / (np.pi * lam * dz))
        q_lo = 1.0 / np.sqrt(rho * q_m ** -2 * (dz / dz_max) ** 2 + q_hi ** -2)
        if dz == dz_max:
            q_lo = 0.0
        H = _butterworth_band(q, q_lo, q_hi, order)
        S = np.exp(-2 * np.pi ** 2 * (alpha * dz) ** 2 * grid.q2)
        num += H * S * fft2(phi)
        den += H * S ** 2
        bands.append((dz, q_lo, q_hi))
    Phi = num / (den + eps)
    Phi[0, 0] = 0
    phi = np.real(ifft2(Phi))[:ny, :nx]
    I0 = I0[:ny, :nx]
    rec = Reconstruction(np.sqrt(np.maximum(I0, 0)) * np.exp(1j * phi), "mftie",
                         dict(chi_max=chi_max, rho=rho, full=full, coherence=coherence, bands=bands))
    return rec


def gftie(data, params, n_iter=50, zeta=1.0, beta=0.97, R_LP=None, pad_factor=2.0, planes="auto",
          return_history=False):
    """Gradient-flipping TIE (GF-TIE) after Parvizi, Van den Broek & Koch,
    Adv. Struct. Chem. Imaging 2, 3 (2016).

    The in-focus intensity is padded with its mean and dI/dz with zeros to ``pad_factor``
    times the field of view. Iteratively: (i) phi = TIE^-1(dI/dz) (Fourier solution, periodic on
    the padded array); (ii) G = grad phi; (iii) gradient flipping: G' = G where |G|_1 > delta and
    G' = -beta G where |G|_1 <= delta, delta = zeta * std(|G|_1) (inside the field of view);
    (iv) D(G') = -div(I G')/k; (v) dI/dz is replaced by D(G') in the padding area and by
    F^-1[h F D(G') + (1-h) F dI/dz_exp] inside the field of view, with the Gaussian low-pass
    h(q) = exp(-R_LP^2 q^2) (q angular): high frequencies stay tied to the data, the low
    frequencies (and thereby the boundary conditions) are determined by the sparsity of the
    phase gradient.
    """
    lam = params.wavelength_nm
    k = 2 * np.pi / lam
    z = np.asarray(params.defoci_nm, float)
    if planes == "auto":
        # pair at which the defocus transfer pi lambda dz q^2 reaches ~1 rad at 1/4 Nyquist
        qq = 0.25 * 0.5 / params.pixel_size_nm
        planes = 1.0 / (np.pi * params.wavelength_nm * qq ** 2)
        if not _symmetric_pairs(z):
            planes = "nearest"
    if planes == "nearest":
        I0c, dIdz, _ = _tie_derivative(data, z, 0.0, 1, None, True)
    elif planes == "poly":
        I0c, dIdz, _ = _tie_derivative(data, z, 0.0, 3, None, False)
    else:   # planes = defocus value: use the symmetric pair closest to +-planes (plus focus)
        pairs = _symmetric_pairs(z)
        im, ip, dz = min(pairs, key=lambda pr: abs(pr[2] - float(planes)))
        i0 = np.where(np.abs(z) < 1e-9)[0]
        I0c = data[i0[0]] if len(i0) else 0.5 * (data[im] + data[ip])
        dIdz = (data[ip] - data[im]) / (2 * dz)
    ny, nx = I0c.shape
    Py, Px = int(round(ny * pad_factor)), int(round(nx * pad_factor))
    y0, x0 = (Py - ny) // 2, (Px - nx) // 2
    fov = (slice(y0, y0 + ny), slice(x0, x0 + nx))
    I = np.full((Py, Px), I0c.mean())
    I[fov] = I0c
    I = np.maximum(I, 0.05 * I.mean())
    d_exp = np.zeros((Py, Px))
    d_exp[fov] = dIdz
    m = np.zeros((Py, Px), bool)
    m[fov] = True
    grid = FourierGrid((Py, Px), params.pixel_size_nm)
    if R_LP is None:
        R_LP = 0.1 * min(ny, nx) * params.pixel_size_nm
    h = np.exp(-(R_LP ** 2) * (2 * np.pi) ** 2 * grid.q2)
    d = d_exp.copy()
    hist = []
    for it in range(n_iter):
        phi = _tie_full_inverse(d, I, grid, k, q_reg=0.0, Ifloor=0.0)
        gx, gy = _grad(phi, grid)
        g1 = np.abs(gx) + np.abs(gy)
        delta = zeta * np.std(g1[m])
        flip = m & (g1 <= delta)
        gx2 = np.where(flip, -beta * gx, gx)
        gy2 = np.where(flip, -beta * gy, gy)
        DG = -_div(I * gx2, I * gy2, grid) / k
        inside = np.real(ifft2(h * fft2(DG) + (1 - h) * fft2(d_exp)))
        d_new = np.where(m, inside, DG)
        hist.append(float(np.sqrt(np.mean((d_new - d) ** 2))))
        d = d_new
    phi = _tie_full_inverse(d, I, grid, k, q_reg=0.0, Ifloor=0.0)[fov]
    rec = Reconstruction(np.sqrt(np.maximum(I0c, 0)) * np.exp(1j * phi), "gftie",
                         dict(n_iter=n_iter, zeta=zeta, beta=beta, R_LP=R_LP, pad_factor=pad_factor), hist)
    return rec


def hybrid(data, params, offaxis_wave=None, refine="frwr", **kw):
    """Hybrid inline/off-axis holography (Ozsoy-Keskinbora et al. 2014; Seo et al. 2021):
    the (low-resolution, low-frequency-accurate) wave reconstructed from an off-axis hologram
    of the same area is used as the start wave of a non-linear focal-series reconstruction."""
    if offaxis_wave is None:
        raise ValueError("hybrid reconstruction needs offaxis_wave=<complex array>")
    w = np.asarray(offaxis_wave, complex)
    w = w / np.sqrt(np.mean(np.abs(w) ** 2)) * np.sqrt(np.mean(data))
    kw = {k: v for k, v in kw.items() if k not in ("counts",)}
    r = METHODS[refine]["func"](data, params, init=w, **kw)
    r.method = "hybrid"
    r.options.update(start="off-axis", refinement=refine)
    return r


def mal(data, params, n_iter=100, init="pam", callback=None, **kw):
    """MAL-type maximum-likelihood focal-series reconstruction (Coene et al. 1996;
    Thust et al. 1996): Gaussian noise (least squares), quasi-coherent (envelope) image
    model for linear and non-linear terms, started from the linear (PAM) solution,
    optimized with L-BFGS."""
    r = gradient_solver(data, params, model="envelope", loss="lsq", n_iter=n_iter, init=init,
                        callback=callback, name="mal", **kw)
    return r


def ml_poisson(data, params, n_iter=100, init="pam", n_defocus=5, n_tilt=3, callback=None, **kw):
    """Poisson maximum-likelihood with rigorous partial coherence: incoherent sum over a
    Gauss-Hermite quadrature of defocus and illumination tilts ("full TCC" for thin
    objects), cf. mixed-state ptychography (Thibault & Menzel 2013)."""
    return gradient_solver(data, params, model="incoherent", loss="poisson", n_iter=n_iter,
                           init=init, n_defocus=n_defocus, n_tilt=n_tilt, callback=callback,
                           name="ml_poisson", **kw)


# ---------------------------------------------------------------------------- registry
METHODS = {
    "tie": dict(func=lambda d, p, **k: tie(d, p, full=False, **k), category="direct-linear",
                linear=True, coherence="none", noise="none (Tikhonov)",
                description="TIE, uniform intensity, nearest planes"),
    "tie_full": dict(func=lambda d, p, **k: tie(d, p, full=True, **k), category="direct-linear",
                     linear=True, coherence="none", noise="none (Tikhonov)",
                     description="TIE with non-uniform intensity (Teague auxiliary function)"),
    "tie_poly": dict(func=lambda d, p, **k: tie(d, p, full=True, nearest=False,
                                                 order=k.pop("order", 3), **k),
                     category="direct-linear", linear=True, coherence="none", noise="LSQ fit",
                     description="higher-order TIE: polynomial fit through all planes"),
    "tie_fp": dict(func=lambda d, p, **k: __import__("inlineholo.fokker_planck", fromlist=["tie_fp"]).tie_fp(
                       d, p, **{kk: v for kk, v in k.items() if kk in ("planes", "q_reg", "return_diffusion")}),
                   category="direct-linear", linear=True, coherence="Fokker-Planck: dz^2 diffusion terms eliminated",
                   noise="none (Tikhonov)",
                   description="Fokker-Planck-consistent TIE (3 nearest planes, quadratic fit) + diffusion field"),
    "mftie": dict(func=lambda d, p, **k: mftie(d, p, **{kk: v for kk, v in k.items() if kk != "counts"}),
                  category="direct-linear", linear=True,
                  coherence="spatial envelope per defocus pair (flux-preserving blur)", noise="HNDS band filters",
                  description="multi-focus TIE with overlapping filters (Eljarrat et al. 2018)"),
    "gftie": dict(func=lambda d, p, **k: gftie(d, p, **{kk: v for kk, v in k.items() if kk != "counts"}),
                  category="direct-linear + sparsity", linear=False, coherence="none",
                  noise="gradient-sparsity prior", description="gradient-flipping TIE (Parvizi et al. 2016)"),
    "pam": dict(func=pam, category="direct-linear", linear=True, coherence="envelope (exact for linear terms)",
                noise="Wiener", description="parabola method / linear focal-series restoration"),
    "schiske": dict(func=schiske, category="direct-linear", linear=True,
                    coherence="envelope (exact for linear terms)", noise="Wiener",
                    description="Schiske linear least-squares filter with twin separation"),
    "gs": dict(func=gerchberg_saxton, category="projection", linear=False, coherence="none (coherent)",
               noise="amplitude LSQ (implicit)", description="sequential Gerchberg-Saxton/Misell"),
    "iwfr": dict(func=iwfr, category="projection", linear=False, coherence="none (coherent)",
                 noise="amplitude LSQ (implicit)", description="IWFR (Allen et al.)"),
    "frwr": dict(func=lambda d, p, **k: frwr3_method(d, p, **{kk: v for kk, v in k.items() if kk != "counts"}),
                 category="projection", linear=False,
                 coherence="flux-preserving intensity blur (spatial) + MTF (+ temporal envelope, ext.)",
                 noise="amplitude projections (implicit LSQ)",
                 description="FRWR: in-memory/GPU port of the FRWR3 program (Koch 2008, 2014)"),
    "frwr_gf": dict(func=lambda d, p, **k: frwr3_method(d, p, **{**dict(gradient_flipping=True, background=True,
                                                                     name="frwr_gf"),
                                                             **{kk: v for kk, v in k.items() if kk != "counts"}}),
                    category="projection", linear=False, coherence="as FRWR (+ incoherent background)",
                    noise="amplitude projections (implicit LSQ)",
                    description="FRWR3 with gradient flipping + incoherent background"),
    "gpfrwr": dict(func=lambda d, p, **k: frwr3_method(d, p, **{**dict(gradient_flipping=True, phase_prediction=True,
                                                                    background=True, name="gpfrwr"),
                                                            **{kk: v for kk, v in k.items() if kk != "counts"}}),
                   category="projection", linear=False, coherence="as FRWR (+ incoherent background)",
                   noise="amplitude projections (implicit LSQ)",
                   description="FRWR3 with gradient flipping + TIE phase prediction + background (GP-FRWR)"),
    "frwr_gd": dict(func=frwr, category="optimization", linear=False,
                    coherence="flux-preserving intensity blur (spatial) + envelope (temporal)",
                    noise="Gaussian (LSQ)", description="flux-preserving least squares, steepest descent"),
    "gpfrwr_gd": dict(func=gpfrwr, category="optimization", linear=False,
                      coherence="as frwr_gd (+ incoherent background)", noise="Gaussian (LSQ)",
                      description="frwr_gd + gradient flipping + published phase prediction"),
    "gpfrwr_lsq": dict(func=lambda d, p, **k: gpfrwr(d, p, **{**dict(prediction="lsq", phi_max=0.5, damping=0.5,
                                                                    every=5, flip_fraction=0.3), **k}),
                       category="optimization", linear=False, coherence="as frwr_gd (+ incoherent background)",
                       noise="Gaussian (LSQ)",
                       description="gpfrwr_gd with least-squares weighted phase prediction (this work)"),
    "mal": dict(func=mal, category="optimization", linear=False,
                coherence="quasi-coherent envelope (linear+non-linear)", noise="Gaussian (LSQ)",
                description="MAL-type maximum likelihood (L-BFGS), linear start"),
    "ml_poisson": dict(func=ml_poisson, category="optimization", linear=False,
                       coherence="rigorous incoherent sum (full TCC, thin object)", noise="Poisson ML",
                       description="Poisson ML with incoherent quadrature"),
    "hybrid": dict(func=hybrid, category="hybrid", linear=False, coherence="as refinement method",
                   noise="as refinement method", description="off-axis start + FRWR refinement (hybrid holography)"),
    "mcnn": dict(func=lambda d, p, **k: __import__("inlineholo.mcnn", fromlist=["mcnn"]).mcnn(d, p, **k),
                 category="learned", linear=False, coherence="implicit (training data)",
                 noise="implicit (training data)", description="multi-resolution CNN (Wang et al. 2020), needs a trained model"),
    "gd": dict(func=gradient_solver, category="optimization", linear=False,
               coherence="selectable", noise="selectable", description="generic gradient solver"),
}


def reconstruct(stack, params: ImagingParameters, method="frwr", normalize=True, **options):
    """Reconstruct the exit wave from a focal series.

    Parameters
    ----------
    stack : array (N, ny, nx), raw intensities (counts) or normalized images
    params : ImagingParameters (defoci, aberrations, coherence, pixel size, ...)
    method : key of METHODS
    normalize : divide each image by its mean before reconstruction; the mean count is
        passed on as ``counts`` (used e.g. for the automatic Wiener regularization)
    options : method-specific keyword arguments (see the individual functions)
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; available: {list(METHODS)}")
    stack = np.asarray(stack, float)
    if stack.ndim != 3 or len(stack) != params.n_images:
        raise ValueError("stack must be (N, ny, nx) with N = len(params.defoci_nm)")
    if normalize:
        data, scale = normalize_stack(stack)
        # integer-valued data are interpreted as electron counts (Poisson statistics)
        sub = stack.ravel()[:: max(1, stack.size // 20000)]
        is_counts = np.allclose(sub, np.round(sub)) and np.mean(sub) > 0
        if not method.startswith("tie"):
            options.setdefault("counts", float(np.mean(scale)) if is_counts else None)
    else:
        data = stack
    t = time.time()
    rec = METHODS[method]["func"](data, params, **options)
    rec.method = method
    if not rec.runtime_s:
        rec.runtime_s = time.time() - t
    rec.options.setdefault("method", method)
    return rec


def list_methods():
    return {k: {kk: vv for kk, vv in v.items() if kk != "func"} for k, v in METHODS.items()}
