# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Thon-ring fitting: determine defocus and two-fold astigmatism (optionally C3)
from the power spectra of focal-series images (cf. Thon 1966; Zemlin et al. 1978;
Mindell & Grigorieff 2003; Rohou & Grigorieff 2015).

Model of the background-subtracted amplitude spectrum::

    M(q) = sin^2( chi(q) - atan(w) ) - <...>,     w = amplitude-contrast ratio

The score is the normalized cross-correlation between the background-subtracted,
locally normalized amplitude spectrum and M within an annulus q_min < |q| < q_max.
A 1D search on the rotational average gives C1; a Nelder-Mead refinement in 2D gives
(C1, C12a, C12b).  Note that for negligible C3 the spectrum is invariant under
chi -> -chi, i.e. the sign of the defocus is ambiguous; restrict the search window
to the nominal sign.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

from .control import StopRequested
from .optics import FourierGrid, wavelength_nm


def _crop_to_q(B, grid: FourierGrid, q_max, margin=1.15):
    """Crop an (unshifted, square) spectrum to |q| <= margin*q_max: same dq, fewer pixels."""
    n = B.shape[0]
    if B.shape[0] != B.shape[1]:
        return B, grid
    dq = 1.0 / (n * grid.pixel)
    m = int(2 * np.ceil(margin * q_max / dq)) + 4
    m += m % 2
    if m >= n:
        return B, grid
    c = n // 2
    Bs = np.fft.fftshift(B)[c - m // 2:c + m // 2, c - m // 2:c + m // 2]
    return np.fft.ifftshift(Bs), FourierGrid((m, m), grid.pixel * n / m)


def series_defoci(n, offset, step, m=1.0, n0=None):
    """FRWR series (Eq. 1 of the FRWR manual): df_k = offset + step sign(k - n0) |k - n0|^m."""
    n0 = (n - 1) // 2 if n0 is None else n0
    k = np.arange(n) - n0
    return offset + step * np.sign(k) * np.abs(k) ** m


def power_spectrum(img, tile="auto"):
    """Periodogram (unshifted) with Hann window; Welch averaging over tiles if given
    (``tile='auto'``: half the image size for images >= 256 px)."""
    img = np.asarray(img, float)
    if tile == "auto":
        tile = min(img.shape) // 2 if min(img.shape) >= 256 else None
    if tile is None or tile >= min(img.shape):
        w = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
        F = np.fft.fft2((img - img.mean()) * w)
        return np.abs(F) ** 2
    w = np.outer(np.hanning(tile), np.hanning(tile))
    acc = np.zeros((tile, tile))
    n = 0
    for y in range(0, img.shape[0] - tile + 1, tile // 2):
        for x in range(0, img.shape[1] - tile + 1, tile // 2):
            t = img[y:y + tile, x:x + tile]
            acc += np.abs(np.fft.fft2((t - t.mean()) * w)) ** 2
            n += 1
    return acc / max(n, 1)


def preprocess_spectrum(ps, smooth_frac=0.04):
    """Amplitude spectrum with smooth background removed and local rms normalized."""
    A = np.fft.fftshift(np.sqrt(ps))
    s = max(1.0, smooth_frac * A.shape[0])
    bg = gaussian_filter(A, s, mode="nearest")
    B = A - bg
    rms = np.sqrt(gaussian_filter(B ** 2, 2 * s, mode="nearest")) + 1e-12
    return np.fft.ifftshift(B / rms)


def ctf_model(grid: FourierGrid, aberrations, lam, w=0.07):
    chi = grid.chi(aberrations, lam)
    return np.sin(chi - np.arctan(w)) ** 2


def _score(B, M, mask):
    b = B[mask] - B[mask].mean()
    m = M[mask] - M[mask].mean()
    return float(np.sum(b * m) / np.sqrt(np.sum(b * b) * np.sum(m * m) + 1e-30))


def radial_profile(img, grid: FourierGrid, nbins):
    edges = np.linspace(0, grid.q_nyquist, nbins + 1)
    idx = np.clip(np.digitize(grid.q, edges) - 1, 0, nbins - 1)
    s = np.bincount(idx.ravel(), img.ravel(), nbins)
    c = np.bincount(idx.ravel(), minlength=nbins)
    return 0.5 * (edges[1:] + edges[:-1]), s / np.maximum(c, 1)


def fit_thon(image, pixel_nm, voltage_V=300e3, c1_range=(-200.0, 200.0), c3_nm=0.0,
             q_min=None, q_max=None, fit_astig=True, amp_contrast=0.07, tile="auto",
             extra_aberrations=None, auto_qmax=True, smooth_px=1.0, progress=None):
    """Fit defocus and two-fold astigmatism to the Thon rings of one image.

    progress : optional callable(stage, n_evaluations, best_score); it may raise
        :class:`~inlineholo.control.StopRequested`, in which case the best parameters found so
        far are returned (with ``stopped=True``).
    Returns dict(c_1_0, c_1_2_a, c_1_2_b, score, score_1d, profile=(q, data, model)).
    """
    lam = wavelength_nm(voltage_V)
    ps = power_spectrum(image, tile)
    grid = FourierGrid(ps.shape, pixel_nm)
    B = preprocess_spectrum(ps)
    if q_min is None:
        q_min = 0.06 * grid.q_nyquist
    if q_max is None:
        q_max = 0.9 * grid.q_nyquist
    B, grid = _crop_to_q(B, grid, q_max)            # only the fitted annulus is evaluated
    mask = (grid.q > q_min) & (grid.q < q_max)
    base = dict(extra_aberrations or {})
    base["c_3_0"] = c3_nm
    # 1D search on rotational averages
    nb = B.shape[0] // 2
    qr, Br = radial_profile(B, grid, nb)
    m1 = (qr > q_min) & (qr < q_max)
    phi0 = np.zeros_like(qr)
    dC = 0.05 / (np.pi * lam * q_max ** 2)
    c1s = np.arange(c1_range[0], c1_range[1] + dC, dC)
    best, bc1 = -2, c1_range[0]
    from .optics import chi_polar
    for c1 in c1s:
        ab = dict(base, c_1_0=c1)
        chi = chi_polar(lam * qr, phi0, ab, lam)
        M = np.sin(chi - np.arctan(amp_contrast)) ** 2
        sc = _score(Br, M, m1)
        if sc > best:
            best, bc1 = sc, c1
    score1d = best
    if auto_qmax:
        # limit the 2D fit to the range where rings are actually visible: local 1D
        # correlation of data and best model in sliding windows
        chi = chi_polar(lam * qr, phi0, dict(base, c_1_0=bc1), lam)
        Mr = np.sin(chi - np.arctan(amp_contrast)) ** 2
        win = max(6, nb // 16)
        good = []
        for i in range(0, nb - win):
            sl = slice(i, i + win)
            if qr[i] < q_min:
                continue
            good.append((qr[i + win // 2], _score(Br[sl], Mr[sl], np.ones(win, bool))))
        ok = [qq for qq, c in good if c > 0.3]
        if ok:
            q_max = min(q_max, max(ok))
        mask = (grid.q > q_min) & (grid.q < q_max)
    if smooth_px:
        B = np.fft.ifftshift(gaussian_filter(np.fft.fftshift(B), smooth_px))

    state = {"n": 0, "best": (np.inf, np.array([bc1, 0.0, 0.0])), "stage": "1D search"}

    def make_neg(msk):
        def neg(x):
            x = np.asarray(x, float)
            ab = dict(base, c_1_0=x[0], c_1_2_a=x[1] if fit_astig else 0.0,
                      c_1_2_b=x[2] if fit_astig else 0.0)
            f = -_score(B, ctf_model(grid, ab, lam, amp_contrast), msk)
            state["n"] += 1
            if f < state["best"][0] and state["stage"].endswith("full"):
                state["best"] = (f, x.copy())
            if progress is not None:
                progress(state["stage"], state["n"], -min(f, state["best"][0]))
            return f
        return neg

    nm = dict(xatol=1e-3, fatol=1e-6, maxiter=800)
    stopped = False
    try:
        if progress is not None:
            progress("1D search", 0, score1d)
        if fit_astig:
            # coarse-to-fine: low-q annulus first (robust, few rings), then full range
            stages = [q_min + f * (q_max - q_min) for f in (0.35, 0.65, 1.0)]
            sa = max(1.0, 0.05 * abs(bc1))
            starts = [(bc1, 0, 0), (bc1, sa, 0), (bc1, 0, sa), (bc1, -sa, 0), (bc1, 0, -sa)]
            for k, qm in enumerate(stages):
                state["stage"] = f"2D stage {k + 1}/3" + (" full" if k == 2 else "")
                neg = make_neg((grid.q > q_min) & (grid.q < qm))
                res = min((minimize(neg, s0, method="Nelder-Mead", options=nm) for s0 in starts),
                          key=lambda r: r.fun)
                starts = [res.x]
                state["best"] = (res.fun, np.asarray(res.x, float))
            x = res.x
        else:
            state["stage"] = "C1 full"
            neg = make_neg(mask)
            res = minimize(lambda v: neg([v[0], 0, 0]), [bc1], method="Nelder-Mead")
            x = [res.x[0], 0.0, 0.0]
        fbest = res.fun
    except StopRequested:
        stopped = True
        fbest, x = state["best"]
        if not np.isfinite(fbest):
            fbest = -score1d
    ab = dict(base, c_1_0=x[0], c_1_2_a=x[1], c_1_2_b=x[2])
    chi = chi_polar(lam * qr, phi0, dict(base, c_1_0=x[0]), lam)
    Mr = np.sin(chi - np.arctan(amp_contrast)) ** 2
    return dict(c_1_0=float(x[0]), c_1_2_a=float(x[1]), c_1_2_b=float(x[2]),
                score=float(-fbest), score_1d=float(score1d),
                profile=(qr, Br, Mr), q_range=(q_min, q_max), aberrations=ab, stopped=stopped)


def fit_series(stack, params, window_nm=None, fit_astig=True, fit_c3=False,
               c3_values=None, **kw):
    """Fit all images of a focal series around the nominal defoci in ``params``.

    window_nm : half width of the defocus search window around each nominal value
        (default: max(10 nm, 30 % of |C1|)).  Returns (list of per-image dicts, c3).
    """
    c3 = params.aberrations.get("c_3_0", 0.0)
    extra = {k: v for k, v in params.aberrations.items()
             if k not in ("c_1_0", "c_1_2_a", "c_1_2_b", "c_3_0")}

    def run(c3v):
        out = []
        for img, d in zip(stack, params.defoci_nm):
            w = window_nm if window_nm is not None else max(10.0, 0.3 * abs(d))
            out.append(fit_thon(img, params.pixel_size_nm, params.voltage_V, (d - w, d + w),
                                c3v, fit_astig=fit_astig, extra_aberrations=extra, **kw))
        return out

    if fit_c3:
        vals = c3_values if c3_values is not None else c3 * np.array([0.5, 0.75, 1, 1.25, 1.5]) + 0.0
        best = None
        for v in vals:
            r = run(v)
            s = np.mean([x["score"] for x in r])
            if best is None or s > best[0]:
                best = (s, v, r)
        return best[2], best[1]
    return run(c3), c3


def thon_composite(image, pixel_nm, aberrations, voltage_V=300e3, amp_contrast=0.07, tile="auto"):
    """Composite (fftshifted) image: left half = processed spectrum, right half = model."""
    lam = wavelength_nm(voltage_V)
    ps = power_spectrum(image, tile)
    grid = FourierGrid(ps.shape, pixel_nm)
    B = np.fft.fftshift(preprocess_spectrum(ps))
    M = np.fft.fftshift(ctf_model(grid, aberrations, lam, amp_contrast))
    B = (B - np.percentile(B, 1)) / (np.percentile(B, 99) - np.percentile(B, 1) + 1e-12)
    out = B.copy()
    nx = B.shape[1]
    out[:, nx // 2:] = M[:, nx // 2:]
    return np.clip(out, 0, 1)


def fit_series_global(stack, params, window_nm=20.0, fit_scale=True, fit_astig=True,
                      amp_contrast=0.07, q_min=None, q_max=None, tile="auto", min_abs_defocus=0.0,
                      model="scale", n0=None, m=1.0, fit_m=False, progress=None, maxiter=1500):
    """Series-consistent Thon-ring fit.

    model='scale': C1_n = offset + scale * C1_n(nominal)  (nominal = current values)
    model='power': C1_n = offset + step sign(n - n0) |n - n0|^m  (Eq. 1 of the FRWR manual;
        non-linear defocus series), with m fixed or fitted (``fit_m``).
    In both cases a common two-fold astigmatism (C12a, C12b) is fitted (plus the per-image values
    of the parameter table) and the summed 2D score of all images is maximized, so that images
    close to focus (few rings) are constrained by the strongly defocused ones.  Images with
    |C1_nominal| < min_abs_defocus are excluded from the score.
    progress : optional callable(stage, n_evaluations, max_evaluations, best_score); it may raise
        StopRequested (the best parameters so far are then returned with stopped=True).
    Returns dict(offset, scale, step, m, n0, model, c_1_2_a, c_1_2_b, defoci, score,
    per_image_scores, stopped).
    """
    lam = params.wavelength_nm
    nominal = np.asarray(params.defoci_nm, float)
    n = len(nominal)
    n0 = (n - 1) // 2 if n0 is None else int(n0)
    prep = []
    use = [i for i in range(len(stack)) if abs(nominal[i]) >= min_abs_defocus]
    for i in use:
        ps = power_spectrum(stack[i], tile)
        grid = FourierGrid(ps.shape, params.pixel_size_nm)
        B = preprocess_spectrum(ps)
        B = np.fft.ifftshift(gaussian_filter(np.fft.fftshift(B), 1.0))
        qa = 0.06 * grid.q_nyquist if q_min is None else q_min
        qb = 0.8 * grid.q_nyquist if q_max is None else q_max
        B, grid = _crop_to_q(B, grid, qb)
        prep.append((i, grid, B, (grid.q > qa) & (grid.q < qb)))
        if progress is not None:
            progress(f"spectra {len(prep)}/{len(use)}", 0, maxiter, float("nan"))
    base = {k: v for k, v in params.aberrations.items() if k not in ("c_1_0",)}
    a0 = base.pop("c_1_2_a", 0.0)
    b0 = base.pop("c_1_2_b", 0.0)
    kk = np.arange(n) - n0

    def defoci(x):
        if model == "power":
            off, step, mm = x[0], x[1], float(np.clip(x[2], 0.3, 6.0))     # keep m physical
            return off + step * np.sign(kk) * np.abs(kk) ** mm
        return x[0] + x[1] * nominal

    def scores(x):
        d = defoci(x)
        a, b = x[-2], x[-1]
        out = []
        for i, grid, B, msk in prep:
            ab = dict(base, c_1_0=d[i], c_1_2_a=a + params.astig_a_nm[i], c_1_2_b=b + params.astig_b_nm[i])
            out.append(_score(B, ctf_model(grid, ab, lam, amp_contrast), msk))
        return np.array(out)

    if model == "power":
        s = np.sign(kk) * np.abs(kk) ** m
        A = np.c_[np.ones(n), s]
        off0, step0 = np.linalg.lstsq(A, nominal, rcond=None)[0]
        x0 = np.array([off0, step0, m, a0, b0])
        free = [True, True, bool(fit_m), fit_astig, fit_astig]
        shift = np.array([1.0, 0, 0, 0, 0])
    else:
        x0 = np.array([0.0, 1.0, a0, b0])
        free = [True, fit_scale, fit_astig, fit_astig]
        shift = np.array([1.0, 0, 0, 0])
    state = {"n": 0, "best": (np.inf, x0.copy())}

    def evaluate(x, stage):
        f = -float(np.mean(scores(x)))
        state["n"] += 1
        if f < state["best"][0]:
            state["best"] = (f, np.array(x, float))
        if progress is not None:
            progress(stage, state["n"], maxiter, -state["best"][0])
        return f

    stopped = False
    try:
        # coarse search of the offset
        dC = 0.25
        if prep:
            g = prep[0][1]
            dC = 0.05 / (np.pi * lam * (0.8 * g.q_nyquist) ** 2)
        offs = np.arange(-window_nm, window_nm + dC, dC)
        if model == "power" and fit_m:
            # coarse scan of m (step re-fitted to the nominal series for each m), offset search for each
            cand = []
            for mm in np.clip(m + np.arange(-0.3, 0.301, 0.05), 0.3, None):
                s_ = np.sign(kk) * np.abs(kk) ** mm
                o_, st_ = np.linalg.lstsq(np.c_[np.ones(n), s_], nominal, rcond=None)[0]
                xm = x0.copy()
                xm[:3] = o_, st_, mm
                sc = [evaluate(xm + o * shift, f"m scan (m = {mm:.2f})") for o in offs]
                cand.append((min(sc), xm + offs[int(np.argmin(sc))] * shift))
            x0 = min(cand, key=lambda c: c[0])[1]
        else:
            sc0 = [evaluate(x0 + o * shift, "offset search") for o in offs]
            x0 = x0 + offs[int(np.argmin(sc0))] * shift
        fr = np.array(free)

        def neg(v):
            x = x0.copy()
            x[fr] = v
            return evaluate(x, "Nelder-Mead")

        res = minimize(neg, x0[fr], method="Nelder-Mead",
                       options=dict(xatol=1e-4, fatol=1e-6, maxiter=maxiter))
        x = x0.copy()
        x[fr] = res.x
    except StopRequested:
        stopped = True
        x = state["best"][1]
    per = scores(x)
    d = defoci(x)
    out = dict(model=model, c_1_2_a=float(x[-2]), c_1_2_b=float(x[-1]), defoci=d, score=float(per.mean()),
               per_image_scores=dict(zip(use, per.tolist())), stopped=stopped, n0=n0)
    if model == "power":
        out.update(offset=float(x[0]), step=float(x[1]), m=float(np.clip(x[2], 0.3, 6.0)), scale=1.0)
    else:
        out.update(offset=float(x[0]), scale=float(x[1]), step=None, m=None)
    return out
