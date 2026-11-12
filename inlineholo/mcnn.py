# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Multi-resolution convolutional neural network (MCNN) for focal-series phase retrieval,
after Wang, Eljarrat, Mueller, Henninen, Erni & Koch, Sci. Rep. 10, 5730 (2020).

A U-Net whose decoder levels are connected to additional output heads, so that the network is
trained to match the target at *all* resolutions (coarse outputs carry the low spatial
frequencies, which a plain U-Net learns only slowly).  Loss: mean absolute error summed over
all output scales.  Input: the N normalized images of a focal series (I/<I> - 1); output: the
(unwrapped, zero-mean) phase and the log-amplitude of the exit wave.

Requires PyTorch (optional dependency).  Training data are simulated with the fast
flux-preserving forward model of :mod:`inlineholo.forward` from random objects
(:func:`inlineholo.simulate.random_mr_object`) with random dose.
"""
from __future__ import annotations

import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

from .forward import ForwardModel, normalize_stack
from .reconstruct import Reconstruction
from . import simulate as sim


def _require():
    if not HAVE_TORCH:
        raise ImportError("MCNN needs PyTorch: pip install torch")


if HAVE_TORCH:
    class _Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.c = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular"), nn.LeakyReLU(0.1),
                                   nn.Conv2d(cout, cout, 3, padding=1, padding_mode="circular"), nn.LeakyReLU(0.1))

        def forward(self, x):
            return self.c(x)

    class MCNN(nn.Module):
        """U-Net with multi-resolution output heads (one per decoder level)."""

        def __init__(self, n_in, n_out=2, width=16, depth=4):
            super().__init__()
            ch = [width * 2 ** i for i in range(depth + 1)]
            self.enc = nn.ModuleList([_Block(n_in if i == 0 else ch[i - 1], ch[i]) for i in range(depth)])
            self.bottom = _Block(ch[depth - 1], ch[depth])
            self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
            self.dec = nn.ModuleList([_Block(2 * ch[i], ch[i]) for i in reversed(range(depth))])
            self.heads = nn.ModuleList([nn.Conv2d(ch[depth], n_out, 1)] +
                                       [nn.Conv2d(ch[i], n_out, 1) for i in reversed(range(depth))])

        def forward(self, x):
            skips = []
            for e in self.enc:
                x = e(x)
                skips.append(x)
                x = F.avg_pool2d(x, 2)
            x = self.bottom(x)
            outs = [self.heads[0](x)]
            for k, (u, d) in enumerate(zip(self.up, self.dec)):
                x = u(x)
                x = d(torch.cat([x, skips[-1 - k]], 1))
                outs.append(self.heads[k + 1](x))
            return outs          # coarse ... fine


def make_training_set(params, n_samples=1024, n=128, dose_range=(1.0, 300.0), seed=0, model="blur"):
    """Simulate (inputs, targets) for training: inputs (S, N, n, n), targets (S, 2, n, n)."""
    rng = np.random.default_rng(seed)
    fm = ForwardModel(params, (n, n), model)
    X = np.empty((n_samples, params.n_images, n, n), np.float32)
    Y = np.empty((n_samples, 2, n, n), np.float32)
    for s in range(n_samples):
        wave, phase = sim.random_mr_object(n, params.pixel_size_nm, params.voltage_V, rng)
        I = fm.forward(wave)
        dose = np.exp(rng.uniform(np.log(dose_range[0]), np.log(dose_range[1])))
        d = sim.dose_per_pixel(dose, params.pixel_size_nm, params.n_images)
        I = rng.poisson(np.maximum(I, 0) * d).astype(float)
        I, _ = normalize_stack(I)
        X[s] = I - 1
        Y[s, 0] = phase - phase.mean()
        Y[s, 1] = np.log(np.maximum(np.abs(wave), 1e-3))
    return X, Y


def train(params, X, Y, width=16, depth=4, epochs=None, time_budget_s=600, batch=8, lr=1e-3,
          seed=0, verbose=True):
    """Train an MCNN with the multi-scale MAE loss; returns (model, loss history)."""
    _require()
    torch.manual_seed(seed)
    net = MCNN(X.shape[1], 2, width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    targets = None
    hist = []
    t0 = time.time()
    ep = 0
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(len(X))
        for b in range(0, len(X) - batch + 1, batch):
            idx = perm[b:b + batch]
            xb, yb = Xt[idx], Yt[idx]
            outs = net(xb)
            loss = 0.0
            for o in outs:
                f = yb.shape[-1] // o.shape[-1]
                tgt = F.avg_pool2d(yb, f) if f > 1 else yb
                loss = loss + torch.mean(torch.abs(o - tgt))
            opt.zero_grad()
            loss.backward()
            opt.step()
            hist.append(loss.item())
            if time.time() - t0 > time_budget_s:
                break
        ep += 1
        if verbose:
            print(f"epoch {ep}: loss {np.mean(hist[-20:]):.4f}  ({time.time() - t0:.0f} s)")
        if time.time() - t0 > time_budget_s or (epochs is not None and ep >= epochs):
            break
    net.eval()
    return net, hist


def save(net, path, meta=None):
    _require()
    torch.save(dict(state=net.state_dict(), n_in=net.enc[0].c[0].in_channels,
                    width=net.enc[0].c[0].out_channels, depth=len(net.enc), meta=meta or {}), path)


def load(path):
    _require()
    d = torch.load(path, map_location="cpu", weights_only=False)
    net = MCNN(d["n_in"], 2, d["width"], d["depth"])
    net.load_state_dict(d["state"])
    net.eval()
    return net, d.get("meta", {})


def mcnn(data, params, model=None, model_path=None, **kw):
    """Reconstruct with a trained MCNN (``model`` object or ``model_path``).  The image size
    must be divisible by 2**depth.  The network must have been trained for the same number of
    images, defoci and imaging conditions."""
    _require()
    t0 = time.time()
    net = model if model is not None else load(model_path)[0]
    I, _ = normalize_stack(data)
    with torch.no_grad():
        out = net(torch.from_numpy((I - 1)[None].astype(np.float32)))[-1][0].numpy()
    phase, logamp = out[0], out[1]
    return Reconstruction(np.exp(logamp) * np.exp(1j * phase), "mcnn", dict(model_path=str(model_path)), [],
                          time.time() - t0)
