# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Minimal array backend so that the iterative solvers run on CPU (NumPy) or GPU.

Supported back ends
-------------------
``numpy``  always available (CPU, scipy.fft with all cores)
``cupy``   NVIDIA GPUs (``pip install cupy-cuda12x``)
``torch``  PyTorch tensors on ``cuda`` / ``mps`` / ``cpu``

``get_backend("auto")`` picks CuPy if a CUDA device is visible, then PyTorch with CUDA
or Apple MPS, and falls back to NumPy.  All solvers written against the small API
below (``B.exp``, ``B.fft2``, ...) run unchanged on any back end; data never leave the
device during an iteration and nothing is written to disk.
"""
from __future__ import annotations

import numpy as np


class _NumpyBackend:
    name = "numpy"

    def __init__(self, dtype="float64"):
        import scipy.fft as sfft
        self._fft = sfft
        self.xp = np
        self.real_dtype = np.dtype(dtype)
        self.complex_dtype = np.result_type(self.real_dtype, np.complex64)
        self.device = "cpu"

    # creation / transfer
    def asarray(self, a, complex_=False):
        return np.asarray(a, dtype=self.complex_dtype if complex_ else self.real_dtype)

    def to_numpy(self, a):
        return np.asarray(a)

    def asint(self, a):
        return np.asarray(a, dtype=np.int64)

    def zeros(self, shape, complex_=False):
        return np.zeros(shape, self.complex_dtype if complex_ else self.real_dtype)

    def ones(self, shape):
        return np.ones(shape, self.real_dtype)

    def full(self, shape, v):
        return np.full(shape, v, self.real_dtype)

    def arange(self, n):
        return np.arange(n)

    def copy(self, a):
        return a.copy()

    def item(self, a):
        return float(a)

    # FFTs (unnormalized forward, 1/N inverse, numpy convention)
    def fft2(self, a):
        return self._fft.fft2(a, workers=-1)

    def ifft2(self, a):
        return self._fft.ifft2(a, workers=-1)

    def rfft2(self, a):
        return self._fft.rfft2(a, workers=-1)

    def irfft2(self, a, shape):
        return self._fft.irfft2(a, s=shape, workers=-1)

    # elementwise / reductions (numpy names)
    def __getattr__(self, k):
        return getattr(np, k)


class _CupyBackend(_NumpyBackend):
    name = "cupy"

    def __init__(self, dtype="float32"):
        import cupy as cp
        self.xp = cp
        self.real_dtype = np.dtype(dtype)
        self.complex_dtype = np.result_type(self.real_dtype, np.complex64)
        self.device = "cuda"

    def asarray(self, a, complex_=False):
        return self.xp.asarray(a, dtype=self.complex_dtype if complex_ else self.real_dtype)

    def to_numpy(self, a):
        return self.xp.asnumpy(a)

    def asint(self, a):
        return self.xp.asarray(a, dtype=self.xp.int64)

    def zeros(self, shape, complex_=False):
        return self.xp.zeros(shape, self.complex_dtype if complex_ else self.real_dtype)

    def ones(self, shape):
        return self.xp.ones(shape, self.real_dtype)

    def full(self, shape, v):
        return self.xp.full(shape, v, self.real_dtype)

    def arange(self, n):
        return self.xp.arange(n)

    def item(self, a):
        return float(a)

    def fft2(self, a):
        return self.xp.fft.fft2(a)

    def ifft2(self, a):
        return self.xp.fft.ifft2(a)

    def rfft2(self, a):
        return self.xp.fft.rfft2(a)

    def irfft2(self, a, shape):
        return self.xp.fft.irfft2(a, s=shape)

    def __getattr__(self, k):
        return getattr(self.xp, k)


class _TorchBackend:
    """NumPy-like subset on top of torch tensors."""
    name = "torch"

    def __init__(self, device=None, dtype="float32"):
        import torch
        self.torch = torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device == "mps":
            dtype = "float32"              # MPS has no float64
        self.device = device
        self.real_dtype = getattr(torch, str(np.dtype(dtype)))
        self.complex_dtype = torch.complex64 if self.real_dtype == torch.float32 else torch.complex128
        self.xp = self
        self.pi = np.pi
        self.newaxis = None

    def asarray(self, a, complex_=False):
        t = self.torch
        if isinstance(a, t.Tensor):
            return a.to(device=self.device, dtype=self.complex_dtype if complex_ else
                        (a.dtype if a.dtype in (t.int64, t.int32, t.bool) else self.real_dtype))
        a = np.asarray(a)
        if complex_ or np.iscomplexobj(a):
            return t.as_tensor(a.astype(np.complex128 if self.complex_dtype == t.complex128 else np.complex64),
                               device=self.device)
        if a.dtype.kind in "iub":
            return t.as_tensor(a, device=self.device)
        return t.as_tensor(a, dtype=self.real_dtype, device=self.device)

    def to_numpy(self, a):
        if isinstance(a, self.torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    def asint(self, a):
        return self.torch.as_tensor(np.asarray(a, np.int64), device=self.device)

    def zeros(self, shape, complex_=False):
        return self.torch.zeros(shape, dtype=self.complex_dtype if complex_ else self.real_dtype,
                                device=self.device)

    def ones(self, shape):
        return self.torch.ones(shape, dtype=self.real_dtype, device=self.device)

    def full(self, shape, v):
        return self.torch.full(shape, float(v), dtype=self.real_dtype, device=self.device)

    def arange(self, n):
        return self.torch.arange(n, device=self.device)

    def copy(self, a):
        return a.clone()

    def item(self, a):
        return float(a.item()) if isinstance(a, self.torch.Tensor) else float(a)

    def fft2(self, a):
        return self.torch.fft.fft2(a)

    def ifft2(self, a):
        return self.torch.fft.ifft2(a)

    def rfft2(self, a):
        return self.torch.fft.rfft2(a)

    def irfft2(self, a, shape):
        return self.torch.fft.irfft2(a, s=tuple(shape))

    # numpy-style helpers
    def _t(self, x):
        t = self.torch
        return x if isinstance(x, t.Tensor) else t.as_tensor(x, dtype=self.real_dtype, device=self.device)

    def exp(self, x): return self.torch.exp(self._t(x))
    def cos(self, x): return self.torch.cos(self._t(x))
    def sin(self, x): return self.torch.sin(self._t(x))
    def sqrt(self, x): return self.torch.sqrt(self._t(x))
    def abs(self, x): return self.torch.abs(self._t(x))
    def arctan2(self, y, x): return self.torch.atan2(self._t(y), self._t(x))
    def angle(self, z): return self.torch.angle(z)
    def real(self, z): return z.real if z.is_complex() else z
    def imag(self, z): return z.imag
    def conj(self, z): return self.torch.conj(z)
    def floor(self, x): return self.torch.floor(self._t(x))
    def sign(self, x): return self.torch.sign(self._t(x))
    def hypot(self, a, b): return self.torch.hypot(self._t(a), self._t(b))
    def maximum(self, a, b):
        a = self._t(a)
        return self.torch.clamp(a, min=float(b)) if np.isscalar(b) else self.torch.maximum(a, self._t(b))
    def minimum(self, a, b):
        a = self._t(a)
        return self.torch.clamp(a, max=float(b)) if np.isscalar(b) else self.torch.minimum(a, self._t(b))
    def clip(self, a, lo, hi): return self.torch.clamp(self._t(a), lo, hi)
    def where(self, c, a, b):
        t = self.torch
        if not isinstance(a, t.Tensor):
            a = t.as_tensor(a, dtype=b.dtype if isinstance(b, t.Tensor) else self.real_dtype, device=self.device)
        if not isinstance(b, t.Tensor):
            b = t.as_tensor(b, dtype=a.dtype, device=self.device)
        return t.where(c, a, b)
    def sum(self, a, axis=None):
        return self.torch.sum(a) if axis is None else self.torch.sum(a, dim=axis)
    def mean(self, a, axis=None):
        return self.torch.mean(a) if axis is None else self.torch.mean(a, dim=axis)
    def max(self, a): return self.torch.max(a)
    def min(self, a): return self.torch.min(a)
    def argmin(self, a): return int(self.torch.argmin(a).item())
    def isfinite(self, a): return self.torch.isfinite(a)
    def roll(self, a, shift, axis): return self.torch.roll(a, shifts=shift, dims=axis)
    def cumsum(self, a, axis=0): return self.torch.cumsum(a, dim=axis)
    def stack(self, lst, axis=0): return self.torch.stack(lst, dim=axis)
    def take(self, a, idx): return a.reshape(-1)[idx]
    def fftfreq(self, n, d=1.0):
        return self.asarray(np.fft.fftfreq(n, d))


def get_backend(name: str | None = "auto", dtype: str | None = None, device=None):
    """Return a backend instance.  ``name`` in {'auto', 'numpy', 'cupy', 'torch'}."""
    name = (name or "auto").lower()
    if name == "auto":
        try:
            import cupy as cp  # noqa
            if cp.cuda.runtime.getDeviceCount() > 0:
                return _CupyBackend(dtype or "float32")
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available() or (getattr(torch.backends, "mps", None) is not None
                                             and torch.backends.mps.is_available()):
                return _TorchBackend(device, dtype or "float32")
        except Exception:
            pass
        return _NumpyBackend(dtype or "float64")
    if name == "numpy":
        return _NumpyBackend(dtype or "float64")
    if name == "cupy":
        return _CupyBackend(dtype or "float32")
    if name == "torch":
        return _TorchBackend(device, dtype or "float32")
    raise ValueError(f"unknown backend {name!r}")


def available_backends():
    out = ["numpy"]
    try:
        import cupy as cp  # noqa
        if cp.cuda.runtime.getDeviceCount() > 0:
            out.append("cupy")
    except Exception:
        pass
    try:
        import torch
        out.append("torch" + ("(cuda)" if torch.cuda.is_available() else
                              "(mps)" if getattr(torch.backends, "mps", None) is not None
                              and torch.backends.mps.is_available() else "(cpu)"))
    except Exception:
        pass
    return out
