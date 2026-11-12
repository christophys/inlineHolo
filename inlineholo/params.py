# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Imaging parameters of a focal series."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np

from .optics import wavelength_nm, ABERRATION_NAMES


@dataclass
class ImagingParameters:
    """All parameters needed to model / reconstruct a focal series.

    Lengths in nm, angles in rad, voltage in V.

    Attributes
    ----------
    voltage_V : acceleration voltage
    pixel_size_nm : sampling of the images (object-plane pixel size)
    defoci_nm : C1 (= c_1_0) of every image, positive = overfocus
    aberrations : global residual aberrations (Krivanek notation, nm), e.g.
        ``{"c_3_0": 1e4, "c_1_2_a": 1.0}``; the defocus entry is ignored (use defoci_nm)
    astig_a_nm, astig_b_nm : optional per-image two-fold astigmatism (C12a, C12b),
        added to the global value
    focal_spread_nm : standard deviation of the defocus distribution (temporal coherence)
    convergence_rad : standard deviation (per axis) of illumination tilts (spatial coherence)
    aperture_rad : objective aperture semi-angle (None = no aperture)
    mtf : optional detector MTF parameters (dict; see ``forward.detector_mtf``)
    """

    voltage_V: float = 300e3
    pixel_size_nm: float = 0.05
    defoci_nm: np.ndarray = field(default_factory=lambda: np.zeros(1))
    aberrations: dict = field(default_factory=dict)
    astig_a_nm: np.ndarray | None = None
    astig_b_nm: np.ndarray | None = None
    focal_spread_nm: float = 0.0
    convergence_rad: float = 0.0
    aperture_rad: float | None = None
    mtf: dict | None = None
    notes: str = ""

    def __post_init__(self):
        self.defoci_nm = np.atleast_1d(np.asarray(self.defoci_nm, dtype=float))
        n = len(self.defoci_nm)
        for k in ("astig_a_nm", "astig_b_nm"):
            v = getattr(self, k)
            setattr(self, k, np.zeros(n) if v is None else np.atleast_1d(np.asarray(v, float)))
        for k in self.aberrations:
            if k not in ABERRATION_NAMES:
                raise ValueError(f"unknown aberration {k!r}; use Krivanek names like c_3_0")

    @property
    def n_images(self):
        return len(self.defoci_nm)

    @property
    def wavelength_nm(self):
        return wavelength_nm(self.voltage_V)

    def aberrations_for(self, i: int, extra_defocus: float = 0.0) -> dict:
        """Full aberration set of image i (global + per-image defocus/astigmatism)."""
        ab = {k: float(v) for k, v in self.aberrations.items() if k != "c_1_0"}
        ab["c_1_0"] = float(self.defoci_nm[i]) + float(self.aberrations.get("c_1_0", 0.0)) \
            + extra_defocus
        ab["c_1_2_a"] = ab.get("c_1_2_a", 0.0) + float(self.astig_a_nm[i])
        ab["c_1_2_b"] = ab.get("c_1_2_b", 0.0) + float(self.astig_b_nm[i])
        return ab

    def subset(self, idx):
        idx = np.asarray(idx)
        p = self.copy()
        p.defoci_nm = p.defoci_nm[idx]
        p.astig_a_nm = p.astig_a_nm[idx]
        p.astig_b_nm = p.astig_b_nm[idx]
        return p

    def copy(self):
        return ImagingParameters.from_dict(self.to_dict())

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_json(self):
        return json.dumps(self.to_dict())

    def summary(self):
        lam = self.wavelength_nm
        s = [f"U = {self.voltage_V/1e3:.1f} kV (lambda = {lam*1e3:.3f} pm)",
             f"pixel = {self.pixel_size_nm:.4g} nm, N_images = {self.n_images}",
             f"defoci [nm] = {np.array2string(self.defoci_nm, precision=1)}",
             f"focal spread = {self.focal_spread_nm:.3g} nm, "
             f"convergence (rms) = {self.convergence_rad*1e3:.3g} mrad"]
        if self.aberrations:
            s.append("aberrations [nm]: " + ", ".join(f"{k}={v:.4g}" for k, v in self.aberrations.items()))
        return "\n".join(s)
