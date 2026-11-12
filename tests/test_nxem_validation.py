# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""NXem files written by inlineholo comply with the NXem application definition (pynxtools)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import inlineholo as ih  # noqa: E402
from inlineholo import nxem  # noqa: E402

pytest.importorskip("pynxtools")


def _series(n=48):
    w = ih.simulate.mr_object(n=n, pixel=2.0, seed=4, film_nm=2.0)
    p = ih.simulate.mr_parameters(pixel_size_nm=2.0, defoci_nm=np.array([-800., 0., 800.]))
    I, _ = ih.simulate.simulate_focal_series(w, p, None, n_defocus=1, n_tilt=1)
    return w, p, I


META = {"experiment": {"start_time": "2026-11-01T10:00:00+01:00", "doi": "10.1000/xyz"},
        "user": {"name": "A. User", "orcid": "0000-0002-1825-0097", "affiliation": "Lab"},
        "sample": {"name": "Au on C", "atom_types": "Au; C", "preparation_date": "2026-10-01",
                   "identifier": "S-17", "identifier_type": "local", "thickness_nm": "20"},
        "instrument": {"vendor": "JEOL", "model": "JEM-ARM200F", "emitter_type": "cold_field_emission",
                       "cs_corrector": True, "corrector_vendor": "CEOS", "detector": "OneView",
                       "software": "DigitalMicrograph 3.5"}}


@pytest.mark.parametrize("meta", [META, None])
def test_written_files_are_valid_nxem(tmp_path, meta):
    w, p, I = _series()
    fn = str(tmp_path / "s.nxs")
    nxem.write_focal_series(fn, I, p, ground_truth=w, metadata=meta)
    for m in ("tie", "tie_fp", "frwr"):
        r = ih.reconstruct(I, p, m, **({"n_iter": 4, "intermediate_loops": 2} if m == "frwr" else {}))
        nxem.write_reconstruction(fn, r, p, label=m)
    nxem.write_strain(fn, np.zeros((48, 48)), p, (2.0, 0.0), 1.0, source="/entry1/roi1/img/image1")
    nxem.write_curve(fn, np.linspace(0, 1, 10), {"frc": np.ones(10), "half_bit_threshold": np.ones(10) * .2},
                     p, "half-set FRC", "half_set_frc", axis_units="1/nm")
    ok, problems, undocumented = nxem.validate(fn)
    assert ok, problems
    assert not problems and not undocumented
    # everything can be read back
    stack, p2, gt, recs = nxem.read_focal_series(fn)
    assert list(recs) == ["tie", "tie_fp", "frwr"] and np.allclose(gt, w)
    kinds = [k for k, _ in nxem.read_maps(fn).values()]
    for k in ("amplitude", "phase", "D", "strain_gg", "half_set_frc"):
        assert k in kinds, k
    if meta:
        m2 = nxem.read_metadata(fn)
        assert m2["sample"]["atom_types"] == "Au, C" and m2["instrument"]["model"] == "JEM-ARM200F"
        assert m2["user"]["orcid"] == META["user"]["orcid"]


def test_demo_files_are_valid():
    d = os.path.join(os.path.dirname(__file__), "..", "data")
    files = [f for f in sorted(os.listdir(d)) if f.endswith(".nxs")] if os.path.isdir(d) else []
    if not files:
        pytest.skip("no demo files")
    for f in files:
        ok, problems, _ = nxem.validate(os.path.join(d, f))
        assert ok, (f, problems)


def test_missing_required_metadata():
    miss = {(s, k) for s, k, _ in nxem.missing_required(nxem.default_metadata())}
    assert {("sample", "atom_types"), ("sample", "preparation_date"), ("instrument", "vendor"),
            ("instrument", "model"), ("instrument", "emitter_type"), ("experiment", "start_time")} == miss
