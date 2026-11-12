# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Rewrite NXem files of inlineholo <= 0.2 in the current (pynxtools-validated) layout.

usage: python tools_upgrade_nxem.py old.nxs new.nxs [metadata.yaml]
"""
import json
import sys

import h5py
import numpy as np

from inlineholo import nxem
from inlineholo.reconstruct import Reconstruction


def upgrade(old, new, metadata=None, derived=True):
    stack, p, gt, recs = nxem.read_focal_series(old)
    info = {}
    with h5py.File(old, "r") as f:
        e = f["entry1"]
        desc = nxem._get(e, "experiment_description", "focal series for inline electron holography")
        is_sim = bool(nxem._get(e, "sample1/is_simulation", False))
        extra = nxem._get(e, "measurement/instrument/inline_holography/extra_metadata", None)
        for k, im in (e["roi1/img"].items() if "roi1/img" in e else []):
            if "image_2d" in im and "imag" in im["image_2d"]:
                lab = nxem._get(im, "reconstruction/label", k)
                info[lab] = dict(method=nxem._get(im, "reconstruction/method", lab),
                                 options=json.loads(nxem._get(im, "reconstruction/options", "{}") or "{}"),
                                 runtime=float(nxem._get(im, "reconstruction/runtime", 0.0) or 0.0),
                                 background=im["reconstruction/incoherent_background"][()]
                                 if "reconstruction/incoherent_background" in im else None)
    nxem.write_focal_series(new, stack, p, ground_truth=gt, description=desc, is_simulation=is_sim,
                            extra_metadata=json.loads(extra) if extra else None, metadata=metadata)
    for lab, w in recs.items():
        d = info.get(lab, {})
        r = Reconstruction(w, d.get("method", lab), d.get("options", {}), [], d.get("runtime", 0.0),
                           d.get("background"))
        nxem.write_reconstruction(new, r, p, label=lab, derived=derived)
    return new


if __name__ == "__main__":
    meta = None
    if len(sys.argv) > 3:
        import yaml
        meta = yaml.safe_load(open(sys.argv[3]))
    upgrade(sys.argv[1], sys.argv[2], meta)
    print(nxem.validate(sys.argv[2])[:2])
