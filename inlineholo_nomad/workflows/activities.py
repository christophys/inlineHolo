# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Activity of the inline-holography action: copy the focal series out of the NOMAD upload, run the
headless pipeline (:mod:`inlineholo.pipeline`) and add the NXem result to the upload, which NOMAD
then processes (the NXem file becomes an entry).

The NOMAD-specific parts (reading raw files, adding files to an upload) are small functions so that
:func:`run_in_upload` can be tested without a NOMAD installation.
"""
import os
import re
import shutil
import tempfile
import time

from temporalio import activity
from temporalio.exceptions import ApplicationError

from .models import ReconstructFocalSeriesInput, ReconstructFocalSeriesResult

_IMG_IDX = re.compile(r"^(.*?)(\d+)\.img$", re.IGNORECASE)
_EXT = (".nxs", ".h5", ".hdf5", ".tif", ".tiff", ".dm3", ".dm4", ".img", ".emd", ".mrc", ".st", ".npy", ".npz", ".ser")


def _safe(path):
    p = "/".join(s for s in str(path or "").replace("\\", "/").split("/") if s not in ("", "."))
    if any(s == ".." for s in p.split("/")):
        raise ApplicationError(f"invalid path {path!r}", non_retryable=True)
    return p


def input_files(path, is_file, listdir):
    """Raw files to copy for ``path``: the file (with the other frames of an indexed .img series) or the
    supported single-image files of a folder.  listdir(folder) -> [(name, is_file)]."""
    path = _safe(path)
    folder, name = (path.rsplit("/", 1) if "/" in path else ("", path))
    if is_file(path):
        m = _IMG_IDX.match(name)
        if not m:
            return [path], False
        names = [n for n, f in listdir(folder) if f and (mm := _IMG_IDX.match(n)) and mm.group(1) == m.group(1)]
        return [f"{folder}/{n}" if folder else n for n in names], False
    names = [n for n, f in listdir(path) if f and n.lower().endswith(_EXT)]
    if not names:
        raise ApplicationError(f"no file or supported images at {path!r} in the upload", non_retryable=True)
    return [f"{path}/{n}" if path else n for n in names], True


def run_in_upload(data: ReconstructFocalSeriesInput, *, is_file, listdir, copy_out, add_files, tmp_root=None,
                  progress=None) -> ReconstructFocalSeriesResult:
    """Copy -> reconstruct -> add result.  copy_out(raw_path, local_path); add_files(local_files, folder)."""
    from inlineholo.pipeline import run

    work = tempfile.mkdtemp(prefix="inlineholo_", dir=tmp_root)
    try:
        files, is_folder = input_files(data.path, is_file, listdir)
        src = os.path.join(work, "input")
        for k, f in enumerate(files):
            local = os.path.join(src, os.path.basename(f))
            os.makedirs(src, exist_ok=True)
            copy_out(f, local)
            if progress:
                progress("copy", (k + 1) / len(files), f"copied {f}")
        local_in = src if is_folder else os.path.join(src, os.path.basename(_safe(data.path)))
        if is_folder and not data.output_name:
            out_name = os.path.basename(_safe(data.path)) + f"_{data.method}.nxs"
        else:
            out_name = os.path.basename(data.output_name) if data.output_name else None
        try:
            res = run(local_in, os.path.join(work, "output"), method=data.method, n_iter=data.n_iter,
                      align=data.align, binning=data.binning, overrides=data.overrides(), out_name=out_name,
                      callback=progress)
        except (ValueError, KeyError, OSError) as e:
            raise ApplicationError(f"reconstruction failed: {e}", non_retryable=True) from e
        folder = _safe(data.output_folder)
        add_files(res["files"], folder)
        rel = [f"{folder}/{os.path.basename(f)}" if folder else os.path.basename(f) for f in res["files"]]
        return ReconstructFocalSeriesResult(
            upload_id=data.upload_id, files=rel, method=res["method"], n_images=res["n_images"], shape=res["shape"],
            R_final=res["R_final"], runtime_s=res["runtime_s"],
            message=f"{res['method']} reconstruction of {data.path} ({res['n_images']} images) written to "
                    f"{', '.join(rel)}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------- activity
@activity.defn
def reconstruct_focal_series(data: ReconstructFocalSeriesInput) -> ReconstructFocalSeriesResult:
    from .backend import get_backend

    be = get_backend()
    be.check_access(data.upload_id, data.user_id)
    uid = data.upload_id
    last = [0.0]

    def progress(stage, frac, msg):
        now = time.time()
        if now - last[0] > 2 or frac >= 1:
            last[0] = now
            activity.heartbeat(f"{stage} {frac:.0%}: {msg}")
            activity.logger.info(f"{stage} {frac:.0%}: {msg}")

    return run_in_upload(data, is_file=lambda p: be.is_file(uid, p), listdir=lambda p: be.listdir(uid, p),
                         copy_out=lambda p, local: be.copy_out(uid, p, local),
                         add_files=lambda files, folder: be.add_files(uid, files, folder),
                         tmp_root=be.tmp_root(), progress=progress)
