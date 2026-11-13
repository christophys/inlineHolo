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


# --------------------------------------------------------------------- NOMAD bindings
def _nomad_upload(data):
    from nomad.files import UploadFiles
    from nomad.processing import Upload

    try:
        upload = Upload.get(data.upload_id)
    except Exception as e:  # noqa: BLE001
        raise ApplicationError(f"upload {data.upload_id} not found", non_retryable=True) from e
    if data.user_id not in upload.writers:
        raise ApplicationError("only the main author and the co-authors of the upload can run this action on it",
                               non_retryable=True)
    if upload.published:
        raise ApplicationError("the upload is published; results cannot be added", non_retryable=True)
    return upload, UploadFiles.get(data.upload_id)


def _add_to_upload(upload, files, folder, tmp_root, attempts=60):
    """Add the files to the upload (NOMAD processes them); waits while the upload is being processed."""
    from nomad.processing.base import ProcessAlreadyRunning

    staged = tempfile.mkdtemp(prefix="inlineholo_add_", dir=tmp_root)
    ops = []
    for f in files:
        dst = os.path.join(staged, os.path.basename(f))
        shutil.copy2(f, dst)
        ops.append(dict(op="ADD", path=dst, target_dir=folder, temporary=True))
    for k in range(attempts):
        try:
            upload.reload()
            upload.process_upload(file_operations=ops, only_updated_files=True)
            return
        except ProcessAlreadyRunning:
            activity.heartbeat("waiting for the processing of the upload")
            time.sleep(5)
    raise ApplicationError("the upload stayed busy; the result could not be added", non_retryable=False)


@activity.defn
def reconstruct_focal_series(data: ReconstructFocalSeriesInput) -> ReconstructFocalSeriesResult:
    from nomad.config import config

    upload, uf = _nomad_upload(data)
    tmp_root = os.path.abspath(config.fs.tmp)
    os.makedirs(tmp_root, exist_ok=True)

    def is_file(p):
        return uf.raw_path_is_file(p)

    def listdir(p):
        return [(os.path.basename(e.path), e.is_file) for e in uf.raw_listdir(p)]

    def copy_out(p, local):
        with uf.raw_file(p, "rb") as fi, open(local, "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 22)

    last = [0.0]

    def progress(stage, frac, msg):
        now = time.time()
        if now - last[0] > 2 or frac >= 1:
            last[0] = now
            activity.heartbeat(f"{stage} {frac:.0%}: {msg}")
            activity.logger.info(f"{stage} {frac:.0%}: {msg}")

    return run_in_upload(data, is_file=is_file, listdir=listdir, copy_out=copy_out,
                         add_files=lambda files, folder: _add_to_upload(upload, files, folder, tmp_root),
                         tmp_root=tmp_root, progress=progress)
