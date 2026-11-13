# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Activities of the batch / multi-stage action.

Large arrays never go through Temporal (payloads are limited to a few MB): every series has a working
folder in the shared folder of the action instance; the activities pass only paths and small records.

* ``plan_batch``          (CPU) expands the input paths into series and creates their working folders
* ``prepare_series``      (CPU) copies the raw files out of the upload, loads them, FRWR alignment
* ``reconstruct_series``  (GPU or CPU queue) reconstruction of the prepared series
* ``write_series_result`` (CPU) NXem file, added to the upload
* ``finish_batch``        (CPU) summary (JSON) into the upload, removes the working folders

Long activities heartbeat every few seconds (also during long single steps, from a helper thread) and
stop when the action is cancelled.
"""
import contextvars
import json
import os
import re
import shutil
import threading
import time

from temporalio import activity
from temporalio.exceptions import ApplicationError

from .activities import _EXT, _IMG_IDX, _safe, input_files
from .backend import get_backend
from .batch_models import (BatchReconstructInput, PrepareResult, SeriesResult, SeriesTask, StageResult)
from pydantic import BaseModel

_STACK_EXT = (".nxs", ".h5", ".hdf5", ".dm3", ".dm4", ".tif", ".tiff", ".emd", ".mrc", ".st", ".npy", ".npz", ".ser")


class PlanInput(BaseModel):
    data: BatchReconstructInput
    instance_id: str


class FinishInput(BaseModel):
    instance_id: str
    upload_id: str
    output_folder: str
    started: str = ""
    method: str = ""
    results: list[SeriesResult]


class Heartbeat:
    """Heartbeats from a helper thread (long single steps) + progress messages + cancellation."""

    def __init__(self, every_s=5.0):
        self.every = every_s
        self.msg = "started"
        self._stop = threading.Event()
        self._ctx = contextvars.copy_context()
        self._t = threading.Thread(target=lambda: self._ctx.run(self._loop), daemon=True)

    def _loop(self):
        while not self._stop.wait(self.every):
            try:
                activity.heartbeat(self.msg)
            except Exception:  # noqa: BLE001
                return

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()

    def progress(self, stage, frac, msg):
        """Pipeline callback: remembers the message; raises if the activity was cancelled."""
        self.msg = f"{stage} {frac:.0%}: {msg}"
        if activity.is_cancelled():
            from inlineholo.control import StopRequested
            raise StopRequested()


def _name(path):
    return os.path.splitext(os.path.basename(path.rstrip("/")))[0] or "series"


@activity.defn
def plan_batch(inp: PlanInput) -> list[SeriesTask]:
    d, be = inp.data, get_backend()
    be.check_access(d.upload_id, d.user_id)
    uid = d.upload_id
    series = []                                             # (name, sources, is_folder)
    for p in d.paths:
        p = _safe(p)
        if be.is_file(uid, p):
            files, _ = input_files(p, lambda q: be.is_file(uid, q), lambda q: be.listdir(uid, q))
            series.append((_name(p), files, False))
            continue
        try:
            items = be.listdir(uid, p)
        except Exception as e:  # noqa: BLE001
            raise ApplicationError(f"{p!r} not found in the upload", non_retryable=True) from e
        files = [n for n, f in items if f and n.lower().endswith(_EXT)]
        if d.folder_mode == "one_series":
            if not files:
                raise ApplicationError(f"no images in {p!r}", non_retryable=True)
            series.append((_name(p), [f"{p}/{n}" if p else n for n in files], True))
            continue
        seen_img = set()
        for n in files:
            m = _IMG_IDX.match(n)
            if m:                                           # indexed .img series: one series per prefix
                if m.group(1) in seen_img:
                    continue
                seen_img.add(m.group(1))
                src, _ = input_files(f"{p}/{n}" if p else n, lambda q: be.is_file(uid, q),
                                     lambda q: be.listdir(uid, q))
                series.append((m.group(1).rstrip("_-") or _name(n), src, False))
            elif n.lower().endswith(_STACK_EXT):
                series.append((_name(n), [f"{p}/{n}" if p else n], False))
        if not files:
            raise ApplicationError(f"no focal series in {p!r}", non_retryable=True)
    if not series:
        raise ApplicationError("no focal series found", non_retryable=True)
    root = be.work_root(inp.instance_id)
    tasks, used = [], set()
    for k, (name, src, is_folder) in enumerate(series):
        nm = re.sub(r"[^A-Za-z0-9_.+-]", "_", name)
        while nm in used:
            nm += f"_{k}"
        used.add(nm)
        wd = os.path.join(root, f"series_{k:03d}")
        os.makedirs(wd, exist_ok=True)
        tasks.append(SeriesTask(index=k, name=nm, upload_id=uid, user_id=d.user_id, sources=src, is_folder=is_folder,
                                work_dir=wd, method=d.method, n_iter=d.n_iter, align=d.align, binning=d.binning,
                                overrides=d.overrides(), output_folder=_safe(d.output_folder),
                                stage_timeout_hours=d.stage_timeout_hours))
    activity.logger.info(f"batch of {len(tasks)} series: {', '.join(t.name for t in tasks)}")
    return tasks


@activity.defn
def prepare_series(task: SeriesTask) -> PrepareResult:
    """Copy out of the upload, load, FRWR alignment; the prepared series is stored in the work folder."""
    from inlineholo.control import StopRequested
    from inlineholo.pipeline import prepare

    be = get_backend()
    src = os.path.join(task.work_dir, "input")
    os.makedirs(src, exist_ok=True)
    with Heartbeat() as hb:
        try:
            for k, f in enumerate(task.sources):
                hb.progress("copy", (k + 1) / len(task.sources), f)
                be.copy_out(task.upload_id, f, os.path.join(src, os.path.basename(f)))
            path = src if task.is_folder else os.path.join(src, os.path.basename(task.sources[0]))
            pr = prepare(path, task.overrides, task.align, task.binning, callback=hb.progress)
            pr.save(os.path.join(task.work_dir, "prepared"))
        except StopRequested:
            raise
        except Exception as e:  # noqa: BLE001  -- one bad series must not stop the batch
            activity.logger.warning(f"{task.name}: preparation failed: {e!r}")
            return PrepareResult(index=task.index, name=task.name, ok=False, message=f"{type(e).__name__}: {e}")
        finally:
            shutil.rmtree(src, ignore_errors=True)
    a = pr.align_summary or {}
    return PrepareResult(index=task.index, name=task.name, ok=True, n_images=int(len(pr.stack)),
                         shape=list(pr.stack.shape[1:]), R_align=a.get("R"), max_drift_px=a.get("max_drift_px"),
                         defocus_range_nm=[float(pr.params.defoci_nm.min()), float(pr.params.defoci_nm.max())],
                         message="aligned" if task.align else "loaded")


@activity.defn
def reconstruct_series(task: SeriesTask) -> StageResult:
    """Reconstruction of the prepared series (registered on the CPU and on the GPU task queue)."""
    from inlineholo.pipeline import Prepared, r_final, reconstruct_prepared

    t0 = time.time()
    queue = activity.info().task_queue
    with Heartbeat() as hb:
        pr = Prepared.load(os.path.join(task.work_dir, "prepared"))
        opts = {}
        if task.method in ("frwr", "frwr_gf", "gpfrwr"):
            opts["backend"] = "auto"                   # GPU (CuPy/PyTorch) if the worker has one
        rec = reconstruct_prepared(pr, task.method, task.n_iter, callback=hb.progress, **opts)
    import pickle
    with open(os.path.join(task.work_dir, "reconstruction.pkl"), "wb") as fh:
        pickle.dump(rec, fh)
    return StageResult(ok=True, R_final=r_final(rec), runtime_s=round(time.time() - t0, 1), worker_queue=queue,
                       message=f"{task.method} on {queue}")


@activity.defn
def write_series_result(task: SeriesTask) -> str:
    """NXem file of the series, added to the upload; returns its path in the upload."""
    import pickle

    from inlineholo.pipeline import Prepared, write_result

    pr = Prepared.load(os.path.join(task.work_dir, "prepared"))
    with open(os.path.join(task.work_dir, "reconstruction.pkl"), "rb") as fh:
        rec = pickle.load(fh)
    out = os.path.join(task.work_dir, "output", f"{task.name}_{task.method}.nxs")
    with Heartbeat():
        write_result(out, pr, rec)
        get_backend().add_files(task.upload_id, [out], task.output_folder)
    return f"{task.output_folder}/{os.path.basename(out)}" if task.output_folder else os.path.basename(out)


@activity.defn
def finish_batch(inp: FinishInput) -> str:
    """Summary of the batch (JSON) into the upload; removes the shared working folders."""
    be = get_backend()
    root = be.work_root(inp.instance_id)
    stamp = (inp.started or time.strftime("%Y-%m-%dT%H%M%S")).replace(":", "")
    fn = os.path.join(root, f"batch_summary_{stamp}.json")
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump({"action_instance_id": inp.instance_id, "upload_id": inp.upload_id, "method": inp.method,
                   "results": [r.model_dump() for r in inp.results]}, fh, indent=2)
    try:
        be.add_files(inp.upload_id, [fn], inp.output_folder)
    finally:
        for d in os.listdir(root):
            if d.startswith("series_"):
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
    return f"{inp.output_folder}/{os.path.basename(fn)}" if inp.output_folder else os.path.basename(fn)


@activity.defn
def cleanup_batch(instance_id: str) -> None:
    """Remove the working folders of the series (after cancellation or failure)."""
    root = get_backend().work_root(instance_id)
    for d in os.listdir(root):
        if d.startswith("series_"):
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
