# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Batch / multi-stage action on a real Temporal server (the Temporal CLI dev server), without NOMAD.

Two workers as in NOMAD: a "CPU" worker (workflows + CPU activities + a stand-in for NOMAD's
request_signal_input_activity) and a "GPU" worker (only the reconstruction activity).  Uploads are
local folders (LocalBackend).  Needs temporalio and the ``temporal`` CLI: on the PATH, in
$TEMPORAL_CLI, or downloadable by temporalio; otherwise the tests are skipped.
"""
import asyncio
import os
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("temporalio")
import inlineholo as ih  # noqa: E402
from temporalio import activity  # noqa: E402
from temporalio.client import Client  # noqa: E402
from temporalio.contrib.pydantic import pydantic_data_converter  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Worker  # noqa: E402

from inlineholo_nomad.workflows import backend as be_mod  # noqa: E402
from inlineholo_nomad.workflows.batch_activities import (cleanup_batch, finish_batch, plan_batch,  # noqa: E402
                                                         prepare_series, reconstruct_series, write_series_result)
from inlineholo_nomad.workflows.batch_models import (CPU_QUEUE, GPU_QUEUE, ApprovalInput,  # noqa: E402
                                                   BatchReconstructInput, BatchResult)
from inlineholo_nomad.workflows.batch_workflows import (BatchReconstructWorkflow, PrepareSeriesWorkflow,  # noqa: E402
                                                      ReconstructSeriesWorkflow)

SIGNAL_REQUESTS = []


@activity.defn(name="request_signal_input_activity")
async def fake_request_signal_input(data: dict) -> None:
    """Stand-in for NOMAD's activity (which stores the request so that the GUI shows a form)."""
    SIGNAL_REQUESTS.append(data)


def _cli():
    for c in (os.environ.get("TEMPORAL_CLI"), shutil.which("temporal"), "/home/claude/temporal/temporal"):
        if c and os.path.isfile(c):
            return c
    return None


@pytest.fixture(scope="module")
def upload_root(tmp_path_factory):
    """Upload U1: raw/a.nxs (needs alignment: drift), raw/b.nxs, raw/broken.nxs; folder frames/ (one series)."""
    root = tmp_path_factory.mktemp("uploads")
    raw = root / "U1" / "raw"
    raw.mkdir(parents=True)
    df = np.array([-6000., -3000., 0., 3000., 6000.])
    p = ih.simulate.mr_parameters(pixel_size_nm=2.0, defoci_nm=df)
    for name, seed in (("a", 3), ("b", 4)):
        w = ih.simulate.mr_object(n=256, pixel=2.0, seed=seed, film_nm=2.0)
        I, _ = ih.simulate.simulate_focal_series(w, p, None, n_defocus=1, n_tilt=1)
        if name == "a":
            I = np.array([np.roll(f, (k - 2, 2 - k), (0, 1)) for k, f in enumerate(I)])
        ih.nxem.write_focal_series(raw / f"{name}.nxs", I + 1e-3, p, is_simulation=True)
    (raw / "broken.nxs").write_bytes(b"not a nexus file")
    fr = root / "U1" / "frames"
    fr.mkdir()
    for z, f in zip(df, I):
        np.save(fr / f"F{z:+.0f}nm.npy", f + 1e-3)
    return root


@pytest.fixture(scope="module")
def temporal():
    cli = _cli()
    try:
        env = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            WorkflowEnvironment.start_local(data_converter=pydantic_data_converter, dev_server_existing_path=cli))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no Temporal dev server: {e}")
    yield env
    asyncio.new_event_loop().run_until_complete(env.shutdown())


async def _run(env, upload_root, data, gpu_worker=True, on_running=None):
    be_mod.set_backend(be_mod.LocalBackend(str(upload_root)))
    client = await Client.connect(env.client.service_client.config.target_host,
                                  data_converter=pydantic_data_converter)
    ex = ThreadPoolExecutor(8)
    cpu = Worker(client, task_queue=CPU_QUEUE,
                 workflows=[BatchReconstructWorkflow, PrepareSeriesWorkflow, ReconstructSeriesWorkflow],
                 activities=[plan_batch, prepare_series, reconstruct_series, write_series_result, finish_batch,
                             cleanup_batch, fake_request_signal_input], activity_executor=ex)
    workers = [cpu]
    if gpu_worker:
        workers.append(Worker(client, task_queue=GPU_QUEUE, activities=[reconstruct_series], activity_executor=ex))
    async with cpu:
        if gpu_worker:
            await workers[1].__aenter__()
        try:
            wid = f"inlineholo-test-{uuid.uuid4()}"
            h = await client.start_workflow(BatchReconstructWorkflow.run, data, id=wid, task_queue=CPU_QUEUE)
            if on_running:
                await on_running(h)
            res = await h.result()
        finally:
            if gpu_worker:
                await workers[1].__aexit__(None, None, None)
    return res if isinstance(res, BatchResult) else BatchResult.model_validate(res), wid


def test_batch_cpu_alignment_gpu_reconstruction(temporal, upload_root):
    data = BatchReconstructInput(upload_id="U1", user_id="me", paths=["raw"], method="frwr", n_iter=3, align=True,
                                 binning=2, max_parallel=2, output_folder="batch1")
    res, wid = asyncio.new_event_loop().run_until_complete(_run(temporal, upload_root, data))
    st = {r.name: r for r in res.results}
    assert res.n_series == 3 and res.n_done == 2, res
    assert st["broken"].status == "alignment failed"
    for n in ("a", "b"):
        r = st[n]
        assert r.status == "done" and r.reconstructed_on == GPU_QUEUE and r.R_align is not None
        f = upload_root / "U1" / r.file
        assert f.is_file() and r.file == f"batch1/{n}_frwr.nxs"
        stack, p, _, recs = ih.nxem.read_focal_series(str(f))
        assert stack.shape[0] == 5 and stack.shape[1] < 256          # aligned and cropped
    assert (upload_root / "U1" / res.summary_file).is_file()
    work = upload_root / "_work" / wid
    assert not any(d.startswith("series_") for d in os.listdir(work))   # working folders removed


def test_batch_gpu_fallback_and_approval(temporal, upload_root):
    """No GPU worker: reconstruction falls back to the CPU queue; approval of only one series."""
    SIGNAL_REQUESTS.clear()
    data = BatchReconstructInput(upload_id="U1", user_id="me", paths=["raw/a.nxs", "frames"], folder_mode="one_series",
                                 method="tie_full", align=False, approve_alignment=True, reconstruct_on_gpu=True,
                                 gpu_wait_minutes=0.05, output_folder="batch2")

    async def approve(h):
        for _ in range(600):
            st = await h.query(BatchReconstructWorkflow.progress)
            if st.get("stage") == "waiting for approval":
                break
            await asyncio.sleep(0.2)
        assert SIGNAL_REQUESTS and SIGNAL_REQUESTS[0]["signal_fn_name"] == "approve_alignment"
        assert "| 1 | frames |" in SIGNAL_REQUESTS[0]["content"]
        await h.signal(BatchReconstructWorkflow.approve_alignment, ApprovalInput(approve_all=False, approved=[1]))

    res, _ = asyncio.new_event_loop().run_until_complete(_run(temporal, upload_root, data, gpu_worker=False,
                                                              on_running=approve))
    st = {r.index: r for r in res.results}
    assert st[0].status == "not approved" and st[1].status == "done"
    assert st[1].reconstructed_on == CPU_QUEUE and st[1].file == "batch2/frames_tie_full.nxs"
    assert (upload_root / "U1" / "batch2" / "frames_tie_full.nxs").is_file()


def test_single_action_workflow(temporal, upload_root):
    """The single-series action passes Temporal's workflow sandbox and runs end to end."""
    from inlineholo_nomad.workflows.activities import reconstruct_focal_series
    from inlineholo_nomad.workflows.models import ReconstructFocalSeriesInput
    from inlineholo_nomad.workflows.single_workflows import ReconstructFocalSeriesWorkflow

    async def go():
        be_mod.set_backend(be_mod.LocalBackend(str(upload_root)))
        client = await Client.connect(temporal.client.service_client.config.target_host,
                                      data_converter=pydantic_data_converter)
        async with Worker(client, task_queue=CPU_QUEUE, workflows=[ReconstructFocalSeriesWorkflow],
                          activities=[reconstruct_focal_series], activity_executor=ThreadPoolExecutor(2)):
            return await client.execute_workflow(
                ReconstructFocalSeriesWorkflow.run,
                ReconstructFocalSeriesInput(upload_id="U1", user_id="me", path="raw/b.nxs", method="tie_full",
                                            output_folder="single"),
                id=f"inlineholo-single-{uuid.uuid4()}", task_queue=CPU_QUEUE)

    res = asyncio.new_event_loop().run_until_complete(go())
    assert res.files == ["single/b_tie_full.nxs"] and (upload_root / "U1" / "single" / "b_tie_full.nxs").is_file()


def test_batch_cancel_stops_running_reconstruction(temporal, upload_root, caplog):
    """Cancelling the action stops the running reconstruction activity (via its heartbeats) within seconds."""
    import logging
    import time
    from temporalio.client import WorkflowFailureError
    from temporalio.exceptions import CancelledError
    caplog.set_level(logging.WARNING, logger="temporalio.activity")
    data = BatchReconstructInput(upload_id="U1", user_id="me", paths=["raw/a.nxs"], method="frwr", n_iter=5000,
                                 align=False, reconstruct_on_gpu=False, output_folder="batch3")
    t = {}

    def stopped():
        return [r for r in caplog.records if r.exc_info and isinstance(r.exc_info[1], CancelledError)
                and "reconstruct_series" in r.getMessage()]

    async def cancel(h):
        for _ in range(600):
            st = await h.query(BatchReconstructWorkflow.progress)
            if st.get("stage") == "reconstruction":
                break
            await asyncio.sleep(0.2)
        await asyncio.sleep(2)
        t["cancel"] = time.time()
        await h.cancel()
        for _ in range(600):                          # keep the workers running until the activity stopped
            if stopped():
                break
            await asyncio.sleep(0.1)

    with pytest.raises(WorkflowFailureError):
        asyncio.new_event_loop().run_until_complete(_run(temporal, upload_root, data, gpu_worker=False,
                                                         on_running=cancel))
    s = stopped()
    assert s and s[0].created - t["cancel"] < 30      # the running FRWR loop was interrupted
    assert not (upload_root / "U1" / "batch3").exists()
    work = upload_root / "_work"
    assert not [d for w in work.iterdir() for d in os.listdir(w) if d.startswith("series_")]   # cleaned up
