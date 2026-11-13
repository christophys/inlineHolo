# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Temporal workflows of the batch / multi-stage action.

``BatchReconstructWorkflow`` (the action, CPU queue)
  1. ``plan_batch``: the series of the input paths
  2. one ``PrepareSeriesWorkflow`` child per series (at most ``max_parallel`` at a time):
     copy + load + FRWR alignment on a CPU worker
  3. optional: asks for the approval of the alignment (NOMAD signal input) and waits for the signal
     ``approve_alignment``
  4. one ``ReconstructSeriesWorkflow`` child per approved series: reconstruction on the GPU queue
     (CPU fallback if no GPU worker takes it within ``gpu_wait_minutes``), then NXem into the upload
     on a CPU worker
  5. ``finish_batch``: summary into the upload

The query ``progress`` returns the state of every series while the workflow runs.
"""
import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ChildWorkflowError, TimeoutError as TemporalTimeoutError, TimeoutType

with workflow.unsafe.imports_passed_through():
    from .batch_activities import (FinishInput, PlanInput, cleanup_batch, finish_batch, plan_batch, prepare_series,
                                   reconstruct_series, write_series_result)
    from .batch_models import (APPROVAL_SIGNAL, CPU_QUEUE, GPU_QUEUE, ApprovalInput, BatchReconstructInput,
                               BatchResult, PrepareResult, ReconstructOutcome, ReconstructTask, SeriesResult,
                               SeriesTask)

HEARTBEAT = timedelta(seconds=30)      # SDK sends heartbeats at most every 0.8 x this: cancellation latency
ONCE = RetryPolicy(maximum_attempts=1)
TWICE = RetryPolicy(maximum_attempts=2, non_retryable_error_types=["StopRequested", "ValueError"])


def _why(e):
    c = e
    while getattr(c, "cause", None) is not None:
        c = c.cause
    return str(c) or type(c).__name__


@workflow.defn
class PrepareSeriesWorkflow:
    @workflow.run
    async def run(self, task: SeriesTask) -> PrepareResult:
        return await workflow.execute_activity(
            prepare_series, task, task_queue=CPU_QUEUE, retry_policy=TWICE, heartbeat_timeout=HEARTBEAT,
            start_to_close_timeout=timedelta(hours=task.stage_timeout_hours))


@workflow.defn
class ReconstructSeriesWorkflow:
    @workflow.run
    async def run(self, rt: ReconstructTask) -> ReconstructOutcome:
        t = rt.task
        kw = dict(retry_policy=TWICE, heartbeat_timeout=HEARTBEAT,
                  start_to_close_timeout=timedelta(hours=t.stage_timeout_hours))
        try:
            res = await workflow.execute_activity(
                reconstruct_series, t, task_queue=rt.queue,
                schedule_to_start_timeout=timedelta(minutes=rt.wait_minutes) if rt.fallback_queue else None, **kw)
        except ActivityError as e:
            c = e.cause
            if not (rt.fallback_queue and isinstance(c, TemporalTimeoutError)
                    and c.type == TimeoutType.SCHEDULE_TO_START):
                raise
            workflow.logger.info(f"{t.name}: no worker on {rt.queue} within {rt.wait_minutes} min, "
                                 f"running on {rt.fallback_queue}")
            res = await workflow.execute_activity(reconstruct_series, t, task_queue=rt.fallback_queue, **kw)
        f = await workflow.execute_activity(write_series_result, t, task_queue=CPU_QUEUE, retry_policy=TWICE,
                                            heartbeat_timeout=HEARTBEAT, start_to_close_timeout=timedelta(hours=2))
        return ReconstructOutcome(result=res, file=f)


@workflow.defn
class BatchReconstructWorkflow:
    def __init__(self):
        self.approval: ApprovalInput | None = None
        self.state: dict = {"stage": "planning", "series": {}}

    @workflow.signal(name=APPROVAL_SIGNAL)
    async def approve_alignment(self, data: ApprovalInput) -> None:
        self.approval = data

    @workflow.query
    def progress(self) -> dict:
        return self.state

    def _set(self, k, name, status, **kw):
        self.state["series"][str(k)] = {"name": name, "status": status, **kw}

    @workflow.run
    async def run(self, data: BatchReconstructInput) -> BatchResult:
        try:
            return await self._run(data)
        except (asyncio.CancelledError, Exception):
            # cancelled or failed: remove the working folders of the series, then end as before
            self.state["stage"] = "cleaning up"
            await workflow.execute_activity(cleanup_batch, workflow.info().workflow_id, task_queue=CPU_QUEUE,
                                            start_to_close_timeout=timedelta(minutes=10),
                                            retry_policy=RetryPolicy(maximum_attempts=3))
            raise

    async def _run(self, data: BatchReconstructInput) -> BatchResult:
        wid = workflow.info().workflow_id
        started = workflow.now().strftime("%Y-%m-%dT%H%M%S")
        tasks = await workflow.execute_activity(plan_batch, PlanInput(data=data, instance_id=wid),
                                                task_queue=CPU_QUEUE, retry_policy=RetryPolicy(maximum_attempts=3),
                                                start_to_close_timeout=timedelta(minutes=30))
        for t in tasks:
            self._set(t.index, t.name, "queued")
        sem = asyncio.Semaphore(data.max_parallel)

        # ---- stage 1: copy, load, align (CPU workers, in parallel)
        self.state["stage"] = "alignment" if data.align else "loading"

        async def prep(t: SeriesTask) -> PrepareResult:
            async with sem:
                self._set(t.index, t.name, "aligning" if data.align else "loading")
                try:
                    r = await workflow.execute_child_workflow(PrepareSeriesWorkflow.run, t, task_queue=CPU_QUEUE,
                                                              id=f"{wid}-prepare-{t.index:03d}")
                except ChildWorkflowError as e:
                    r = PrepareResult(index=t.index, name=t.name, ok=False, message=_why(e))
                self._set(t.index, t.name, "prepared" if r.ok else "alignment failed", R_align=r.R_align,
                          message=r.message)
                return r

        preps = await asyncio.gather(*[prep(t) for t in tasks])
        ok = [p.index for p in preps if p.ok]
        approved = set(ok)

        # ---- stage 2: approval of the alignment (human in the loop)
        if data.approve_alignment and ok:
            self.state["stage"] = "waiting for approval"
            rows = "\n".join(f"| {p.index} | {p.name} | {'✔' if p.ok else '✘ ' + p.message[:60]} | {p.n_images} | "
                             f"{'×'.join(map(str, p.shape))} | {'' if p.R_align is None else f'{p.R_align:.4f}'} | "
                             f"{'' if p.max_drift_px is None else f'{p.max_drift_px:.1f}'} |" for p in preps)
            content = ("| # | series | alignment | images | size (px) | R (FRWR) | max. drift (px) |\n"
                       "|---|---|---|---|---|---|---|\n" + rows +
                       "\n\nApprove all, or list the indices of the series to reconstruct.")
            await workflow.execute_activity(
                "request_signal_input_activity",
                {"action_instance_id": wid, "user_id": data.user_id, "signal_fn_name": APPROVAL_SIGNAL,
                 "title": "Approve the alignment", "content": content,
                 "description": f"{len(ok)} of {len(preps)} series aligned; the reconstruction starts after the "
                                f"approval.", "initial_data": {"approve_all": True, "approved": ok, "comment": ""}},
                task_queue=CPU_QUEUE, start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=3))
            try:
                await workflow.wait_condition(lambda: self.approval is not None,
                                              timeout=timedelta(hours=data.approval_timeout_hours))
            except asyncio.TimeoutError:
                self.approval = ApprovalInput(approve_all=False, approved=[], comment="no approval in time")
            a = self.approval
            approved = set(ok) if a.approve_all else set(a.approved) & set(ok)
            self.state["approval"] = a.model_dump()

        # ---- stage 3: reconstruction (GPU queue, CPU fallback) + NXem into the upload
        self.state["stage"] = "reconstruction"
        queue = GPU_QUEUE if data.reconstruct_on_gpu else CPU_QUEUE
        fallback = CPU_QUEUE if data.reconstruct_on_gpu else ""
        by_index = {t.index: t for t in tasks}
        results: dict[int, SeriesResult] = {}
        for p in preps:
            if not p.ok:
                results[p.index] = SeriesResult(index=p.index, name=p.name, status="alignment failed",
                                                message=p.message)
            elif p.index not in approved:
                results[p.index] = SeriesResult(index=p.index, name=p.name, status="not approved", R_align=p.R_align)
                self._set(p.index, p.name, "not approved")

        async def rec(p: PrepareResult):
            t = by_index[p.index]
            async with sem:
                self._set(t.index, t.name, "reconstructing", R_align=p.R_align)
                try:
                    o = await workflow.execute_child_workflow(
                        ReconstructSeriesWorkflow.run,
                        ReconstructTask(task=t, queue=queue, fallback_queue=fallback,
                                        wait_minutes=data.gpu_wait_minutes),
                        id=f"{wid}-reconstruct-{t.index:03d}", task_queue=CPU_QUEUE)
                    r = SeriesResult(index=t.index, name=t.name, status="done", file=o.file, R_align=p.R_align,
                                     R_final=o.result.R_final, reconstructed_on=o.result.worker_queue,
                                     runtime_s=o.result.runtime_s, message=o.result.message)
                except ChildWorkflowError as e:
                    r = SeriesResult(index=t.index, name=t.name, status="failed", R_align=p.R_align, message=_why(e))
                self._set(t.index, t.name, r.status, R_final=r.R_final, file=r.file, on=r.reconstructed_on)
                results[t.index] = r

        await asyncio.gather(*[rec(p) for p in preps if p.ok and p.index in approved])

        # ---- summary
        self.state["stage"] = "summary"
        ordered = [results[k] for k in sorted(results)]
        summary = await workflow.execute_activity(
            finish_batch, FinishInput(instance_id=wid, upload_id=data.upload_id, output_folder=data.output_folder,
                                      started=started, method=data.method, results=ordered),
            task_queue=CPU_QUEUE, retry_policy=RetryPolicy(maximum_attempts=3),
            start_to_close_timeout=timedelta(hours=1))
        self.state["stage"] = "done"
        return BatchResult(upload_id=data.upload_id, n_series=len(tasks),
                           n_done=sum(r.status == "done" for r in ordered), results=ordered, summary_file=summary)
