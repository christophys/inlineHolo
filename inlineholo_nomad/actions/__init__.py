# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""NOMAD action: reconstruction of a focal series of a NOMAD upload on a NOMAD (CPU) worker.

Started from the NOMAD GUI (Actions, form generated from
:class:`~inlineholo_nomad.workflows.models.ReconstructFocalSeriesInput`), from the inline-holography
dashboard ("run as NOMAD action"), via the API
(``POST {api}/v1/actions/inlineholo_nomad.actions:inlineholo_reconstruction/start`` with
``{"data": {"upload_id": ..., "path": ..., "method": "frwr"}}``) or in Python::

    from nomad.actions.manager import start_action
    start_action(action_name="inlineholo_nomad.actions:inlineholo_reconstruction",
                 data=ReconstructFocalSeriesInput(upload_id=..., user_id=..., path=...))
"""
from nomad.actions import TaskQueue
from nomad.config.models.plugins import ActionEntryPoint
from pydantic import Field

ACTION_ID = "inlineholo_nomad.actions:inlineholo_reconstruction"
BATCH_ACTION_ID = "inlineholo_nomad.actions:inlineholo_batch"


class InlineHoloActionEntryPoint(ActionEntryPoint):
    task_queue: str = Field(default=TaskQueue.CPU, description="task queue of the worker")

    def load(self):
        from nomad.actions import Action

        from ..workflows.activities import reconstruct_focal_series
        from ..workflows.single_workflows import ReconstructFocalSeriesWorkflow

        return Action(task_queue=self.task_queue, workflow=ReconstructFocalSeriesWorkflow,
                      activities=[reconstruct_focal_series])


inlineholo_reconstruction = InlineHoloActionEntryPoint(
    name="Inline holography: reconstruct a focal series",
    description="Exit-wave reconstruction (FRWR, TIE, Schiske, MAL, ...) of a TEM focal series in a NOMAD upload, "
                "optionally with FRWR alignment; the amplitude and phase are added to the upload as NXem.",
)


class InlineHoloBatchEntryPoint(ActionEntryPoint):
    """Batch / multi-stage action: the workflows and the CPU activities run on the CPU workers, the
    reconstruction activity is also registered on the GPU task queue (``uv run poe gpuworker``)."""

    task_queue: str = Field(default=TaskQueue.CPU, description="task queue of the workflows")

    def load(self):
        from nomad.actions import Action

        from ..workflows.batch_activities import (cleanup_batch, finish_batch, plan_batch, prepare_series,
                                                  reconstruct_series, write_series_result)
        from ..workflows.batch_workflows import (BatchReconstructWorkflow, PrepareSeriesWorkflow,
                                                 ReconstructSeriesWorkflow)

        return Action(task_queue=self.task_queue, workflow=BatchReconstructWorkflow,
                      child_workflows=[PrepareSeriesWorkflow, ReconstructSeriesWorkflow],
                      activities=[plan_batch, prepare_series, reconstruct_series, write_series_result, finish_batch,
                                  cleanup_batch],
                      task_queue_activities={TaskQueue.GPU: [reconstruct_series]})


inlineholo_batch = InlineHoloBatchEntryPoint(
    name="Inline holography: batch reconstruction (alignment on CPU, reconstruction on GPU)",
    description="Several focal series of an upload in parallel: FRWR alignment on CPU workers, optional approval of "
                "the alignment, reconstruction on GPU workers (CPU fallback), NXem results and a summary in the "
                "upload.",
)
