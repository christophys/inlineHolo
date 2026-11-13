# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""NOMAD action: reconstruction of a focal series of a NOMAD upload on a NOMAD (CPU) worker.

Started from the NOMAD GUI (Actions, form generated from
:class:`~inlineholo_nomad.actions.models.ReconstructFocalSeriesInput`), from the inline-holography
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


class InlineHoloActionEntryPoint(ActionEntryPoint):
    task_queue: str = Field(default=TaskQueue.CPU, description="task queue of the worker")

    def load(self):
        from nomad.actions import Action

        from .activities import reconstruct_focal_series
        from .workflows import ReconstructFocalSeriesWorkflow

        return Action(task_queue=self.task_queue, workflow=ReconstructFocalSeriesWorkflow,
                      activities=[reconstruct_focal_series])


inlineholo_reconstruction = InlineHoloActionEntryPoint(
    name="Inline holography: reconstruct a focal series",
    description="Exit-wave reconstruction (FRWR, TIE, Schiske, MAL, ...) of a TEM focal series in a NOMAD upload, "
                "optionally with FRWR alignment; the amplitude and phase are added to the upload as NXem.",
)
