# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Temporal workflow of the inline-holography action (one activity; no automatic retries, since a
reconstruction is long and deterministic)."""
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from .activities import reconstruct_focal_series
    from .models import ReconstructFocalSeriesInput, ReconstructFocalSeriesResult


@workflow.defn
class ReconstructFocalSeriesWorkflow:
    @workflow.run
    async def run(self, data: ReconstructFocalSeriesInput) -> ReconstructFocalSeriesResult:
        return await workflow.execute_activity(
            reconstruct_focal_series,
            data,
            start_to_close_timeout=timedelta(hours=data.timeout_hours),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
