# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Inputs, intermediate records and results of the batch / multi-stage action."""
from typing import Literal

from pydantic import BaseModel, Field

from .models import Method

APPROVAL_SIGNAL = "approve_alignment"


class BatchReconstructInput(BaseModel):
    """Reconstruct several focal series of a NOMAD upload in parallel on the NOMAD workers:
    alignment on CPU workers, reconstruction on GPU workers (if available), optional approval of the
    alignment before the reconstruction starts."""

    upload_id: str = Field(..., description="ID of the NOMAD upload with the focal series (not published)")
    user_id: str = Field("", description="set by NOMAD to the user who starts the action")
    paths: list[str] = Field(..., min_length=1,
                             description="files or folders in the upload. A folder holds either several focal-series "
                                         "files (NXem, DM3/DM4/TIFF stacks; each one is a series) or the single "
                                         "images of one series (see folder_mode)")
    folder_mode: Literal["series_files", "one_series"] = Field(
        "series_files", description="series_files: every stack/NXem file in a folder is one series; one_series: "
                                    "a folder is one series of single images (ordered by the defocus in their names)")
    method: Method = Field("frwr", description="reconstruction method")
    n_iter: int = Field(50, ge=1, le=5000, description="iterations of the iterative methods")
    align: bool = Field(True, description="FRWR alignment (drift, rotation, magnification, distortions vs. defocus) "
                                          "before the reconstruction")
    binning: int = Field(2, ge=1, le=8, description="binning of the FRWR alignment fit")
    approve_alignment: bool = Field(False, description="wait for the approval of the alignment results (NOMAD asks "
                                                       "for it) before the reconstructions start")
    approval_timeout_hours: float = Field(72.0, gt=0, le=720, description="how long to wait for the approval")
    reconstruct_on_gpu: bool = Field(True, description="run the reconstructions on the GPU task queue; falls back to "
                                                       "the CPU queue if no GPU worker picks them up in time")
    gpu_wait_minutes: float = Field(10.0, gt=0, le=1440, description="wait for a GPU worker before the CPU fallback")
    max_parallel: int = Field(4, ge=1, le=64, description="series processed at the same time")
    output_folder: str = Field("inlineholo_results", description="folder in the upload for the NXem results")
    voltage_kV: float | None = Field(None, description="acceleration voltage, if not in the files")
    pixel_size_nm: float | None = Field(None, description="pixel size, if not in the files")
    defocus_start_nm: float | None = Field(None, description="defocus of the first image, if not in the files")
    defocus_step_nm: float | None = Field(None, description="defocus step, if not in the files")
    cs_mm: float | None = Field(None, description="spherical aberration C3 in mm, if not in the files")
    focal_spread_nm: float | None = Field(None, description="focal spread")
    convergence_mrad: float | None = Field(None, description="illumination convergence")
    stage_timeout_hours: float = Field(12.0, gt=0, le=168, description="maximum run time of one stage of one series")

    def overrides(self):
        keys = ("voltage_kV", "pixel_size_nm", "defocus_start_nm", "defocus_step_nm", "cs_mm", "focal_spread_nm",
                "convergence_mrad")
        return {k: getattr(self, k) for k in keys if getattr(self, k) is not None}


class SeriesTask(BaseModel):
    """One focal series of the batch (created by the planning activity)."""
    index: int
    name: str
    upload_id: str
    user_id: str
    sources: list[str]            # raw files of the upload that make up the series
    is_folder: bool = False
    work_dir: str                 # shared working folder of this series
    method: str = "frwr"
    n_iter: int = 50
    align: bool = True
    binning: int = 2
    overrides: dict = Field(default_factory=dict)
    output_folder: str = "inlineholo_results"
    stage_timeout_hours: float = 12.0


class PrepareResult(BaseModel):
    index: int
    name: str
    ok: bool
    message: str = ""
    n_images: int = 0
    shape: list[int] = Field(default_factory=list)
    R_align: float | None = None
    max_drift_px: float | None = None
    defocus_range_nm: list[float] = Field(default_factory=list)


class ApprovalInput(BaseModel):
    """Answer to the approval request: which series are reconstructed."""
    approve_all: bool = Field(True, description="reconstruct all series whose alignment succeeded")
    approved: list[int] = Field(default_factory=list, description="indices of the series to reconstruct "
                                                                  "(used if approve_all is false)")
    comment: str = Field("", description="optional comment (stored with the action)")


class ReconstructTask(BaseModel):
    task: SeriesTask
    queue: str                    # task queue of the reconstruction activity
    fallback_queue: str = ""      # used if nobody picks the task up on `queue` in time
    wait_minutes: float = 10.0


class StageResult(BaseModel):
    ok: bool = True
    message: str = ""
    R_final: float | None = None
    runtime_s: float = 0.0
    worker_queue: str = ""


class SeriesResult(BaseModel):
    index: int
    name: str
    status: Literal["done", "failed", "not approved", "alignment failed"]
    file: str = ""
    R_align: float | None = None
    R_final: float | None = None
    reconstructed_on: str = ""
    runtime_s: float = 0.0
    message: str = ""


class BatchResult(BaseModel):
    upload_id: str
    n_series: int
    n_done: int
    results: list[SeriesResult] = Field(default_factory=list)
    summary_file: str = ""


class ReconstructOutcome(BaseModel):
    result: StageResult
    file: str = ""


# task queues of the NOMAD workers (values of nomad.actions.TaskQueue; not imported to keep the workflow code
# independent of NOMAD)
CPU_QUEUE = "cpu-task-queue"
GPU_QUEUE = "gpu-task-queue"
