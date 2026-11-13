# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Input and output of the inline-holography action (pydantic: serializable for Temporal; the input
model is also the form the NOMAD GUI shows for the action)."""
from typing import Literal

from pydantic import BaseModel, Field

Method = Literal["frwr", "gpfrwr", "frwr_gf", "frwr_gd", "gpfrwr_lsq", "mal", "ml_poisson", "iwfr", "gs",
                 "tie", "tie_full", "tie_poly", "tie_fp", "mftie", "gftie", "pam", "schiske"]


class ReconstructFocalSeriesInput(BaseModel):
    """Reconstruct the exit wave (amplitude and phase) of a focal series in a NOMAD upload."""

    upload_id: str = Field(..., description="ID of the NOMAD upload that contains the focal series; the results "
                                            "are written into the same upload (it must not be published)")
    user_id: str = Field("", description="set by NOMAD to the user who starts the action")
    path: str = Field(..., description="file in the upload (NXem .nxs, DM3/DM4 or TIFF stack, one frame of an "
                                       "indexed .img series) or a folder of single images ordered by the defocus "
                                       "in their names (e.g. F-1000nm.dm3)")
    method: Method = Field("frwr", description="reconstruction method (see the inlineHolo README); FRWR is the "
                                               "full-resolution wave reconstruction")
    n_iter: int = Field(50, ge=1, le=5000, description="iterations of the iterative methods")
    align: bool = Field(False, description="FRWR alignment first (drift, rotation, magnification and distortions "
                                           "vs. defocus) for raw, unaligned series")
    binning: int = Field(2, ge=1, le=8, description="binning used for the FRWR alignment fit")
    output_folder: str = Field("inlineholo_results", description="folder in the upload for the NXem result")
    output_name: str = Field("", description="file name of the result (default: <input>_<method>.nxs)")
    voltage_kV: float | None = Field(None, description="acceleration voltage, if not in the file")
    pixel_size_nm: float | None = Field(None, description="pixel size in the object plane, if not in the file")
    defocus_start_nm: float | None = Field(None, description="defocus of the first image (linear series), if the "
                                                             "file has no defoci")
    defocus_step_nm: float | None = Field(None, description="defocus step of the series, if the file has no defoci")
    cs_mm: float | None = Field(None, description="spherical aberration C3 in mm, if not in the file")
    focal_spread_nm: float | None = Field(None, description="focal spread (temporal coherence)")
    convergence_mrad: float | None = Field(None, description="illumination convergence (spatial coherence)")
    timeout_hours: float = Field(12.0, gt=0, le=168, description="maximum run time")

    def overrides(self):
        keys = ("voltage_kV", "pixel_size_nm", "defocus_start_nm", "defocus_step_nm", "cs_mm", "focal_spread_nm",
                "convergence_mrad")
        return {k: getattr(self, k) for k in keys if getattr(self, k) is not None}


class ReconstructFocalSeriesResult(BaseModel):
    upload_id: str
    files: list[str] = Field(default_factory=list, description="paths of the written files in the upload")
    method: str = ""
    n_images: int = 0
    shape: list[int] = Field(default_factory=list)
    R_final: float | None = None
    runtime_s: float = 0.0
    message: str = ""
