# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""inlineholo - focal-series inline electron holography: forward models, reconstruction
algorithms, Fourier-ring-correlation benchmarking, Thon-ring fitting and NXem I/O."""
__version__ = "0.2.0"

from .params import ImagingParameters  # noqa: E402
from .optics import wavelength_nm, interaction_constant, FourierGrid  # noqa: E402
from .forward import ForwardModel, normalize_stack  # noqa: E402
from .reconstruct import reconstruct, METHODS, list_methods, Reconstruction  # noqa: E402
from .frc import frc, compare_to_truth, resolution_from_frc, half_bit_threshold, binomial_split  # noqa: E402
from . import simulate, nxem, ctffit, align, offaxis, strain, frwr3, frwr3_io, backend, fokker_planck  # noqa: E402
from .frwr3 import run_frwr3, frwr3_from_cdf, load_config as load_frwr3_config  # noqa: E402

__all__ = ["ImagingParameters", "wavelength_nm", "interaction_constant", "FourierGrid",
           "ForwardModel", "normalize_stack", "reconstruct", "METHODS", "list_methods",
           "Reconstruction", "frc", "compare_to_truth", "resolution_from_frc",
           "half_bit_threshold", "binomial_split", "simulate", "nxem", "ctffit", "align",
           "offaxis", "strain", "frwr3", "frwr3_io", "backend", "run_frwr3", "frwr3_from_cdf",
           "load_frwr3_config", "fokker_planck"]
