# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Temporal workflows, activities and their models of the NOMAD actions.

This package must not import NOMAD at module level: Temporal re-imports workflow modules (and their
parent packages) in its sandbox, where NOMAD's imports are not allowed.  NOMAD is only used inside
activities (``backend.NomadBackend``).  The entry points live in :mod:`inlineholo_nomad.actions`.
"""
