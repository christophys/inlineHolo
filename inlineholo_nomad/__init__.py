# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""NOMAD plugin of inlineHolo: a dashboard (the inline-holography app inside the NOMAD GUI) and an
action (reconstruction of a focal series from a NOMAD upload on a NOMAD worker; results go back
into the upload as NXem).

Entry points (pyproject.toml, group ``nomad.plugin``):

* ``inlineholo_nomad.dashboards:inlineholo_dashboard`` – served at
  ``{api_base_path}/dashboards/inlineholo/``
* ``inlineholo_nomad.actions:inlineholo_reconstruction`` – Temporal workflow on the CPU queue

Nothing heavy is imported here: NOMAD imports the entry points in every process.
"""
