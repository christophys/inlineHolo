# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""NOMAD dashboard: the inline-holography app (same GUI as the Voila / Solara app) inside NOMAD.

Served by NOMAD at ``{api_base_path}/dashboards/inlineholo/`` (e.g.
http://localhost:8000/nomad-oasis/dashboards/inlineholo/), embedded in the GUI or in a new tab.
The user's NOMAD login is passed on: the NOMAD import (tab 1) and export (tab 6) work with this
NOMAD without an API token, and a reconstruction can be sent to a NOMAD worker as an action.

Environment variables (of the NOMAD app process):

* ``INLINEHOLO_DASHBOARD_URL``  – serve the app elsewhere (e.g. the container of galaxy/Dockerfile) and
  only link it from NOMAD (``external_url``; no NOMAD login is passed on then)
* ``INLINEHOLO_DATA``           – folder with example data shown in tab 1 (read only)
* ``INLINEHOLO_NOMAD_API``      – API URL the app uses for NOMAD (default: this NOMAD, from its config)
"""
import os

from nomad.config.models.plugins import DashboardEntryPoint


class InlineHoloDashboardEntryPoint(DashboardEntryPoint):
    def load(self):
        from .server import create_app

        return create_app()


_external = os.environ.get("INLINEHOLO_DASHBOARD_URL") or None

inlineholo_dashboard = InlineHoloDashboardEntryPoint(
    name="Inline holography",
    description="Exit-wave reconstruction from TEM focal series: Thon-ring fit, FRWR alignment, FRWR/TIE/... "
                "reconstruction, DIH strain, NXem export to NOMAD.",
    id_url_safe="inlineholo",
    launch_modes=["embedded", "tab"],
    external_url=_external,
)
