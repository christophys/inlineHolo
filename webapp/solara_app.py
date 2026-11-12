# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""The inline-holography app as a stand-alone web application (no Jupyter / Voila).

    solara run webapp/solara_app.py --host 0.0.0.0 --port 8765

Solara serves the same ipywidgets GUI (inlineholo.app.App) from its own web server; every browser
session gets its own App instance.  Environment: INLINEHOLO_DATA (data folder shown in tab 1).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import solara  # noqa: E402

from inlineholo.app import App  # noqa: E402

DATA = os.environ.get("INLINEHOLO_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


@solara.component
def Page():
    app = solara.use_memo(lambda: App(data_dir=DATA), [])
    solara.Title("Inline electron holography")
    solara.display(app.ui)
