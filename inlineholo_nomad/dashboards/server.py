# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""FastAPI app of the dashboard: the Solara server of the inline-holography app, mounted by NOMAD.

Solara serves the ipywidgets GUI (:class:`inlineholo.app.App`) over a websocket; NOMAD mounts this
app at ``{api_base_path}/dashboards/inlineholo``, so the browser talks to the same origin as the
NOMAD GUI and sends the NOMAD ``Authorization`` cookie (used by :mod:`.page`).

Note: Solara keeps the state of each browser session in the process that served it, so the NOMAD
app must run with one FastAPI worker process (``nomad admin run appworker --dev`` / ``poe start``
does) or with sticky sessions; otherwise serve the app separately (``INLINEHOLO_DASHBOARD_URL``).
"""
import os
import sys

PAGE = "inlineholo_nomad.dashboards.page:Page"


def _solara_environment():
    os.environ.setdefault("SOLARA_TELEMETRY_MIXPANEL_ENABLE", "False")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if "SOLARA_ASSETS_PROXY_CACHE_DIR" not in os.environ:
        # offline JavaScript assets from the solara-assets package (no CDN access needed)
        for root in (sys.prefix, sys.base_prefix, "/usr/local", "/usr"):
            cdn = os.path.join(root, "share", "solara", "cdn")
            if os.path.isdir(cdn):
                os.environ["SOLARA_ASSETS_PROXY_CACHE_DIR"] = cdn
                break
    os.environ["SOLARA_APP"] = PAGE


def create_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    _solara_environment()
    import solara.server.app as solara_app
    import solara.server.starlette as solara_starlette

    if solara_app.apps.get("__default__") is None or getattr(solara_app.apps["__default__"], "fullname", "") != PAGE:
        solara_app.apps["__default__"] = solara_app.AppScript(PAGE)
    # NOMAD does not run the lifespan of mounted apps: start Solara here
    solara_starlette.on_startup()

    app = FastAPI(title="inlineHolo dashboard", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        import inlineholo

        return JSONResponse({"status": "ok", "inlineholo": getattr(inlineholo, "__version__", "")})

    app.mount("/", solara_starlette.app)
    return app
