# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""FastAPI app of the dashboard that NOMAD mounts at ``{api_base_path}/dashboards/inlineholo``.

The Solara server of the inline-holography app (:mod:`.page`) runs as a child process of the NOMAD app
(:mod:`.solara_proxy`); this app forwards the pages, assets and websockets to it.  The browser talks
to the same origin as the NOMAD GUI and sends the NOMAD ``Authorization`` cookie, which the page uses.
Several Solara dashboards (e.g. inlineHolo and fit4DSTEM) can so be installed in the same NOMAD.

Note: the sessions live in the child process of the NOMAD app process that started it, so the NOMAD
app must run with one FastAPI worker process (``uv run poe start`` does) or with sticky sessions;
otherwise serve the app separately (``INLINEHOLO_DASHBOARD_URL``).
"""
PAGE = "inlineholo_nomad.dashboards.page:Page"


def create_app():
    from fastapi.responses import JSONResponse

    from .solara_proxy import SolaraProcess, proxy_app

    child = SolaraProcess(PAGE, name="inlineholo").start()

    def healthz():
        import inlineholo

        return JSONResponse({"status": "ok" if child.ready(0) else "starting",
                             "inlineholo": getattr(inlineholo, "__version__", "")})

    return proxy_app(child, title="inlineHolo dashboard", extra_routes={"/healthz": healthz})
