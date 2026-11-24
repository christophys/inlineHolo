# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Run a Solara app in its own process and proxy it into a FastAPI app that NOMAD mounts.

Solara keeps global state per process (the app shown at "/", its root path, the kernels of the
sessions).  Mounted directly in the NOMAD app process, only one Solara dashboard could exist per NOMAD
installation.  Here every dashboard starts its own Solara server (uvicorn on 127.0.0.1, free port) as
a child process, and the FastAPI app returned to NOMAD forwards HTTP requests and the websockets to it:

* the path below the NOMAD mount is forwarded unchanged, the mount path is sent as ``x-script-name``
  (Solara's root path, so that the page loads its assets through NOMAD);
* ``Host``, ``Cookie`` (the NOMAD login) and the other request headers are forwarded, plus
  ``x-forwarded-host``/``x-forwarded-proto``;
* the child process ends with the NOMAD process (atexit and, on Linux, the parent-death signal).

The heavy computations of the app therefore run in a separate process and do not slow down the NOMAD API.
This file is identical in the inlineHolo and fit4DSTEM plugins.
"""
import asyncio
import atexit
import os
import signal
import socket
import subprocess
import sys
import time

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade", "content-length", "host"}


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _die_with_parent():
    """preexec_fn: the child gets SIGTERM when the NOMAD process ends (Linux only)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)    # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001
        pass


def solara_env(page, extra=None):
    """Environment of the Solara child process (offline assets of solara-assets, no telemetry)."""
    env = dict(os.environ)
    env.update(SOLARA_APP=page, SOLARA_TELEMETRY_MIXPANEL_ENABLE="False", MPLBACKEND="Agg",
               SOLARA_THEME_SHOW_BANNER="False")
    if "SOLARA_ASSETS_PROXY_CACHE_DIR" not in env:
        for root in (sys.prefix, sys.base_prefix, "/usr/local", "/usr"):
            cdn = os.path.join(root, "share", "solara", "cdn")
            if os.path.isdir(cdn):
                env["SOLARA_ASSETS_PROXY_CACHE_DIR"] = cdn
                break
    env.update(extra or {})
    return env


class SolaraProcess:
    """The Solara server of one app as a child process."""

    def __init__(self, page, name="solara", env=None, port=None):
        self.page, self.name = page, name
        self.env = solara_env(page, env)
        self.port = port or _free_port()
        self.proc = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return self
        cmd = [sys.executable, "-m", "uvicorn", "solara.server.starlette:app", "--host", "127.0.0.1",
               "--port", str(self.port), "--log-level", "warning", "--ws-max-size", str(256 * 2 ** 20)]
        kw = {"preexec_fn": _die_with_parent} if sys.platform.startswith("linux") else {}
        self.proc = subprocess.Popen(cmd, env=self.env, **kw)
        atexit.register(self.stop)
        return self

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def ready(self, timeout=0.0):
        """True when the server answers (waits up to ``timeout`` seconds)."""
        import urllib.request

        t_end = time.time() + timeout
        while True:
            try:
                with urllib.request.urlopen(self.url + "/readyz", timeout=2) as r:
                    if r.status == 200:
                        return True
            except Exception:  # noqa: BLE001
                pass
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"{self.name}: the Solara server ended (exit code {self.proc.returncode})")
            if time.time() >= t_end:
                return False
            time.sleep(0.25)


def proxy_app(child: SolaraProcess, title="dashboard", extra_routes=None, start_timeout=120.0):
    """FastAPI app that forwards everything to ``child`` (HTTP and websockets)."""
    import httpx
    from fastapi import FastAPI, Request, WebSocket
    from fastapi.responses import PlainTextResponse, StreamingResponse

    app = FastAPI(title=title, docs_url=None, redoc_url=None, openapi_url=None)
    state = {"client": None}

    def client():
        if state["client"] is None:
            state["client"] = httpx.AsyncClient(base_url=child.url, timeout=httpx.Timeout(300.0, connect=10.0))
        return state["client"]

    async def wait_ready():
        if child.ready(0):
            return True
        t_end = time.time() + start_timeout
        while time.time() < t_end:
            await asyncio.sleep(0.5)
            if await asyncio.to_thread(child.ready, 0):
                return True
        return False

    def fwd_headers(scope_headers, root_path, scheme, host):
        h = [(k, v) for k, v in scope_headers if k.lower() not in HOP_BY_HOP and not k.lower().startswith("sec-websocket")]
        h += [("host", host), ("x-script-name", root_path.rstrip("/") or "/"),
              ("x-forwarded-proto", scheme), ("x-forwarded-host", host)]
        return h

    for path, fn in (extra_routes or {}).items():
        app.add_api_route(path, fn, include_in_schema=False)

    @app.websocket("/{path:path}")
    async def ws_proxy(ws: WebSocket, path: str):
        import websockets

        if not await wait_ready():
            await ws.close(code=1013)
            return
        root = ws.scope.get("root_path", "")
        host = ws.headers.get("host", "localhost")
        scheme = "https" if ws.url.scheme == "wss" else "http"
        url = f"ws://127.0.0.1:{child.port}/{path}" + (f"?{ws.url.query}" if ws.url.query else "")
        hdrs = fwd_headers([(k, v) for k, v in ws.headers.items()], root, scheme, host)
        protos = [p.strip() for p in ws.headers.get("sec-websocket-protocol", "").split(",") if p.strip()]
        try:
            kw = {"additional_headers": hdrs}
            try:
                upstream = await websockets.connect(url, subprotocols=protos or None, max_size=None,
                                                    open_timeout=30, **kw)
            except TypeError:                     # websockets < 13
                upstream = await websockets.connect(url, subprotocols=protos or None, max_size=None,
                                                    open_timeout=30, extra_headers=hdrs)
        except Exception:  # noqa: BLE001
            await ws.close(code=1011)
            return
        await ws.accept(subprotocol=upstream.subprotocol)

        async def down():                         # Solara -> browser
            async for msg in upstream:
                if isinstance(msg, bytes):
                    await ws.send_bytes(msg)
                else:
                    await ws.send_text(msg)

        async def up():                           # browser -> Solara
            while True:
                m = await ws.receive()
                if m["type"] == "websocket.disconnect":
                    break
                if m.get("bytes") is not None:
                    await upstream.send(m["bytes"])
                elif m.get("text") is not None:
                    await upstream.send(m["text"])

        tasks = [asyncio.create_task(down()), asyncio.create_task(up())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            await upstream.close()
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    async def http_proxy(request: Request, path: str):
        if not await wait_ready():
            return PlainTextResponse(f"{title} is starting, please reload in a moment", status_code=503)
        root = request.scope.get("root_path", "")
        host = request.headers.get("host", "localhost")
        hdrs = fwd_headers(request.headers.items(), root, request.url.scheme, host)
        req = client().build_request(request.method, "/" + path, params=request.url.query or None, headers=hdrs,
                                     content=await request.body())
        r = await client().send(req, stream=True)
        out_headers = {k: v for k, v in r.headers.items() if k.lower() not in HOP_BY_HOP}
        return StreamingResponse(r.aiter_raw(), status_code=r.status_code, headers=out_headers,
                                 background=_close(r))

    return app


def _close(r):
    from starlette.background import BackgroundTask

    return BackgroundTask(r.aclose)
