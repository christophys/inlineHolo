# inlineHolo as a NOMAD plugin (dashboard + action)

The repository is also a NOMAD plugin (package `inlineholo_nomad`, entry points in `pyproject.toml`):

| entry point | type | what it does |
|---|---|---|
| `inlineholo_nomad.dashboards:inlineholo_dashboard` | dashboard | the inline-holography app (the same GUI as the Voilà / Solara app) inside NOMAD, at `{api_base_path}/dashboards/inlineholo/` |
| `inlineholo_nomad.actions:inlineholo_reconstruction` | action | reconstruction of a focal series of a NOMAD upload on a NOMAD worker; the NXem result is added to the upload |

Both need the new NOMAD GUI and NOMAD ≥ 2.0 (dashboard and action entry points).

## Installation in nomad-distro-dev

```bash
cd nomad-distro-dev
git submodule add https://github.com/christophys/inlineHolo.git packages/inlineHolo
uv add "./packages/inlineHolo[nomad]"      # or: uv add packages/inlineHolo  plus  uv pip install solara solara-assets
uv sync
docker compose up -d                       # infrastructure (MongoDB, Elasticsearch, Temporal, ...)
uv run poe setup                           # first time only
uv run poe start                           # NOMAD API + new GUI (one FastAPI worker)
uv run poe cpuworker                       # in a second terminal: CPU worker that runs the action
```

* **Dashboard:** new GUI (http://localhost:8000/nomad-oasis/gui/v2/) → *Dashboards* → *Inline holography*
  (embedded or in a new tab). Directly: http://localhost:8000/nomad-oasis/dashboards/inlineholo/
* **Action:** new GUI → *Actions* → *Inline holography: reconstruct a focal series* (form with upload ID, file or
  folder, method, iterations, optional FRWR alignment and imaging parameters), or from the dashboard: tab 5 →
  *run on a NOMAD worker*, or via the API:

  ```bash
  curl -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
       http://localhost:8000/nomad-oasis/api/v1/actions/inlineholo_nomad.actions:inlineholo_reconstruction/start \
       -d '{"data": {"upload_id": "<upload id>", "path": "raw/series.dm3", "method": "frwr", "n_iter": 50}}'
  ```

  The result (`<input>_<method>.nxs` in the folder `inlineholo_results` of the upload) contains the focal
  series, the imaging parameters and the exit wave (amplitude and phase) and is processed by NOMAD as an NXem
  entry (with pynxtools). Only authors and co-authors of an unpublished upload can run the action on it.

![dashboard in NOMAD](screenshots/nomad_dashboard_action.png)

*The dashboard served by the NOMAD app (`/nomad-oasis/dashboards/inlineholo/`): signed in with the NOMAD login,
focal series imported from an upload, reconstruction started on a NOMAD worker (tab 5).*

## How the dashboard works

NOMAD calls `load()` of the dashboard entry point once at start-up and mounts the returned FastAPI app. That
app is the **Solara** server of the app (`inlineholo_nomad/dashboards/server.py`); Solara renders the
ipywidgets GUI of `inlineholo.app.App` in the browser and keeps one App per browser session in the NOMAD app
process (`inlineholo_nomad/dashboards/page.py`):

* **NOMAD login:** the browser sends the `Authorization` cookie of the NOMAD GUI to the dashboard (same origin).
  The page uses it for the NOMAD import (tab 1) and export (tab 6): *this NOMAD* is preselected, the uploads of
  the user are listed, no API token is needed. The GUI token expires after some time; then reload the
  dashboard (or paste an app token in tab 1).
* **Server mode:** the app cannot read or write the server's files (no local path / file browser in tabs 1
  and 7); each session has its own working folder, deleted when the session ends; data come from NOMAD
  uploads or browser uploads.
* **Computations** of the dashboard run in threads of the NOMAD app process (as in the stand-alone app, with
  progress and stop buttons). Long reconstructions are better sent to a worker with the action.

### Limits and options

* Solara keeps the session state in the process that served the page: the NOMAD app must run with **one
  FastAPI worker** (as `uv run poe start` / `nomad admin run appworker --dev` does) or behind a proxy with
  sticky sessions.
* For production Oases with several app workers, run the app separately (the container of
  `galaxy/Dockerfile`) and set `INLINEHOLO_DASHBOARD_URL` in the environment of the NOMAD app: the dashboard
  then only links/iframes that URL (`external_url`). The NOMAD login is not passed on in that case (use an
  API token in tab 1).
* Embedding in a GUI served from another origin (e.g. the GUI dev server on port 3000) needs
  `dashboards.frame_ancestors` in `nomad.yaml`:

  ```yaml
  dashboards:
    frame_ancestors: ["'self'", "http://localhost:3000"]
  ```

* Environment variables: `INLINEHOLO_DATA` (read-only example data shown in tab 1), `INLINEHOLO_NOMAD_API`
  (API URL the dashboard uses to reach NOMAD, default: derived from the request / the NOMAD config).
* The action runs on the CPU task queue. The FRWR methods can use a GPU (CuPy/PyTorch back ends); for a GPU
  worker set `task_queue: gpu-task-queue` for the entry point in `nomad.yaml`
  (`plugins.entry_points.options."inlineholo_nomad.actions:inlineholo_reconstruction".task_queue`).

## Tests

`tests/test_nomad_plugin.py` covers the headless pipeline, the action activity (with a stand-in for the upload
files), the dashboard page against a mock NOMAD API and the Solara server mounted in FastAPI; the tests that
need `temporalio`, `fastapi` or `solara` are skipped when these are missing.
