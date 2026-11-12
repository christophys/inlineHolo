# Web app without Jupyter, and a Galaxy Interactive Tool

`webapp/solara_app.py` serves the app (the same ipywidgets GUI as `notebooks/02_inline_holography_app.ipynb`)
with [Solara](https://solara.dev) from its own web server — no Jupyter server, notebook or Voilà:

    pip install "solara==1.58.2" "solara-assets==1.58.2"
    solara run webapp/solara_app.py --host 0.0.0.0 --port 8765        # http://localhost:8765

`solara-assets` provides the browser-side files locally (offline / firewalled servers); point Solara to them with
`SOLARA_ASSETS_PROXY_CACHE_DIR=<python prefix>/share/solara/cdn` if they are not found. Each browser session gets
its own App. `INLINEHOLO_DATA` sets the data folder of tab 1.

`galaxy/Dockerfile` builds a container with the app; `galaxy/interactivetool_inlineholo.xml` is a draft Galaxy
Interactive Tool for it: the selected history datasets appear in the data folder of tab 1, the NXem files written
in tabs 1, 4b and 6 come back as a collection when the tool is stopped. Before submitting it (pull request to
galaxyproject/galaxy `tools/interactive/` or installation by the usegalaxy.eu admins): push the image to a registry,
replace `OWNER`, and test it on a local Galaxy (planemo / the Galaxy training material on Interactive Tools).
