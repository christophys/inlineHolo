# Container of the inline-holography web app for a Galaxy Interactive Tool (or any server).
#   docker build -t inlineholo-web -f galaxy/Dockerfile .          (from the code/ folder)
#   docker run -p 8765:8765 inlineholo-web                          -> http://localhost:8765
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg SOLARA_TELEMETRY_MIXPANEL_ENABLE=False
# only what the web app needs (requirements.txt also lists Jupyter, abTEM and torch for the notebooks)
RUN pip install numpy scipy h5py matplotlib scikit-image tifffile "ipywidgets>=8" pyyaml mrcfile rosettasciio \
                "solara==1.58.2" "solara-assets==1.58.2" "pynxtools>=0.16" "pynxtools-em>=0.4"
COPY inlineholo /opt/inlineholo/inlineholo
COPY webapp /opt/inlineholo/webapp
COPY data/*.nxs /opt/inlineholo/data/
# offline web assets of Solara (no CDN access needed in the browser or the container)
ENV SOLARA_ASSETS_PROXY_CACHE_DIR=/usr/local/share/solara/cdn INLINEHOLO_DATA=/opt/inlineholo/data
WORKDIR /work
EXPOSE 8765
CMD ["solara", "run", "/opt/inlineholo/webapp/solara_app.py", "--host", "0.0.0.0", "--port", "8765", "--production", "--no-open"]
