# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""NOMAD ELN entry, workflow (``workflow2``) and a preview of the workflow graph for the app (tab 6).

The app records every processing step of a session (:class:`WorkflowLog`).  From it, next to the NXem results,
three archive files are written (and uploaded to NOMAD together with the results):

* ``<name>_datasets.archive.yaml``  ``nomad.datamodel.metainfo.eln.ElnFileManager`` listing the focal-series
  file(s) -- core NOMAD, no plugin needed.
* ``<name>.archive.yaml``           the measurement ELN ``inlineholo_nomad.schema_packages.inlineholo.
  ELNInlineHolography`` (a ``Measurement`` in the *NOMAD measurements* category of the nomad-measurements plugin;
  needs the inlineHolo NOMAD plugin), or -- ``schema="core"`` -- the core ``nomad.datamodel.metainfo.eln.
  ELNMeasurement``.  Its ``steps`` are the processing steps with their parameters, its ``description`` explains
  the workflow.
* ``<name>.workflow.archive.yaml``  ``workflow2``: input (focal series) -> tasks (the steps, in order) -> outputs
  (NXem results, measurement ELN), linked by references to sections of the entries above
  (``../upload/archive/mainfile/<file>#/<path>``), see https://docs.nomad-lab.eu/howto/manage/gui/workflows.html.

All paths are relative to the root of the NOMAD upload; ``prefix`` is the folder of the upload the files go to.
:func:`workflow_svg` draws the graph like NOMAD's workflow card (inputs left, tasks, outputs right).
"""
from __future__ import annotations

import datetime as _dt
import html
import json
import os

PLUGIN_SECTION = "inlineholo_nomad.schema_packages.inlineholo.ELNInlineHolography"
CORE_SECTION = "nomad.datamodel.metainfo.eln.ELNMeasurement"
TASK = "nomad.datamodel.metainfo.workflow.Task"

# key -> (name, description); the order is the order of the app's tabs
STEP_INFO = {
    "load": ("Load focal series", "Tab 1: focal series read from {source} ({n_images} images of {shape} px, "
                                  "pixel {pixel_size_nm} nm, C1 = {defocus_min_nm} ... {defocus_max_nm} nm)."),
    "prepare": ("Bin / crop", "Tab 1: binning {binning}, central crop {crop_px} px."),
    "parameters": ("Imaging parameters", "Tab 2: imaging parameters and per-image defocus of the series."),
    "thon": ("Thon-ring fit", "Tab 3: fit of defocus, two-fold astigmatism (and C3) to the power spectra."),
    "thon_apply": ("Apply Thon-ring fit", "Tab 3: fitted values written to the imaging parameters."),
    "drift": ("Drift correction", "Tab 4: cross-correlation registration of the series."),
    "frwr_align": ("FRWR alignment", "Tab 4b: FRWR alignment (drift and defocus-dependent distortions fitted "
                                     "together with the exit wave) and initial reconstruction."),
    "reconstruct": ("Exit-wave reconstruction", "Tab 5: reconstruction of amplitude and phase."),
    "strain": ("Strain (GPA)", "Tab 5: strain map from the reconstructed phase."),
    "halfset": ("Half-set FRC", "Tab 5: resolution estimate by half-set Fourier ring correlation."),
    "dih": ("DIH strain analysis", "Tab 7: strain, shear and rotation from dark-field inline holograms."),
    "export": ("Export NXem", "Tab 6: results written as NeXus/NXem; ELN and workflow archives for NOMAD."),
}


def _now():
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _plain(o):
    """numpy scalars / arrays (recursively) -> plain Python (YAML / JSON)."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if np is not None:
        if isinstance(o, np.ndarray):
            return _plain(o.tolist()) if o.size <= 64 else f"array {o.shape}"
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, complex):
            return str(o)
    if isinstance(o, float):
        return round(o, 6)
    if o is None or isinstance(o, (str, int, bool)):
        return o
    return str(o)


class WorkflowLog(list):
    """Steps of a session: dicts with key, name, time, params, summary."""

    def add(self, key, params=None, summary="", name=None, replace=False):
        nm = name or STEP_INFO.get(key, (key, ""))[0]
        st = {"key": key, "name": nm, "time": _now(), "params": _plain(params or {}), "summary": str(summary or "")}
        if replace:
            for i, s in enumerate(self):
                if s["key"] == key:
                    self[i] = st
                    return st
        self.append(st)
        return st

    def describe(self, step):
        info = STEP_INFO.get(step["key"], (step["name"], ""))[1]
        try:
            info = info.format(**{k: v for k, v in step["params"].items()})
        except (KeyError, IndexError, ValueError):
            info = info.split(":")[0] + "."
        return info


def workflow_description(log, title="Inline electron holography", results=None):
    """Markdown description of the applied workflow (ELN, NXem experiment description)."""
    from . import __version__
    log = log if isinstance(log, WorkflowLog) else WorkflowLog(log)
    lines = [f"## {title} (inlineholo {__version__})", "", "**Workflow**", ""]
    for i, st in enumerate(log, 1):
        par = {k: v for k, v in st["params"].items() if k not in ("source",)}
        lines.append(f"{i}. **{st['name']}** – {log.describe(st)}"
                     + (f" {st['summary']}" if st["summary"] else "")
                     + (f" Parameters: `{json.dumps(par, default=str, ensure_ascii=False)}`" if par else ""))
    if results:
        lines += ["", "**Results:** " + str(results)]
    return "\n".join(lines)


def build_archives(name, log, input_files, results_file=None, prefix="", schema="plugin", settings=None,
                   metadata=None, results_summary=None, reconstructions=None):
    """The three archives as dicts: {filename: archive}.  ``input_files`` and ``results_file`` are paths relative to
    the upload root; the archives themselves are placed in ``prefix``."""
    log = log if isinstance(log, WorkflowLog) else WorkflowLog(log)
    pre = (prefix.strip("/") + "/") if prefix.strip("/") else ""
    meta = metadata or {}
    now = _now()
    ds_name, me_name, wf_name = (f"{name}_datasets.archive.yaml", f"{name}.archive.yaml",
                                 f"{name}.workflow.archive.yaml")
    # ------------------------------------------------------------ datasets
    files = [{"file": f, "description": "focal series" + (f" (part {i + 1})" if len(input_files) > 1 else "")}
             for i, f in enumerate(input_files)]
    ds = {"data": {"m_def": "nomad.datamodel.metainfo.eln.ElnFileManager", "name": f"{name}: focal series",
                   "description": "Input data of the inline-holography reconstruction (inlineholo).",
                   "Files": files}}
    # ------------------------------------------------------------ measurement
    desc = workflow_description(log, results=results_summary)
    def _comment(st):
        c = log.describe(st) + (" " + st["summary"] if st["summary"] else "")
        if st["params"]:
            c += "\n\nParameters: " + json.dumps(st["params"], default=str, ensure_ascii=False)
        return c

    steps = [{"name": f"{i + 1}. {st['name']}", "start_time": st["time"], "comment": _comment(st)}
             for i, st in enumerate(log)]
    sample = meta.get("sample") or {}
    instr = meta.get("instrument") or {}
    data = {"m_def": PLUGIN_SECTION if schema == "plugin" else CORE_SECTION,
            "name": f"{name}: inline electron holography", "datetime": now, "description": desc,
            "method": "inline electron holography (focal-series exit-wave reconstruction)", "steps": steps}
    smp = {"name": str(sample.get("name") or "specimen")}
    if sample.get("identifier"):
        smp["lab_id"] = str(sample["identifier"])
    data["samples"] = [smp]
    ins = {"name": str(instr.get("name") or "transmission electron microscope")}
    if instr.get("serial_number"):
        ins["lab_id"] = str(instr["serial_number"])
    data["instruments"] = [ins]
    if schema == "plugin":
        data["settings"] = {k: v for k, v in _plain(settings or {}).items() if v is not None}
        data["focal_series"] = [{"name": os.path.basename(f), "file": f} for f in input_files]
        data["reconstructions"] = _plain(reconstructions or [])
        if results_file:
            data["nxem_file"] = results_file
        if results_summary:
            data["results_summary"] = str(results_summary)
        data["analysis_software"] = "inlineholo"
    else:
        data["tags"] = ["inline holography", "focal series", "exit-wave reconstruction", "inlineholo"]
    me = {"data": data}
    # ------------------------------------------------------------ workflow
    up = "../upload/archive/mainfile/"
    inputs = [{"name": "focal series" + (f" {os.path.basename(f)}" if len(input_files) > 1 else ""),
               "section": f"{up}{pre}{ds_name}#/data/Files/{i}"} for i, f in enumerate(input_files)]
    outputs = []
    if results_file:
        outputs.append({"name": "NXem results", "section": f"{up}{results_file}#/data"})
    outputs.append({"name": "measurement ELN", "section": f"{up}{pre}{me_name}#/data"})

    def stp(i):
        return f"{up}{pre}{me_name}#/data/steps/{i}"

    tasks = []
    prev = [dict(x) for x in inputs]          # links the next step reads
    last_pre_rec = prev
    rec_outs = []
    for i, st in enumerate(log):
        out = [{"name": st["name"], "section": stp(i)}]
        if st["key"] == "reconstruct":
            ins = [dict(x) for x in last_pre_rec]
            rec_outs += out
        elif st["key"] in ("export",):
            ins = [dict(x) for x in (rec_outs or prev)]
            out = [dict(x) for x in outputs] + out
        elif st["key"] in ("strain", "halfset", "dih") and rec_outs:
            ins = [dict(x) for x in rec_outs[-1:]]
        else:
            ins = [dict(x) for x in prev]
        tasks.append({"m_def": TASK, "name": st["name"] + (f" ({st['params'].get('method')})"
                                                           if st["key"] == "reconstruct" else ""),
                      "inputs": ins, "outputs": out})
        if st["key"] != "reconstruct":
            prev = out[-1:]
            if not rec_outs:
                last_pre_rec = prev
        else:
            prev = out
    if not any(st["key"] == "export" for st in log) and tasks:
        tasks[-1]["outputs"] = tasks[-1]["outputs"] + [dict(x) for x in outputs]
    wf = {"workflow2": {"name": f"{name}: inline-holography reconstruction", "inputs": inputs, "outputs": outputs,
                        "tasks": tasks}}
    return {ds_name: ds, me_name: me, wf_name: wf}


def write_eln(folder, name, log, input_files, results_file=None, prefix="", **kw):
    """Write the archives (see :func:`build_archives`) and ``<name>_workflow.svg`` into ``folder``."""
    import yaml
    os.makedirs(folder, exist_ok=True)
    paths = {}
    archives = build_archives(name, log, input_files, results_file, prefix, **kw)
    for fn, arch in archives.items():
        p = os.path.join(folder, fn)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(_plain(arch), f, sort_keys=False, allow_unicode=True, width=120)
        paths[fn] = p
    wf = archives[f"{name}.workflow.archive.yaml"]["workflow2"]
    p = os.path.join(folder, f"{name}_workflow.svg")
    with open(p, "w", encoding="utf-8") as f:
        f.write(workflow_svg(wf))
    paths["svg"] = p
    return paths


# ====================================================================== preview ==


def _label(link):
    return link.get("name") or link.get("section", "").split("#")[-1]


def _short(t, n):
    return t if len(t) <= n else t[:n - 1] + "…"


def workflow_svg(wf, width=1000):
    """SVG of the workflow graph like NOMAD's workflow card: inputs (left) -> tasks (middle, in order) -> outputs
    (right); edges where a task input refers to the same section as a workflow input or an earlier task output."""
    inputs, outputs, tasks = wf.get("inputs", []), wf.get("outputs", []), wf.get("tasks", [])
    col_x = [20, 320, width - 250]
    box_w = [230, 300, 230]
    row_h = 54
    n_rows = max(len(inputs), len(tasks), len(outputs), 1)
    H = 70 + n_rows * row_h + 10
    pos = {}

    def ypos(i, n):
        return 70 + (n_rows - n) * row_h / 2 + i * row_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{H}" viewBox="0 0 {width} {H}" '
             f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
             '<defs><marker id="ihArr" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
             '<path d="M0,0 L10,4 L0,8 z" fill="#5a6b7b"/></marker></defs>',
             f'<text x="{col_x[0]}" y="28" font-size="14" font-weight="bold" fill="#2a3a4a">'
             f'{html.escape(wf.get("name", "workflow"))}</text>',
             f'<text x="{col_x[0]}" y="54" fill="#5a6b7b">inputs</text>',
             f'<text x="{col_x[1]}" y="54" fill="#5a6b7b">tasks</text>',
             f'<text x="{col_x[2]}" y="54" fill="#5a6b7b">outputs</text>']
    for i, lk in enumerate(inputs):
        y = ypos(i, len(inputs))
        pos[("in", i)] = (col_x[0], y)
        parts.append(f'<g><title>{html.escape(lk.get("section", ""))}</title><rect x="{col_x[0]}" y="{y}" rx="18" '
                     f'width="{box_w[0]}" height="38" fill="#e3f0fb" stroke="#2f80c3"/><text x="{col_x[0] + 14}" '
                     f'y="{y + 24}" fill="#123">{html.escape(_short(_label(lk), 32))}</text></g>')
    for i, t in enumerate(tasks):
        y = ypos(i, len(tasks))
        pos[("task", i)] = (col_x[1], y)
        parts.append(f'<g><rect x="{col_x[1]}" y="{y}" rx="4" width="{box_w[1]}" height="38" fill="#fff1df" '
                     f'stroke="#e08a1e"/><text x="{col_x[1] + 12}" y="{y + 24}" fill="#321">'
                     f'{html.escape(_short(t.get("name", "task"), 40))}</text></g>')
    for i, lk in enumerate(outputs):
        y = ypos(i, len(outputs))
        pos[("out", i)] = (col_x[2], y)
        parts.append(f'<g><title>{html.escape(lk.get("section", ""))}</title><rect x="{col_x[2]}" y="{y}" rx="18" '
                     f'width="{box_w[2]}" height="38" fill="#e5f6e8" stroke="#2e9e4f"/><text x="{col_x[2] + 14}" '
                     f'y="{y + 24}" fill="#132">{html.escape(_short(_label(lk), 32))}</text></g>')
    edges = []
    task_out = {}
    for i, t in enumerate(tasks):
        for lk in t.get("outputs", []):
            task_out.setdefault(lk.get("section"), []).append(i)
    for i, t in enumerate(tasks):
        for lk in t.get("inputs", []):
            s = lk.get("section")
            for j, w in enumerate(inputs):
                if w.get("section") == s:
                    edges.append((("in", j), ("task", i)))
            for k in task_out.get(s, []):
                if k != i:
                    edges.append((("task", k), ("task", i)))
    for j, w in enumerate(outputs):
        for k in task_out.get(w.get("section"), []):
            edges.append((("task", k), ("out", j)))
    for a, b in dict.fromkeys(edges):
        xa, ya = pos[a]
        xb, yb = pos[b]
        wa = box_w[{"in": 0, "task": 1, "out": 2}[a[0]]]
        if a[0] == "task" and b[0] == "task":
            x = xa + wa
            bend = 24 + 10 * min(abs(b[1] - a[1]), 6)
            parts.append(f'<path d="M{x},{ya + 19} C{x + bend},{ya + 19} {x + bend},{yb + 19} {x},{yb + 19}" '
                         f'fill="none" stroke="#5a6b7b" marker-end="url(#ihArr)"/>')
        else:
            xm = (xa + wa + xb) / 2
            parts.append(f'<path d="M{xa + wa},{ya + 19} C{xm},{ya + 19} {xm},{yb + 19} {xb},{yb + 19}" '
                         f'fill="none" stroke="#5a6b7b" marker-end="url(#ihArr)"/>')
    parts.append("</svg>")
    return "\n".join(parts)
