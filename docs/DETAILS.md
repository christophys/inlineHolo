# inlineHolo — detailed documentation

```
(repository root)
  inlineholo/               Python module
    optics.py               wavelength, interaction constant, aberration function (Krivanek/NXem notation), envelopes
    params.py               ImagingParameters
    forward.py              forward models: coherent | envelope | blur (FRWR) | incoherent (full partial coherence) + gradients
    reconstruct.py          TIE, TIE_full, TIE_poly, MFTIE (Eljarrat 2018), GF-TIE (Parvizi 2016), PAM, Schiske,
                            GS/Misell, IWFR, FRWR / FRWR+GF / GP-FRWR (= FRWR3 port, see below), flux-preserving
                            LSQ gradient variants (frwr_gd, gpfrwr_gd, gpfrwr_lsq), MAL, Poisson-ML, hybrid, generic gd
    frwr3.py                FRWR3 (C. T. Koch) ported loop by loop: in memory, NumPy/CuPy/PyTorch (GPU),
                            YAML/.conf configuration with the original frwr3 key names
    frwr3_io.py             frwr3 netCDF input, .img output and .conf control files
    backend.py              array back ends (numpy | cupy | torch, auto-selects a GPU)
    fokker_planck.py        Fokker-Planck TIE (Paganin & Morgan): tie_fp, diffusion-field retrieval
    offaxis.py              off-axis hologram simulation and sideband reconstruction
    strain.py               dark-field inline holography (DIH): strain from the geometric phase (1 or 2 reflections)
    dih.py                  many-beam DIH (port of the DM scripts createDFstack / DFHoloAllBeamsGPA): N reflections,
                            unit cell -> 2D g vectors, weighted least squares strain/rotation or displacement
    app_dih.py              tab 7 of the app (DIH)
    references.py           literature references (with DOI) of the reconstruction methods (tab 5 of the app)
    nomad.py                NOMAD API v1 client: list uploads/folders, download raw files, upload results
    app_nomad.py            NOMAD import (tab 1) and export (tab 6) of the app
webapp/solara_app.py        the app as a stand-alone web application (Solara, no Jupyter)
galaxy/                     Dockerfile and draft Galaxy Interactive Tool for the web app
    mcnn.py                 multi-resolution CNN (Wang 2020; PyTorch, optional): training on simulated series + inference
    frc.py                  Fourier ring correlation, thresholds, comparison with ground truth, binomial split
    simulate.py             HR, MR and DIH (InGaN/GaN MQW) test objects, random objects for training, simulation, dose
    ctffit.py               Thon-ring fitting (defocus, 2-fold astigmatism, C3; per image and series-consistent)
    align.py                drift correction
    nxem.py                 NXem (NeXus) reader/writer for focal series, parameters, ground truth, reconstructions,
                            derived maps and curves; user metadata (METADATA_FIELDS); validation with pynxtools
    io_formats.py           loaders for NXem, TIFF, DM3/DM4, EMD, SER, MRC, NPY (+ metadata extraction,
                            FRWR DigitalMicrograph tags -> imaging parameters)
    dmreader.py             built-in DM3/DM4 reader (memory-mapped, no RosettaSciIO needed)
    app.py                  ipywidgets application (used by the Voila notebook)
  tests/                    pytest suite (gradient checks, NXem round trip, Thon fit, FRWR3 regression vs C++, ...)
  examples/                 frwr3_lowres2.conf/.yaml (your control file), frwr3_all_keys.yaml (all keys + docs)
  frwr3_cpp_patches/        patch + build notes for the C++ FRWR3 on Linux/FFTW 3.3
  tools_upgrade_nxem.py     rewrites NXem files of inlineholo 0.2 in the current, validated layout
  notebooks/
  01_method_comparison.ipynb      simulation + quantitative comparison (FRC vs. ground truth, dose series)
  02_inline_holography_app.ipynb  Voila app for experimental data
  00_download_github_repo.ipynb   download a public GitHub repository (git clone or zip)
  make_notebooks/                 scripts that generate notebooks 01 and 02
  data/                     simulated focal series (NXem, incl. ground truth + reconstructions)
  results/                  CSV tables of the benchmark
```

## Quick start

```bash
pip install numpy scipy h5py matplotlib scikit-image pandas tifffile ipywidgets voila jupyter pyyaml
# NXem validation: pip install pynxtools pynxtools-em
# optional: rosettasciio (DM3/DM4/EMD/SER), mrcfile, torch (MCNN, GPU), cupy (GPU)

cd code && python -m pytest -q tests          # run the tests (in this folder)
cd notebooks
jupyter lab 01_method_comparison.ipynb         # benchmark (~60-80 min on 2 cores incl. MCNN training; FAST=True ~10 min)
voila 02_inline_holography_app.ipynb           # web app
```

```python
import inlineholo as ih
stack, params, gt, recs = ih.nxem.read_focal_series("data/hr_focal_series_sim.nxs")
rec = ih.reconstruct(stack, params, "frwr", n_iter=100)
ih.nxem.write_reconstruction("data/hr_focal_series_sim.nxs", rec, params, label="my FRWR")
print(ih.compare_to_truth(rec.wave, gt, params.pixel_size_nm)["frc_mid"])
```

## FRWR3 in Python (GPU, no intermediate files)

```python
import inlineholo as ih
# 1) directly from an frwr3 netCDF file and control file (YAML or the original .conf)
res = ih.frwr3_from_cdf("series.cdf", "code/examples/frwr3_lowres2.yaml", backend="auto")
res.wave, res.defoci_nm, res.alpha_rad, res.frame_offsets, res.distortions, res.rvalues

# 2) from full frames in memory + ImagingParameters (ROI = top, left, bottom, right)
res = ih.run_frwr3(frames, params, "my_frwr3.yaml", roi=(32, 32, 288, 288))

# 3) through the common interface (pre-aligned data), any frwr3 key via config=
rec = ih.reconstruct(stack, params, "frwr", n_iter=200, config={"fit defocus": 2, "apply GF": 1})
```

* keys and defaults: `code/examples/frwr3_all_keys.yaml` (generated by `ih.frwr3.default_yaml()`);
  `.conf` -> YAML: `ih.frwr3.conf_to_yaml("x.conf", "x.yaml")`
* extensions (marked `[inlineholo]`): `backend`, `precision`, `aberrations: full`, `focal spread envelope`,
  `subpixel shifts`, objective aperture, `roi`; `fit reference points` is implemented as a smooth
  displacement field (disabled in the C++ code); `apply MTF: flag wGauss wExp ratio` is honoured
  (the 2019 C++ parser only reads `MTF: flag aExp wExp aGauss wGauss`)
* validation: identical R-value traces (rel. deviation < 1e-4) for all deterministic features;
  `tests/test_frwr3.py` contains a regression test against a C++ reference run.

## Conventions

* lengths in nm, angles in rad, voltage in V; aberrations in Krivanek notation `c_n_m_a/b`
  (identical to the NXem `NXcorrector_cs` tableau), `c_1_0` = defocus, positive = overfocus
* image wave `psi_n = F^-1[ Psi * exp(-i chi_n) ]` (pure defocus = Fresnel propagation by z = C1)
* focal spread and convergence are **standard deviations** of Gaussian distributions
  (convergence: rms tilt angle per axis)

## NXem files and validation

All files written by `inlineholo.nxem` comply with the NXem application definition shipped with
`pynxtools` / `pynxtools-em` and can be checked with `pynx validate file.nxs` (or
`ih.nxem.validate(path)`). NXem requires some metadata that only the user knows; pass them as

```python
meta = {"sample": {"atom_types": "Si, N", "preparation_date": "2026-10-30", "name": "Si3N4 lamella"},
        "instrument": {"vendor": "JEOL", "model": "JEM-ARM200F", "emitter_type": "cold_field_emission"},
        "experiment": {"start_time": "2026-11-03T14:20:00+01:00"},
        "user": {"name": "...", "orcid": "..."}}
ih.nxem.write_focal_series("s.nxs", stack, params, metadata=meta)
ih.nxem.write_reconstruction("s.nxs", rec, params, label="frwr")      # exit wave + amplitude/phase + extras
ok, problems, undocumented = ih.nxem.validate("s.nxs")
```

`ih.nxem.METADATA_FIELDS` lists all fields (required ones are flagged); in the Voila app they are
editable in tab 6 and can be saved / loaded as YAML. `code/tools_upgrade_nxem.py` rewrites files
written by inlineholo 0.2 in the current layout.

## DM3 focal series, conversion to NXem and FRWR3 alignment

DM3/DM4 files are read by a built-in reader. Tags written by the FRWR DigitalMicrograph plugins
(defocus offset/step, astigmatism, alpha, aperture, Cs, voltage, microscope, specimen, operator,
acquisition time) are converted to imaging parameters and NXem metadata suggestions. In the app,
load large files by path, with the file browser of tab 1 or from the data folder; a non-NXem
file can then be converted to a validated NXem file after filling in the required metadata.
If an uploaded file also exists on the computer running the app (same name and size in the
browser folder, the data folder, the working directory, Downloads, Desktop, Documents or home),
it is read from there, as if loaded by path.

Browser uploads go through the websocket of the Jupyter/Voilà server, whose message size is
limited (10 MiB by default). The files `notebooks/voila.json` and
`notebooks/jupyter_server_config.json` raise the limit to 1 GiB when `voila` / `jupyter lab` is
started in the `notebooks` folder (or copy their content into `~/.jupyter/`).

### Defocus series, Thon-ring fits, stopping long computations

* Tab 2 fills the defocus column with the FRWR series (Eq. 1 of the FRWR manual)
  Δf_n = Δf_offset + step · sign(n − n0) · |n − n0|^m (m = 1 linear; m = 2, 3 for small steps
  near focus and large steps at large defocus). FRWR DM tags prefill these fields.
* Tab 3: the q_min / q_max sliders are in 1/nm (same scale as the FFT axes); the FFT view zooms
  so that the q_max ring fills 90 % of it, and the rings follow the sliders live. The "series
  fit" fits either C1_n = offset + scale·C1_n(table) or Eq. 1 (offset, step, optionally m).
* Thon-ring fits, the FRWR alignment (tab 4b) and the reconstruction (tab 5) run in the
  background with progress bar, status and estimated remaining time, and have a *stop* button.
  A stopped Thon fit keeps the images fitted so far, a stopped alignment leaves the stack
  unchanged, a stopped reconstruction keeps the current wave as its result.

`ih.align.frwr3_align(stack, params, binning=2)` (app: tab 4b) fits drift and the
defocus-dependent distortions (rotation, magnification, elliptical magnification, spiral,
pincushion) with FRWR3 on a binned copy, applies them to the full-resolution frames and crops
to the common area. On the raw CeO2 series `stackExpRaw.dm3` (Koch, Micron 63, 2014) this
reproduces the aligned series `stackExp.dm3` (rotation within 1.3 mrad, magnification within
2.7e-3, correlation 0.97-0.99 per frame).

**Tab 4 "Align/Pre-reconstruct", 4b "FRWR Alignment and Initial Reconstruction".** The exit wave
of the binned FRWR3 alignment run is Fourier-interpolated to the full resolution and placed in
the geometry of the aligned, cropped stack (`ih.align.initial_wave`; the `info` returned by
`frwr3_align` contains `wave`, `wave_box` and `wave_start`). The app shows its amplitude and phase
and writes the aligned series, this initial reconstruction (exit wave, amplitude, phase,
R-values) and all alignment results (distortions at full resolution and as fitted, drift, crop,
reference frame, refined defoci, alpha, astigmatism, FRWR3 settings) with the NXem metadata to a
`.nxs` file; it is also included in the session export of tab 6. The progress estimate weights
every FRWR3 loop with the number of images it projects (`FRWR3.planned_work()` /
`FRWR3.work_done`), since the time per loop is proportional to that number.

**Tab 5 "Reconstruct/Refine".** The start wave can be the FRWR initial reconstruction of tab 4b
or the last reconstruction. With *start with all frames* (FRWR3 key `start with all planes: 1`,
an inlineholo extension) FRWR skips the 2-/3-plane initialization and the step-wise inclusion of
further planes and uses all frames from the first loop; each frame is aligned to the image
simulated from the start wave.


## Dark-field inline holography with several reflections (tab 7)

`inlineholo.dih` ports the DigitalMicrograph scripts of the FRWR tools (menu *DIH-GPA*:
`createDFstack.s`, `DFHoloAllBeamsGPA.s`, `DFHoloAllBeamsGPA_computeDisplacement.s`). For every
reflection g_j a dark-field focal series is reconstructed (tabs 1–5) and its wave added in tab 7
("add current reconstruction" with h k l); alternatively amplitude/phase pairs
(`[200]_amplitude.dm3` + `[200]_phase.dm3`, also .img/.npy/.tif), complex waves (FRWR3
`avgWave.img`, .npy) or the DF stacks of `createDFstack.s` (with their `DFHoloGPA` tags: hkl and
shift of each frame, unit cell, zone, x axis, reference area, resolution) can be loaded. The 2D
g vectors follow from the unit cell (a, b, c, α, β, γ), the zone axis [uvw] and the reflection
that defines the image x axis (plus an optional in-plane rotation); x points right, y up, z
towards the viewer as in the DM scripts. Per pixel, the displacement gradient is the weighted
least-squares solution (weights A_j², the DF intensity) of
∂φ_j/∂x_d = −2π Σ_i g_ji ∂u_i/∂x_d, after subtracting each reflection's mean phase gradient in
the reference area; the results ε_xx, ε_yy, ε_xy and the rotation ω are band-limited to the
chosen resolution (cosine edge). With one reflection, ε_gg along g is returned; the
displacement mode combines the unwrapped phases into u_x, u_y. Shifts between the DF data sets
can be set per reflection or found by correlating the amplitudes, and a red/green overlay
checks the alignment. The maps are written to the NXem export of tab 6.

On the DM test data (`DIH/TestData`, GPA of a synthetic HRTEM image with a 0.456 nm cubic lattice)
the four reflections give ε_xx ≈ +4 % in the right part and ε_yy ≈ +4 % in the lower part of the
image, in agreement with the lattice spacings measured directly in the HRTEM image.

### Image files

* FRWR3/QSTEM `.img` (binary header: header size, number of parameters, comment length, Nx, Ny,
  complex flag, bytes per value, version; then t, dx, dy as doubles, the parameters and the
  comment; data stored with x running slowest). Selecting one frame of an indexed series
  (`image_3.img`) loads `image_0.img, image_1.img, ...` in numerical order; dx (Å) gives the pixel
  size and t, which FRWR3 uses for the defocus in Å, the suggested defoci.
* "load all frames of this folder" (file browser of tab 1) reads a folder of single frames as a
  series, ordered by the defocus in the file names (`F-1000nm.dm3`, `F250nm.dm3`, …), which is
  then used as defocus; uploads of several files are ordered the same way.

### Remaining-time estimates of FRWR3 runs

The time of FRWR3 loops differs by two orders of magnitude: the loops in which the distortions
are fitted and the planes re-read cost ~3 s per image on the 1600×1480 CeO2 frames (binning 2),
plain projections 0.06 s per image, and these costs grow with the number of images of a stage.
`control.ScheduleETA` therefore takes the planned loop sequence (`FRWR3.planned_schedule()`, with
the planes actually chosen after the initialization), learns the cost per image of every kind of
loop online, extrapolates it to the larger later stages and predicts the remaining time from all
loops still to come (plus the warping of the full-resolution frames in tab 4b). On the CeO2
series (472 s) the estimate is within 5 % from the 9-image stage on (after 90 s).


## NOMAD: import and push results back

Tab 1, *import from NOMAD*: NOMAD production, the test server or an Oasis (API URL), an **API token** (NOMAD user
menu → API tokens / app token, with permission to read and write uploads; kept only in memory for the session) and
the **upload ID** of the NOMAD project ("my uploads" lists your uploads). Browse the raw files of the upload,
then *download and load selected* (an indexed .img frame loads its whole series) or *download folder as series*
(single frames named by defocus). The files are cached in `nomad_downloads/<upload ID>/`. With *push results
automatically back*, every file written afterwards (DM3 → NXem conversion in tab 1, tab 4b, tab 6) is uploaded to
the same upload, into the given **subfolder**. Tab 6, *Upload the generated files to NOMAD*, sends all files written
in this session, to the import's upload (filled in) or, without an upload ID, to a new upload. Published uploads
cannot be changed (NOMAD answers 403). The same conventions as the NOMAD upload of the TEM Alignment Twin; the API
calls (`/uploads`, `/uploads/{id}/rawdir/{path}`, `/uploads/{id}/raw/{path}` GET/PUT, `POST /uploads`) are tested
against a local mock of the NOMAD API (`tests/test_nomad.py`).

## Running without Jupyter (web server, Galaxy)

See `galaxy/README.md`: `solara run webapp/solara_app.py` serves the same GUI from its own web server, and
`galaxy/` contains a Dockerfile and a draft Galaxy Interactive Tool.


## License

BSD 3-Clause License, Copyright (c) 2008–2026 Christoph T. Koch — see [LICENSE](../LICENSE). Python files
carry the SPDX identifier `BSD-3-Clause`.
