# Building FRWR3 (2019 source) on Linux with FFTW 3.3 / netCDF 4

The patch `frwr3_linux_fftw33.patch` (apply in the source directory with `patch -p1 < frwr3_linux_fftw33.patch`)
fixes three problems (convert CRLF line endings first, e.g. `sed -i "s/\r$//" *.cpp`) found while validating the Python port (`inlineholo.frwr3`) against the C++ program:

1. **cdfFile.cpp** – `imap`/`stride` were `int[3]` arrays cast to `ptrdiff_t*` for `nc_get_varm_float`;
   on 64-bit systems every plane read fails ("Error reading focalplane").
2. **ImgAlignTools.cpp** – `fftwf_execute_dft()` was called with real-to-complex / complex-to-real plans.
   With FFTW 3.3 this silently produces wrong diffractograms (DC term = 0), the envelope/MTF convolution
   destroys the mean intensity and the reconstruction diverges. Replaced by `fftwf_execute_dft_r2c/_c2r`.
   The hard-coded `-1` x-offsets in `findSmallInBig()` / `findSmallInBigWholeImage()` (which compensated the
   old behaviour) were removed; otherwise the wave drifts by one pixel per re-read.
3. **WaveFunction.cpp** – the destructor called `imageData::~imageData()` explicitly (double free at exit).

Build (Ubuntu: `apt install libfftw3-dev libnetcdf-dev`; remove the bundled `fftw3.h`, `netcdf.h`, `ncvalues.h`):

```bash
for f in bicubint cdfFile frwrData imageData imagelib ImgAlignTools memory microscopeData readparams WaveFunction frwr3; do
  g++ -O3 -w -fpermissive -c $f.cpp -o $f.o; done
g++ -o frwr3 *.o -lfftw3f -lfftw3 -lnetcdf -lm
```

`inlineholo.frwr3_io.write_cdf()` writes the netCDF input, `read_img()` reads the `.img` output.


License: the patch is distributed under the BSD 3-Clause License of this folder (see `../LICENSE`),
Copyright (c) 2008–2026 Christoph T. Koch, the author of FRWR3.
