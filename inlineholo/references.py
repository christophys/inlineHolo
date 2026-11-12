# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Literature references of the reconstruction methods (shown in tab 5 of the app).

Each entry: (citation, DOI or None, URL).  DOIs were checked against the publishers' records;
where a classic paper has no DOI (Gerchberg & Saxton 1972, Optik), a related paper with DOI is
given as well.
"""
from __future__ import annotations

_R = {
    "teague": ("M. R. Teague, Deterministic phase retrieval: a Green's function solution, "
               "J. Opt. Soc. Am. 73, 1434–1441 (1983)", "10.1364/JOSA.73.001434"),
    "paganin_nugent": ("D. Paganin, K. A. Nugent, Noninterferometric phase imaging with partially coherent light, "
                       "Phys. Rev. Lett. 80, 2586–2589 (1998)", "10.1103/PhysRevLett.80.2586"),
    "gureyev": ("T. E. Gureyev, A. Roberts, K. A. Nugent, Partially coherent fields, the transport-of-intensity "
                "equation, and phase uniqueness, J. Opt. Soc. Am. A 12, 1942–1946 (1995)", "10.1364/JOSAA.12.001942"),
    "waller": ("L. Waller, L. Tian, G. Barbastathis, Transport of intensity phase-amplitude imaging with higher order "
               "intensity derivatives, Opt. Express 18, 12552–12561 (2010)", "10.1364/OE.18.012552"),
    "paganin_morgan": ("D. M. Paganin, K. S. Morgan, X-ray Fokker–Planck equation for paraxial imaging, "
                       "Sci. Rep. 9, 17537 (2019)", "10.1038/s41598-019-52284-5"),
    "morgan_paganin": ("K. S. Morgan, D. M. Paganin, Applying the Fokker–Planck equation to grating-based x-ray "
                       "phase and dark-field imaging, Sci. Rep. 9, 17465 (2019)", "10.1038/s41598-019-52283-6"),
    "eljarrat": ("A. Eljarrat, J. Müller, M. R. S. Huang, C. T. Koch, Multi-focus TIE algorithm including partial "
                 "spatial coherence and overlapping filters, Opt. Express 26, 11819–11833 (2018)",
                 "10.1364/OE.26.011819"),
    "parvizi_oe": ("A. Parvizi, W. Van den Broek, C. T. Koch, Gradient flipping algorithm: introducing non-convex "
                   "constraints in wavefront reconstructions with the transport of intensity equation, "
                   "Opt. Express 24, 8344–8359 (2016)", "10.1364/OE.24.008344"),
    "parvizi_asci": ("A. Parvizi, W. Van den Broek, C. T. Koch, Recovering low spatial frequencies in wavefront "
                     "sensing based on intensity measurements, Adv. Struct. Chem. Imaging 2, 3 (2016)",
                     "10.1186/s40679-016-0017-y"),
    "pam": ("M. Op de Beeck, D. Van Dyck, W. Coene, Wave function reconstruction in HRTEM: the parabola method, "
            "Ultramicroscopy 64, 167–183 (1996)", "10.1016/0304-3991(96)00058-7"),
    "schiske": ("P. Schiske, Image reconstruction by means of focus series, J. Microsc. 207, 154 (2002); "
                "reprint of: Zur Frage der Bildrekonstruktion durch Fokusreihen, Proc. 4th Eur. Reg. Conf. Electron "
                "Microsc., Rome, 145–146 (1968)", "10.1046/j.1365-2818.2002.01042.x"),
    "gerchberg_saxton": ("R. W. Gerchberg, W. O. Saxton, A practical algorithm for the determination of phase from "
                         "image and diffraction plane pictures, Optik 35, 237–246 (1972) (no DOI)", None),
    "misell": ("D. L. Misell, A method for the solution of the phase problem in electron microscopy, "
               "J. Phys. D: Appl. Phys. 6, L6–L9 (1973)", "10.1088/0022-3727/6/1/102"),
    "allen_oxley": ("L. J. Allen, M. P. Oxley, Phase retrieval from series of images obtained by defocus variation, "
                    "Opt. Commun. 199, 65–75 (2001)", "10.1016/S0030-4018(01)01556-5"),
    "allen2004": ("L. J. Allen, W. McBride, N. L. O'Leary, M. P. Oxley, Exit wave reconstruction at atomic "
                  "resolution, Ultramicroscopy 100, 91–104 (2004)", "10.1016/j.ultramic.2004.01.012"),
    "koch2008": ("C. T. Koch, A flux-preserving non-linear inline holography reconstruction algorithm for partially "
                 "coherent electrons, Ultramicroscopy 108, 141–150 (2008)", "10.1016/j.ultramic.2007.03.007"),
    "koch2014": ("C. T. Koch, Towards full-resolution inline electron holography, Micron 63, 69–75 (2014)",
                 "10.1016/j.micron.2013.10.009"),
    "ozsoy2022": ("C. Ozsoy-Keskinbora, W. Van den Broek, C. B. Boothroyd, R. E. Dunin-Borkowski, P. A. van Aken, "
                  "C. T. Koch, Synergistic use of gradient flipping and phase prediction for inline electron "
                  "holography, Sci. Rep. 12, 13294 (2022)", "10.1038/s41598-022-17373-y"),
    "coene1992": ("W. Coene, G. Janssen, M. Op de Beeck, D. Van Dyck, Phase retrieval through focus variation for "
                  "ultra-resolution in field-emission transmission electron microscopy, Phys. Rev. Lett. 69, "
                  "3743–3746 (1992)", "10.1103/PhysRevLett.69.3743"),
    "coene1996": ("W. M. J. Coene, A. Thust, M. Op de Beeck, D. Van Dyck, Maximum-likelihood method for "
                  "focus-variation image reconstruction in high resolution transmission electron microscopy, "
                  "Ultramicroscopy 64, 109–135 (1996)", "10.1016/0304-3991(96)00010-1"),
    "frwr_software": ("C. T. Koch, FRWR software (frwr3) and FRWRtools plug-in, Humboldt-Universität zu Berlin",
                      None, "https://www.physik.hu-berlin.de/en/sem/software/software_FRWR"),
}

#: method key -> (short description of what is cited, [reference keys])
METHOD_REFERENCES = {
    "tie": ("transport-of-intensity equation, uniform intensity", ["teague", "paganin_nugent"]),
    "tie_full": ("TIE with non-uniform intensity (Teague's auxiliary function)", ["teague", "gureyev"]),
    "tie_poly": ("higher-order TIE: intensity derivative from a polynomial fit through all planes", ["waller"]),
    "tie_fp": ("Fokker–Planck extension of the TIE (diffusion field)", ["paganin_morgan", "morgan_paganin"]),
    "mftie": ("multi-focus TIE with overlapping filters and partial spatial coherence", ["eljarrat"]),
    "gftie": ("gradient-flipping TIE", ["parvizi_oe", "parvizi_asci"]),
    "pam": ("parabola method (linear focal-series restoration)", ["pam"]),
    "schiske": ("Schiske's linear least-squares focal-series filter", ["schiske"]),
    "gs": ("Gerchberg–Saxton / Misell iteration", ["gerchberg_saxton", "misell"]),
    "iwfr": ("iterative wave-function reconstruction (IWFR)", ["allen_oxley", "allen2004"]),
    "frwr": ("full-resolution wave reconstruction (FRWR3)", ["koch2008", "koch2014", "frwr_software"]),
    "frwr_gf": ("FRWR3 with gradient flipping and incoherent background", ["koch2014", "parvizi_oe", "ozsoy2022"]),
    "gpfrwr": ("FRWR3 with gradient flipping and TIE phase prediction (GP-FRWR)", ["ozsoy2022", "koch2014"]),
    "frwr_gd": ("flux-preserving least squares (FRWR forward model), steepest descent", ["koch2008"]),
    "gpfrwr_gd": ("as frwr_gd, with gradient flipping and phase prediction", ["koch2008", "ozsoy2022"]),
    "gpfrwr_lsq": ("as gpfrwr_gd, with least-squares weighted phase prediction (inlineholo variant)",
                   ["koch2008", "ozsoy2022"]),
    "mal": ("maximum-likelihood (MAL) focal-series reconstruction", ["coene1996", "coene1992"]),
    "ml_poisson": ("Poisson maximum likelihood with incoherent quadrature (inlineholo); related to MAL",
                   ["coene1996"]),
}


def reference(key):
    r = _R[key]
    cit, doi = r[0], r[1]
    url = r[2] if len(r) > 2 else (f"https://doi.org/{doi}" if doi else None)
    return cit, doi, url


def references_html(methods=None):
    """HTML list of the references of the given method keys (all with references if None)."""
    keys = [k for k in (methods or METHOD_REFERENCES) if k in METHOD_REFERENCES]
    out = ["<div style='font-size:12px;line-height:1.45'>"]
    for k in keys:
        what, refs = METHOD_REFERENCES[k]
        items = []
        for rk in refs:
            cit, doi, url = reference(rk)
            link = (f" <a href='{url}' target='_blank'>{'doi:' + doi if doi else url}</a>" if url else "")
            items.append(f"<li>{cit}.{link}</li>")
        out.append(f"<p style='margin:6px 0 2px'><b><code>{k}</code></b> — {what}</p>"
                   f"<ul style='margin:0 0 4px 18px'>{''.join(items)}</ul>")
    out.append("</div>")
    return "".join(out)
