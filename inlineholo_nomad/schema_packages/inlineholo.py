# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""NOMAD schema of the inline-holography measurement ELN written by the inlineHolo app (tab 6).

``ELNInlineHolography`` is a ``Measurement`` (NOMAD base section) in the *NOMAD measurements* category of the
nomad-measurements plugin (FAIRmat-NFDI/nomad-measurements) with the imaging settings, the focal-series data
set(s), the reconstructions and the NXem results file; the processing steps of the session are its ``steps``
(each with its parameters), the ``description`` explains the applied workflow.  The workflow graph itself is the
separate ``<name>.workflow.archive.yaml`` entry (``workflow2``) written next to it.
"""
from nomad.datamodel.data import ArchiveSection, EntryData
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.datamodel.metainfo.basesections import Measurement
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection

try:  # nomad-measurements: category "NOMAD measurements"
    from nomad_measurements.general import NOMADMeasurementsCategory
    _CATEGORIES = [NOMADMeasurementsCategory]
except Exception:  # noqa: BLE001
    _CATEGORIES = []

m_package = SchemaPackage(name="inlineholo_nomad")


def _num(unit=None, display=None, **kw):
    ann = dict(component=ELNComponentEnum.NumberEditQuantity)
    if display:
        ann["defaultDisplayUnit"] = display
    return Quantity(type=float, unit=unit, a_eln=ELNAnnotation(**ann), **kw)


class InlineHoloSettings(ArchiveSection):
    """Imaging settings of the focal series (as used for the reconstruction)."""
    m_def = Section(label="Imaging settings")
    acceleration_voltage = _num("V", "kV")
    pixel_size = _num("nm", description="pixel size in the object plane")
    number_of_images = Quantity(type=int, a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity))
    defocus_min = _num("nm", description="smallest defocus C1 of the series")
    defocus_max = _num("nm", description="largest defocus C1 of the series")
    spherical_aberration = _num("nm", "mm", description="C3")
    focal_spread = _num("nm", description="focal spread (temporal coherence)")
    convergence_semi_angle = _num("mrad", description="illumination convergence (spatial coherence)")


class FocalSeriesFile(ArchiveSection):
    """A data file of the focal series."""
    m_def = Section(label_quantity="name")
    name = Quantity(type=str, a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity))
    file = Quantity(type=str, a_browser=dict(adaptor="RawFileAdaptor"),
                    a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity))


class InlineHoloReconstruction(ArchiveSection):
    """One exit-wave reconstruction of the session (the wave itself is in the NXem file)."""
    m_def = Section(label_quantity="method")
    method = Quantity(type=str, a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity))
    iterations = Quantity(type=int)
    runtime = Quantity(type=float, unit="s")
    final_r_value = Quantity(type=float, description="final R value (data mismatch) of the iterative methods")


class ELNInlineHolography(Measurement, EntryData):
    """Inline electron holography: exit-wave (amplitude and phase) reconstruction from a TEM focal series with
    the inlineHolo app."""
    m_def = Section(categories=_CATEGORIES, label="Inline electron holography (inlineHolo)",
                    a_eln=ELNAnnotation(lane_width="600px"))
    settings = SubSection(section_def=InlineHoloSettings)
    focal_series = SubSection(section_def=FocalSeriesFile, repeats=True)
    reconstructions = SubSection(section_def=InlineHoloReconstruction, repeats=True)
    nxem_file = Quantity(type=str, description="NXem file with the focal series and the reconstructions",
                         a_browser=dict(adaptor="RawFileAdaptor"),
                         a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity))
    results_summary = Quantity(type=str, a_eln=ELNAnnotation(component=ELNComponentEnum.RichTextEditQuantity))
    analysis_software = Quantity(type=str, default="inlineholo",
                                 a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity))

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
        if not self.method:
            self.method = "inline electron holography"


m_package.__init_metainfo__()
