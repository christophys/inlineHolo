# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
from nomad.config.models.plugins import SchemaPackageEntryPoint


class InlineHoloSchemaEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from .inlineholo import m_package

        return m_package


inlineholo_schema = InlineHoloSchemaEntryPoint(
    name="inlineHolo inline-holography measurement",
    description="ELN of an inline electron holography measurement (focal series, exit-wave reconstruction) written "
                "by the inlineHolo app, a Measurement in the NOMAD measurements category of nomad-measurements.",
)
