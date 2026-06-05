"""Shared DSR publication-figure builders.

These are the figures used by:
  - ``scripts/analysis_rodents_complete_clean.py``     (rodent cells)
  - ``scripts/RSA_DSR_ROIs_simple.py``                 (human cells)

Both pipelines call the same three functions; changes here propagate to both
analyses. The implementations currently live in
``mc.analyse.analyse_ephys_clean`` (where they grew up) and are re-exported
here so client scripts can use a pipeline-neutral import path. If/when the
implementations are moved into this file, the public API stays unchanged.
"""

from mc.analyse.analyse_ephys_clean import (
    pub_figure_dsr_overview,
    pub_figure_example_subject,
    pub_figure_model_schematics,
    MODEL_LABELS,
    MODEL_KINDS,
)

# Also expose the A4 layout helper so callers can size their own figures
# consistently with the published ones.
from mc.plotting.figure_layout import (
    subpanel_figure,
    scaled_font_for,
    describe_figure,
    A4_WIDTH_IN,
    A4_HEIGHT_IN,
    USABLE_PAGE_WIDTH_IN,
    TARGET_FONT_PT,
    PUB_RC,
)

__all__ = [
    'pub_figure_dsr_overview',
    'pub_figure_example_subject',
    'pub_figure_model_schematics',
    'MODEL_LABELS', 'MODEL_KINDS',
    'subpanel_figure', 'scaled_font_for', 'describe_figure',
    'A4_WIDTH_IN', 'A4_HEIGHT_IN', 'USABLE_PAGE_WIDTH_IN',
    'TARGET_FONT_PT', 'PUB_RC',
]
