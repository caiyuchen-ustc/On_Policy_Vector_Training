#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-export ../opd_plots/panel_style so the figures in this folder share ONE spec.

The canvas and font numbers live in opd_plots/panel_style.py because that is
where they were measured against fig/opd_layer-method-delta.svg. Copying them
here would mean two files to keep in sync and would defeat the purpose, so this
module only fixes up sys.path and re-exports.

Use it exactly like panel_style:

    from _panel_style import PANEL_FIGSIZE, panel_rc, save_panel
"""

import os
import sys

_OPD_PLOTS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "opd_plots"))
if _OPD_PLOTS not in sys.path:
    sys.path.insert(0, _OPD_PLOTS)

from panel_style import *          # noqa: F401,F403  (re-export)
from panel_style import (          # noqa: F401  explicit: `import *` skips nothing
    DELTA_HEIGHT_PT, FS_AXLABEL, FS_CURVE_AXLABEL, FS_CURVE_TITLE, FS_LEGEND,
    FS_LEGEND_SMALL, FS_TICK, FS_TITLE, PANEL_AXES_RECT, PANEL_FIGSIZE, PANEL_H,
    PANEL_W, WIDE_AXES_RECT, WIDE_FIGSIZE, panel_rc, row_axes_rect, row_figsize,
    save_panel, save_row, save_wide,
)
