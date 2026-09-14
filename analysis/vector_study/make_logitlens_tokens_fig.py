#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logit-lens vocabulary readout of the steering vector, per layer.

For representative layers, this figure shows the highest-logit vocabulary
items obtained by projecting each learned steering vector through the output
RMSNorm and unembedding matrix.

Rerun:
    python make_logitlens_tokens_fig.py

Output:
    figs/opd_logitlens-tokens.svg
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from _panel_style import (
    FS_AXLABEL,
    FS_LEGEND,
    FS_TITLE,
    PANEL_FIGSIZE,
    panel_rc,
)


HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_logitlens-tokens.svg")


# ----------------------------------------------------------------------
# Data
# Keep all layer indices and token contents unchanged.
# ----------------------------------------------------------------------

ROWS = [
    (
        2,
        ["uled", "contingent", "dehy", "credible",
         "yielding", "ython", "Moon"],
        "low",
    ),
    (
        5,
        ["grounds", "oron", "_periods", "ObjectContext",
         "-sharing", "periods", "abler"],
        "low",
    ),
    (
        8,
        ["encer", "oola", "/GPL", "uelles",
         "DECL", "lul", "ynchronized"],
        "low",
    ),
    (
        15,
        [".qq", "HAR", "Mattis", "Relatives",
         "corridors", "/pi", "conting"],
        "mid",
    ),
    (
        20,
        ["ngh", "^-", "WARD", "depicted",
         "=?", "pto", "/svg"],
        "mid",
    ),
    (
        26,
        ["...", "perhaps", "Maybe", "Hmm",
         "But", "Alternatively", "Hmm"],
        "high",
    ),
    (
        28,
        ["Wait", "But", "...", "wait",
         "but", "nhưng", "perhaps"],
        "high",
    ),
    (
        31,
        ["Wait", "nhưng", "But", "Alternatively",
         "perhaps", "But", "Wait"],
        "high",
    ),
    (
        34,
        ["Wait", "Wait", "wait", "wait",
         "_wait", "waits", "But"],
        "high",
    ),
]


REG_COLOR = {
    "low": "#2ca02c",
    "mid": "#c98a3a",
    "high": "#c0392b",
}

REG_BG = {
    "low": "#eef7ee",
    "mid": "#fdf3e7",
    "high": "#fdeceb",
}


def main():
    # Use the same font and plotting style as the entropy panel.
    panel_rc()

    n_rows = len(ROWS)

    # Match the height of the entropy panel while keeping this panel wider.
    # The smaller width multiplier reduces the gaps between tokens.
    fig, ax = plt.subplots(
        figsize=(
            PANEL_FIGSIZE[0] * 1.2,
            PANEL_FIGSIZE[1],
        )
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, n_rows + 0.75)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------

    ax.set_title(
        "Logit Lens: Top Vocabulary Items",
        fontsize=FS_TITLE,
        pad=9,
    )

    # ------------------------------------------------------------------
    # Column headers
    # ------------------------------------------------------------------

    header_y = n_rows + 0.30

    layer_x = 0.055

    # Horizontal region occupied by the tokens.
    token_left = 0.17
    token_right = 0.93
    token_center = (token_left + token_right) / 2

    ax.text(
        layer_x,
        header_y,
        "Layer",
        fontsize=FS_AXLABEL,
        fontweight="bold",
        ha="center",
        va="center",
        color="#333333",
    )

    ax.text(
        token_center,
        header_y,
        "Highest-logit vocabulary items",
        fontsize=FS_AXLABEL,
        fontweight="bold",
        ha="center",
        va="center",
        color="#333333",
    )

    # ------------------------------------------------------------------
    # Table rows
    # ------------------------------------------------------------------

    for i, (layer, tokens, regime) in enumerate(ROWS):
        y = n_rows - 1 - i + 0.5

        # Row background.
        ax.add_patch(
            plt.Rectangle(
                (0.01, y - 0.40),
                0.98,
                0.80,
                facecolor=REG_BG[regime],
                edgecolor="white",
                linewidth=1.0,
                zorder=1,
            )
        )

        # Layer label.
        ax.text(
            layer_x,
            y,
            f"L{layer}",
            fontsize=FS_AXLABEL,
            fontweight="bold",
            ha="center",
            va="center",
            color=REG_COLOR[regime],
            zorder=2,
        )

        # Use a narrower horizontal range to reduce inter-token spacing.
        xs = np.linspace(
            token_left,
            token_right,
            len(tokens),
        )

        for x, token in zip(xs, tokens):
            display_token = token.strip() if token.strip() else "␣"

            ax.text(
                x,
                y,
                display_token,
                fontsize=FS_LEGEND,
                ha="center",
                va="center",
                color=REG_COLOR[regime],
                family="DejaVu Sans",
                zorder=2,
            )

    # ------------------------------------------------------------------
    # Layout and export
    # ------------------------------------------------------------------

    fig.subplots_adjust(
        left=0.02,
        right=0.99,
        bottom=0.04,
        top=0.86,
    )

    os.makedirs(
        os.path.dirname(FIG),
        exist_ok=True,
    )

    # Do not use bbox_inches="tight"; otherwise the exported panel height
    # may differ from the entropy panel.
    fig.savefig(
        FIG,
        format="svg",
        bbox_inches=None,
        pad_inches=0,
    )

    plt.close(fig)

    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()