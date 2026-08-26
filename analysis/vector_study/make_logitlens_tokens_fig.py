#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logit-lens vocabulary readout of the steering vector, per layer.
For a set of representative layers, show the top tokens the vector maps onto
(v @ W_U^T). Low/mid layers -> gibberish subword fragments (a diffuse internal
mode, no real word). High layers -> a tight cluster of reflection / discourse
words (Wait, But, Perhaps, Alternatively, 但, 然而...): direct token manipulation.

Tokens are taken verbatim from out/logitlens_toptokens_qwen3-4b.txt.

Rerun:  python make_logitlens_tokens_fig.py
Output: figs/opd_logitlens-tokens.svg
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "opd_logitlens-tokens.svg")

# (layer, [top tokens], regime)  regime: "low" | "mid" | "high"
# verbatim English top tokens from logitlens_toptokens_qwen3-4b.txt (CJK omitted
# for rendering; the English tokens already carry the point).
ROWS = [
    (2,  ["uled", "contingent", "dehy", "credible", "yielding", "ython", "Moon"], "low"),
    (5,  ["grounds", "oron", "_periods", "periods", "ObjectContext", "abler"], "low"),
    (8,  ["encer", "oola", "/GPL", "uelles", "DECL", "synchronized", "INCLUDED"], "low"),
    (15, [".qq", "HAR", "Mattis", "Relatives", "corridors", "/pi", "conting"], "mid"),
    (20, ["ngh", "WARD", "depicted", "svg", "kj", "^-", "=?"], "mid"),
    (26, ["...", "perhaps", "Maybe", "But", "Alternatively", "Hmm"], "high"),
    (28, ["Wait", "But", "wait", "but", "nhưng", "perhaps"], "high"),
    (31, ["Wait", "But", "Alternatively", "perhaps", "nhưng"], "high"),
    (34, ["Wait", "wait", "waits", "But", "waited", "WAIT"], "high"),
]

REG_COLOR = {"low": "#2ca02c", "mid": "#c98a3a", "high": "#c0392b"}
REG_BG = {"low": "#eef7ee", "mid": "#fdf3e7", "high": "#fdeceb"}
REG_LABEL = {"low": "low", "mid": "mid", "high": "high"}


def main():
    plt.rcParams.update({
        "font.size": 16, "font.family": "DejaVu Sans",
        "figure.dpi": 400, "savefig.dpi": 400,
    })
    n = len(ROWS)
    fig, ax = plt.subplots(figsize=(9.2, 0.62 * n + 1.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, n)
    ax.axis("off")

    ax.text(0.055, n + 0.28, "Layer", fontsize=13, fontweight="bold", ha="center", color="#333")
    ax.text(0.52, n + 0.28, "Top tokens the steering vector maps to  (logit lens)",
            fontsize=13, fontweight="bold", ha="center", color="#333")

    for i, (L, toks, reg) in enumerate(ROWS):
        y = n - 1 - i + 0.5
        # row background
        ax.add_patch(plt.Rectangle((0.0, y - 0.42), 1.0, 0.84, facecolor=REG_BG[reg],
                                   edgecolor="white", lw=1.5, zorder=1))
        # layer label + regime chip
        ax.text(0.055, y, f"L{L}", fontsize=13.5, ha="center", va="center",
                color=REG_COLOR[reg], fontweight="bold", zorder=2)
        # tokens
        toks_str = "   ".join(t.strip() if t.strip() else "␣" for t in toks)
        ax.text(0.135, y, toks_str, fontsize=13.5, ha="left", va="center",
                color=REG_COLOR[reg], family="DejaVu Sans", zorder=2)

    # regime legend on the right side via colored labels
    from matplotlib.patches import Patch
    # handles = [Patch(facecolor=REG_BG[r], edgecolor=REG_COLOR[r],)
    #            for r in ["low", "high"]]
    # ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.16),
    #           fontsize=11, ncol=2, framealpha=0.95, edgecolor="#dddddd")

    fig.suptitle("What Each Layer's Vector Decodes to in the Vocabulary",
                 fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", format="svg", dpi=400)
    print(f"[ok] saved: {FIG}")


if __name__ == "__main__":
    main()
