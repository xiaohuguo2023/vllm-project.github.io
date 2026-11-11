#!/usr/bin/env python3
"""Generate the swizzle-versus-strided figure.

Top panel  -- a heatmap over the 28 TP4/TP8 serving cells, showing swizzled
              MX-scale layout relative to the plain strided layout. Both
              paths were retuned before the comparison, so this is a fair
              layout A/B and not a tuned-vs-untuned artefact.
Bottom     -- why. The CDNA4 swizzled scale layout requires BLOCK_K >= 256.
              The decode down-projection contracts over K = 2880 / TP, so at
              TP8 the K extent the kernel steps through goes from 384 to 512
              elements: 152 of the 512 are masked, against 24 of the 384.

Data provenance
---------------
`swizzle_ab_28cell.csv` is extracted from two 28-cell serving sweeps run in
the same container image with cold-cache parity (vLLM torch-compile, Triton
and inductor caches purged before every cell):

  strided  : hybrid swizzle gate, swizzle off at TP >= 4
  swizzled : swizzle forced on at every TP, with the dispatch JSON retuned
             under the BLOCK_K >= 256 constraint the swizzled path imposes

The BLOCK_K values in the bottom panel are read off those two tuned dispatch
JSONs, not assumed.

Usage:
    python make_swizzle_gate.py --input swizzle_ab_28cell.csv --output-dir .
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

# Cells this close to parity are inside run-to-run noise for this harness and
# are counted as ties rather than wins for either layout.
NOISE_PCT = 2.0

ROWS = [
    (1024, 1024, 4, "TP4   1K/1K"),
    (1024, 1024, 8, "TP8   1K/1K"),
    (8192, 1024, 4, "TP4   8K/1K"),
    (8192, 1024, 8, "TP8   8K/1K"),
]
CONCS = [4, 8, 16, 32, 64, 128, 256]

# Down-projection: N = 2880 (model hidden), K = 2880 / TP per rank.
# BLOCK_K values are the tuned bm16 down-projection entries from each JSON:
#   strided   bm16_n2880_k720 -> 128,  bm16_n2880_k360 -> 128
#   swizzled  bm16_n2880_k720 -> 256,  bm16_n2880_k360 -> 512
PADDING = [
    ("TP4", 720, 128, 256),
    ("TP8", 360, 128, 512),
]
SWIZZLE_MIN_BK = 256

C_STRIDED = "#2f7d4f"
C_SWIZZLE = "#7a4fa3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("swizzle_ab_28cell.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load(path: Path) -> dict[tuple[int, int, int, int], tuple[float, float]]:
    with path.open(newline="") as fh:
        return {
            (int(r["isl"]), int(r["osl"]), int(r["tp"]), int(r["concurrency"])):
                (float(r["strided_tok_s"]), float(r["swizzled_tok_s"]))
            for r in csv.DictReader(fh)
        }


def draw_heatmap(ax, cells: dict) -> tuple[int, int, int]:
    delta = np.array([
        [100.0 * (cells[(isl, osl, tp, c)][1] / cells[(isl, osl, tp, c)][0] - 1.0)
         for c in CONCS]
        for isl, osl, tp, _ in ROWS
    ])

    im = ax.imshow(delta, cmap="coolwarm", vmin=-20, vmax=20, aspect="auto")

    strided_wins = swizzle_wins = ties = 0
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            d = delta[i, j]
            if d <= -NOISE_PCT:
                strided_wins += 1
            elif d >= NOISE_PCT:
                swizzle_wins += 1
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none",
                                       edgecolor="black", linewidth=2.4, zorder=4))
            else:
                ties += 1
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none",
                                       edgecolor="0.35", linewidth=1.0,
                                       hatch="xxx", zorder=3))
            tie = abs(d) < NOISE_PCT
            ax.text(j, i, f"{d:+.1f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(d) > 11 else "0.12",
                    fontweight="normal" if tie else "bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="none", alpha=0.85) if tie else None)

    ax.set_xticks(range(len(CONCS)))
    ax.set_xticklabels([str(c) for c in CONCS])
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([lbl for *_, lbl in ROWS], fontsize=10.5)
    ax.set_xlabel("Concurrent requests")
    ax.axhline(1.5, color="white", linewidth=3)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.annotate(
        "the only region where the swizzled layout still wins:\n"
        "prefill-heavy traffic at high concurrency",
        xy=(6.52, 3.20), xytext=(5.55, 4.62), fontsize=9.5, color="0.15",
        ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color="0.15",
                        connectionstyle="arc3,rad=-0.22"),
        annotation_clip=False, zorder=6,
    )
    ax.text(-1.38, 0.5, "decode-heavy:\nevery cell lost", rotation=90,
            fontsize=9.5, color="0.30", ha="center", va="center")

    cbar = ax.figure.colorbar(im, ax=ax, pad=0.015, fraction=0.030)
    cbar.set_label("swizzled minus strided output throughput (%)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=9)

    handles = [
        Patch(facecolor="#4b6fa8", edgecolor="none", label="strided layout faster"),
        Patch(facecolor="#c1553f", edgecolor="black", linewidth=2.0,
              label="swizzled layout faster"),
        Patch(facecolor="white", edgecolor="0.35", hatch="xxx",
              label=f"within +/-{NOISE_PCT:g}% (tie)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.20),
              ncol=3, frameon=False, fontsize=9.5, handlelength=1.7)

    return strided_wins, swizzle_wins, ties


def draw_padding(ax) -> None:
    """Show how BLOCK_K >= 256 inflates the K extent the kernel steps through."""
    bar_h, group_gap = 0.34, 1.05
    ymax = 0.0
    for g, (tp, k, bk_str, bk_swz) in enumerate(PADDING):
        base = g * group_gap
        for row, (bk, color, name) in enumerate(
                ((bk_str, C_STRIDED, "strided"), (bk_swz, C_SWIZZLE, "swizzled"))):
            steps = math.ceil(k / bk)
            extent = steps * bk
            masked = extent - k
            y = base + row * (bar_h + 0.06)
            ymax = max(ymax, y + bar_h)

            ax.add_patch(Rectangle((0, y), k, bar_h, facecolor=color,
                                   edgecolor="white", linewidth=0.8, zorder=3))
            ax.add_patch(Rectangle((k, y), masked, bar_h, facecolor="white",
                                   edgecolor=color, linewidth=1.3, hatch="///",
                                   zorder=3))
            for s in range(1, steps):
                ax.plot([s * bk, s * bk], [y, y + bar_h], color="white",
                        linewidth=1.4, zorder=4)

            ax.text(-14, y + bar_h / 2, name, ha="right", va="center",
                    fontsize=9.5, color=color, fontweight="bold")
            ax.text(extent + 16, y + bar_h / 2,
                    f"BLOCK_K={bk} x {steps} = {extent}   "
                    f"({masked} masked, +{100 * masked / k:.1f}%)",
                    ha="left", va="center", fontsize=9.5, color="0.20")

        ax.text(-14, base + bar_h + 0.03, f"{tp}:  K = 2880/{tp[2:]} = {k}",
                ha="right", va="center", fontsize=10.5, fontweight="bold",
                color="0.12")

    ax.text(-320, ymax + 0.34,
            f"The swizzled MX-scale layout requires BLOCK_K >= {SWIZZLE_MIN_BK}. "
            f"At TP8 that alone forces K=360 up to a 512-element extent -- "
            f"even BLOCK_K={SWIZZLE_MIN_BK} rounds to 512.\n"
            "At TP4 both paths land on 768, so padding does not explain the TP4 "
            "regressions; there the constraint only doubles the K step.",
            fontsize=9.5, color="0.30", ha="left", va="center")

    ax.set_xlim(-320, 1180)
    ax.set_ylim(-0.22, ymax + 0.70)
    ax.set_yticks([])
    ax.set_xticks([0, 256, 360, 512, 720, 768])
    ax.set_xlabel("Contraction dimension K of the MoE down-projection (elements)")
    ax.set_title("Why: the block-size constraint the swizzled layout imposes",
                 fontsize=11.5, loc="left", pad=10)
    ax.grid(axis="x", alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.legend(handles=[Patch(facecolor="0.55", edgecolor="white", label="real K"),
                       Patch(facecolor="white", edgecolor="0.35", hatch="///",
                             label="masked padding")],
              loc="upper right", bbox_to_anchor=(1.0, -0.15), frameon=False,
              fontsize=9, handlelength=1.7, ncol=2)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = load(args.input)

    fig, (ax, axp) = plt.subplots(
        2, 1, figsize=(13.2, 9.9),
        gridspec_kw={"height_ratios": [1.55, 1.0], "hspace": 0.52},
    )
    strided_wins, swizzle_wins, ties = draw_heatmap(ax, cells)
    draw_padding(axp)

    ax.set_title(
        "Retuning Both Paths, the Plain Strided Layout Wins Almost Everywhere\n"
        "gpt-oss-120B W4A8, MI355X, 28 TP4/TP8 serving cells, cold-cache parity\n"
        f"strided wins {strided_wins}, swizzled wins {swizzle_wins}, "
        f"{ties} within noise",
        fontsize=12.5, pad=14,
    )
    fig.savefig(args.output_dir / "swizzle-gate-heatmap.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output_dir / 'swizzle-gate-heatmap.png'}")
    print(f"  strided {strided_wins} / swizzled {swizzle_wins} / ties {ties}"
          f"  (noise band +/-{NOISE_PCT:g}%)")


if __name__ == "__main__":
    main()
