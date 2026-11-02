#!/usr/bin/env python3
"""Generate the bound-driven optimization workflow diagram.

Top row  -- the five-step loop the article follows: Measure, Bound, Falsify,
            Fix, Re-profile, with the loop closing back on Measure.
Bottom   -- the three bounding models, drawn as an expansion of the Bound
            step, because the whole point is that different bottlenecks obey
            different limits and the model has to match the mechanism.

This figure carries no measurements. Every number on it is a specification
value or a model definition quoted from the article body.

Usage:
    python make_workflow.py --output-dir .
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

STAGES = [
    ("1. Measure", "Find the largest\ninterval in the trace"),
    ("2. Bound", "Estimate how much\nof it the hardware work\ncan plausibly require"),
    ("3. Falsify", "Design a control\nexperiment that can\ndisprove the first\nexplanation"),
    ("4. Fix", "Change the actual\nlimiting layer"),
    ("5. Re-profile", "Let the next\nbottleneck surface"),
]
BOUND_INDEX = 1
FALSIFY_INDEX = 2

# The three models. Kept colour-consistent with the hero figure: blue is the
# collective, red is the host, and the kernel roofline gets its own orange.
MODELS = [
    (
        "tab:orange",
        "Kernel Roofline",
        r"$P_{max} = \min(P_{compute},\ AI \times BW_{HBM})$",
        "GEMMs, normalization, elementwise.\n"
        "MI355X spec roofs: 8 TB/s HBM,\n"
        "5.0 PFLOP/s on the W4A8 path,\n"
        "ridge at ~625 FLOP/byte.\n\n"
        "Answers: chase math utilization,\nor move fewer bytes?",
    ),
    (
        "tab:blue",
        "Collective latency-bandwidth",
        r"$T \approx \alpha R + \beta B_{wire} + T_{wait}$",
        "Large prefill messages: the\n"
        "bandwidth term dominates.\n"
        "Tiny decode messages: the\n"
        "latency term dominates.\n\n"
        r"Warning: $T_{wait}$ sits inside the"
        "\nprofiler range. A long range does\n"
        "not prove a busy interconnect.",
    ),
    (
        "tab:red",
        "Host timeline + control experiment",
        "no useful GPU roofline",
        "Dispatch stalls are diagnosed by\ncomparison, not by a ceiling:\n\n"
        "GPU kernel time vs end-to-end;\n"
        "a run vs an identical re-run;\n"
        "captured vs eager execution;\n"
        "isolated vs contended CPU;\n"
        "per-rank arrival at sync points.",
    ),
]

STAGE_W, STAGE_GAP = 0.174, 0.031
STAGE_X0 = 0.005
STAGE_Y0, STAGE_H = 0.72, 0.20

MODEL_W, MODEL_GAP = 0.300, 0.035
MODEL_X0 = 0.015
MODEL_Y0, MODEL_H = 0.115, 0.400

RAIL_Y = 0.605


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def box(ax, x, y, w, h, facecolor, edgecolor, lw=1.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.018",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=2,
    ))


def arrow(ax, start, end, color="0.35", lw=1.6, style="->", conn="arc3,rad=0",
          ls="-"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=15, color=color,
        linewidth=lw, linestyle=ls, connectionstyle=conn, zorder=1,
        shrinkA=0, shrinkB=0,
    ))


def stage_center(i: int) -> float:
    return STAGE_X0 + i * (STAGE_W + STAGE_GAP) + STAGE_W / 2


def model_center(i: int) -> float:
    return MODEL_X0 + i * (MODEL_W + MODEL_GAP) + MODEL_W / 2


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.6, 7.6))
    ax.set_xlim(-0.018, 1.018)
    ax.set_ylim(0.02, 1.04)
    ax.axis("off")

    # --- the five-step loop -------------------------------------------------
    for i, (title, body) in enumerate(STAGES):
        x = STAGE_X0 + i * (STAGE_W + STAGE_GAP)
        highlight = i == BOUND_INDEX
        box(ax, x, STAGE_Y0, STAGE_W, STAGE_H,
            facecolor="#eef3fa" if not highlight else "#dce9f7",
            edgecolor="tab:blue", lw=2.2 if highlight else 1.6)
        cx = x + STAGE_W / 2
        ax.text(cx, STAGE_Y0 + STAGE_H - 0.045, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color="#12365e", zorder=3)
        ax.text(cx, STAGE_Y0 + 0.062, body, ha="center", va="center",
                fontsize=8.5, color="0.25", zorder=3)

        if i:
            prev_right = x - STAGE_GAP
            arrow(ax, (prev_right, STAGE_Y0 + STAGE_H / 2),
                  (x, STAGE_Y0 + STAGE_H / 2), color="0.35", lw=2.0)

    # --- the loop closing back on Measure -----------------------------------
    loop_y = STAGE_Y0 + STAGE_H + 0.075
    ax.plot([stage_center(4), stage_center(4), stage_center(0)],
            [STAGE_Y0 + STAGE_H, loop_y, loop_y],
            color="0.45", linewidth=1.8, linestyle="--", zorder=1)
    arrow(ax, (stage_center(0), loop_y), (stage_center(0), STAGE_Y0 + STAGE_H),
          color="0.45", lw=1.8, ls="--")
    ax.text(0.5, loop_y + 0.018, "the next bottleneck is a different mechanism, "
            "so the bound changes too",
            ha="center", va="bottom", fontsize=9.5, color="0.45", style="italic")

    # --- fan from Bound down to the three models ----------------------------
    bound_cx = stage_center(BOUND_INDEX)
    arrow(ax, (bound_cx, STAGE_Y0), (bound_cx, RAIL_Y), color="0.35", lw=2.0,
          style="-")
    ax.plot([model_center(0), model_center(len(MODELS) - 1)], [RAIL_Y, RAIL_Y],
            color="0.35", linewidth=2.0, zorder=1)
    for i in range(len(MODELS)):
        arrow(ax, (model_center(i), RAIL_Y), (model_center(i), MODEL_Y0 + MODEL_H),
              color="0.35", lw=2.0)

    ax.text(bound_cx + 0.012, RAIL_Y + 0.035,
            "different bottlenecks obey different limits -- pick the model "
            "that matches the mechanism",
            ha="left", va="bottom", fontsize=9.5, color="0.30")

    # --- the three model panels ---------------------------------------------
    for i, (color, name, formula, body) in enumerate(MODELS):
        x = MODEL_X0 + i * (MODEL_W + MODEL_GAP)
        cx = x + MODEL_W / 2
        box(ax, x, MODEL_Y0, MODEL_W, MODEL_H,
            facecolor="white", edgecolor=color, lw=2.0)
        ax.add_patch(FancyBboxPatch(
            (x, MODEL_Y0 + MODEL_H - 0.062), MODEL_W, 0.062,
            boxstyle="round,pad=0.004,rounding_size=0.016",
            facecolor=color, edgecolor=color, linewidth=1.0, zorder=3,
        ))
        ax.text(cx, MODEL_Y0 + MODEL_H - 0.031, name, ha="center", va="center",
                fontsize=11 if len(name) <= 30 else 9.8,
                fontweight="bold", color="white", zorder=4)
        ax.text(cx, MODEL_Y0 + MODEL_H - 0.108, formula, ha="center",
                va="center", fontsize=10.5, color="0.15", zorder=4)
        ax.text(cx, MODEL_Y0 + 0.128, body, ha="center", va="center",
                fontsize=9, color="0.25", zorder=4)

    fig.suptitle(
        "A Bound-Driven Optimization Workflow\n"
        "The trace says where the time went; a bound says whether the work "
        "could have needed it",
        fontsize=13.5, y=1.005,
    )
    fig.savefig(args.output_dir / "workflow.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output_dir / 'workflow.png'}")


if __name__ == "__main__":
    main()
