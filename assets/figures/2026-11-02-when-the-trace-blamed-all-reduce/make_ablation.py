#!/usr/bin/env python3
"""Generate the ablation figure.

The figure is deliberately split into three panels, because the stages were
not all measured with the same metric and their gains must not be summed.

Left   -- the one ladder that WAS measured cumulatively: bottleneck-rank time
          inside a single profiled TP8 prefill window, baseline -> AITER path
          -> wider HIP graph capture. A waterfall is legitimate here because
          each step was taken on top of the previous one in the same window.
Right  -- the stages that were measured on their own metric, in their own
          configuration. They are drawn as cards, not as bars on a shared
          axis, precisely so that nothing invites adding them up.
Bottom -- the only end-to-end number: measured full-stack throughput against
          the previously published MI355X-vLLM results, per cell, plus the
          geometric mean per request shape. This is measured directly. It is
          not the product or sum of anything above it.

There is no per-stage serving-throughput ladder for a single cell, because
none was ever run that way: the stages were validated on the metric closest
to each change. That is why the left panel uses profiled-window time rather
than tokens per second.

Data provenance
---------------
`ablation_fullsweep.csv`  the 32 cells of the 56-cell sweep that have a
                          published baseline counterpart, both columns in
                          TOTAL token throughput (input + output), which is
                          what the published baseline reports.
LADDER / SEPARATE         quoted from the article's ablation table; the
                          profiled-window figures come from the same trace
                          clips as the earlier trace figures.

Usage:
    python make_ablation.py --input ablation_fullsweep.csv --output-dir .
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Cumulative, same profiled window, each step taken on top of the previous.
LADDER = [
    ("Baseline\nstock Triton MoE", 6.11, "28.1%"),
    ("+ AITER\nkernel path", 3.97, "22.5%"),
    ("+ wider HIP\ngraph capture", 3.94, "6-12%"),
]

# Measured separately, on their own metric, in their own configuration.
SEPARATE = [
    ("Grouped-GEMM tuning",
     "per-shape gfx950 W4A8 dispatch",
     "better tile and stage choices on the hot\n"
     "expert shapes; no standalone full-sweep\n"
     "number was measured, so none is quoted"),
    ("Swizzle gate",
     "28 TP4/TP8 serving cells",
     "strided layout wins 23 of 28 cells,\n"
     "by up to 18.9% on decode-heavy 1K/1K"),
    ("Host-stall removal",
     "repeated TP8 runs under CPU contention",
     "the 130-160 ms _fused_add_rmsnorm_pad\n"
     "stalls disappear; residual cross-rank\n"
     "imbalance settles at 6-12%"),
]

# The cell quoted as representative in the bottom panel: the decode-heavy cell
# whose measured speedup is closest to the decode-heavy geometric mean.
REPRESENTATIVE = (1024, 1024, 4, 32)

SHAPES = [(1024, "1K/1K  decode-heavy", "tab:blue"),
          (8192, "8K/1K  prefill-heavy", "tab:orange")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("ablation_fullsweep.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load(path: Path) -> dict[tuple[int, int, int, int], float]:
    with path.open(newline="") as fh:
        return {
            (int(r["isl"]), int(r["osl"]), int(r["tp"]), int(r["concurrency"])):
                float(r["ours_total_tok_s"]) / float(r["baseline_total_tok_s"])
            for r in csv.DictReader(fh)
        }


def geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def draw_ladder(ax) -> None:
    xs = range(len(LADDER))
    vals = [v for _, v, _ in LADDER]
    ax.bar(xs, vals, width=0.56, color=["#8d99a6", "#4c72b0", "#2f5d94"])

    for x, (label, v, imb) in zip(xs, LADDER):
        ax.text(x, v + 0.10, f"{v:.2f} s", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
        ax.text(x, 0.16, f"imbalance\n{imb}", ha="center", va="bottom",
                fontsize=8.5, color="white")

    for i in range(len(LADDER) - 1):
        lo, hi = vals[i], vals[i + 1]
        ax.plot([i + 0.28, i + 1 - 0.28], [lo, lo], color="0.35",
                linestyle="--", linewidth=1.1, zorder=4)
        ax.annotate("", xy=(i + 0.72, hi), xytext=(i + 0.72, lo),
                    arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.6))
        y = (lo + hi) / 2 if lo - hi > 0.5 else lo + 0.85
        ax.text(i + 0.80, y,
                f"-{lo - hi:.2f} s\n({100 * (lo - hi) / lo:.0f}%)",
                fontsize=9, color="tab:red", ha="left", va="center")

    ax.set_xticks(list(xs))
    ax.set_xticklabels([l for l, _, _ in LADDER], fontsize=9.5)
    ax.set_ylabel("Bottleneck-rank time in the profiled window (s)")
    ax.set_ylim(0, 7.4)
    ax.set_title("Measured cumulatively\nsame profiled TP8 prefill window, "
                 "each step on top of the last", fontsize=11, pad=8)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_separate(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Measured separately\ndifferent metric, different "
                 "configuration - these do not add up", fontsize=11, pad=8)

    h, gap = 0.250, 0.050
    for i, (name, metric, result) in enumerate(SEPARATE):
        y = 1.0 - (i + 1) * h - i * gap
        ax.add_patch(FancyBboxPatch(
            (0.012, y), 0.976, h,
            boxstyle="round,pad=0.004,rounding_size=0.02",
            facecolor="#f6f8fb", edgecolor="0.62", linewidth=1.2))
        ax.text(0.038, y + h - 0.055, name, fontsize=10.5, fontweight="bold",
                color="#12365e", va="center")
        ax.text(0.976, y + h - 0.055, metric, fontsize=9, color="0.45",
                ha="right", va="center", style="italic")
        ax.text(0.038, y + 0.075, result, fontsize=9, color="0.25", va="center")

    ax.text(0.5, 0.008,
            "no arithmetic across these panels: the layers interact,\n"
            "and only the bottom panel was measured end to end",
            fontsize=9, color="tab:red", ha="center", va="bottom")


def draw_sweep(ax, ratios: dict) -> None:
    x = 0
    ticks, ticklabels, boundaries = [], [], []
    rep_x = None
    for isl, label, color in SHAPES:
        cells = sorted((k for k in ratios if k[0] == isl),
                       key=lambda k: (k[2], k[3]))
        start = x
        for k in cells:
            rep = k == REPRESENTATIVE
            if rep:
                rep_x = x
            ax.bar(x, ratios[k], width=0.74, color=color,
                   edgecolor="black" if rep else "none",
                   linewidth=2.2 if rep else 0, zorder=3)
            ax.text(x, ratios[k] + 0.025, f"{ratios[k]:.2f}", ha="center",
                    va="bottom", fontsize=7.6, rotation=90, color="0.25")
            ticks.append(x)
            ticklabels.append(f"TP{k[2]}\n{k[3]}")
            x += 1
        gm = geomean([ratios[k] for k in cells])
        ax.plot([start - 0.45, x - 0.55], [gm, gm], color=color,
                linestyle="--", linewidth=2.0, zorder=4)
        # Labelled in the clear space above the bars, not on top of them.
        ax.text(start - 0.40, 2.96,
                f"{label}\ngeometric mean {gm:.2f}x  (dashed)", fontsize=10,
                color=color, ha="left", va="top", fontweight="bold", zorder=5)
        boundaries.append(x - 0.5)
        x += 1

    ax.axvline(boundaries[0], color="0.75", linewidth=1.0)
    ax.axhline(1.0, color="0.35", linewidth=1.2)

    ax.annotate(
        f"representative cell: TP{REPRESENTATIVE[2]}, 1K/1K, "
        f"{REPRESENTATIVE[3]} concurrent\n"
        f"{ratios[REPRESENTATIVE]:.2f}x, the cell closest to the "
        f"decode-heavy geometric mean",
        xy=(rep_x, ratios[REPRESENTATIVE] + 0.20), xytext=(rep_x + 1.4, 2.42),
        fontsize=9, color="0.12", ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="0.12",
                        connectionstyle="arc3,rad=0.16"), zorder=6)

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=7.4)
    ax.set_ylabel("Total token throughput\nrelative to published baseline")
    ax.set_ylim(0.95, 3.02)
    ax.set_xlim(-1.0, x - 0.6)
    ax.set_xlabel("Tensor parallelism and concurrent requests "
                  "(the 32 of 56 cells that have a published baseline counterpart)")
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ratios = load(args.input)

    fig = plt.figure(figsize=(13.8, 10.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92],
                          width_ratios=[1.0, 1.42],
                          hspace=0.46, wspace=0.20)
    draw_ladder(fig.add_subplot(gs[0, 0]))
    draw_separate(fig.add_subplot(gs[0, 1]))
    ax = fig.add_subplot(gs[1, :])
    draw_sweep(ax, ratios)

    gms = {isl: geomean([r for k, r in ratios.items() if k[0] == isl])
           for isl, _, _ in SHAPES}
    best = max(ratios.values())
    ax.set_title(
        "Measured end to end\n"
        f"full-stack throughput against the previously published MI355X-vLLM "
        f"results: {gms[1024]:.2f}x decode-heavy, {gms[8192]:.2f}x "
        f"prefill-heavy, up to {best:.2f}x on a single cell",
        fontsize=11, pad=10)

    fig.suptitle(
        "Ablation: Each Layer on the Metric That Was Actually Measured\n"
        "gpt-oss-120B W4A8, MI355X - the three panels use three different "
        "metrics and are not additive",
        fontsize=13.5, y=0.975)
    fig.savefig(args.output_dir / "ablation.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output_dir / 'ablation.png'}")
    for isl, label, _ in SHAPES:
        print(f"  {label}: geomean {gms[isl]:.3f}x")
    print(f"  best cell {best:.3f}x; representative "
          f"{REPRESENTATIVE} = {ratios[REPRESENTATIVE]:.3f}x")


if __name__ == "__main__":
    main()
