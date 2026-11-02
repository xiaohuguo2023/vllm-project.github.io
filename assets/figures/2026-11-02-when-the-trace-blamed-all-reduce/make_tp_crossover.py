#!/usr/bin/env python3
"""Generate the TP4/TP8 crossover figure used by the blog.

Reads the published 56-cell serving sweep CSV and plots output token
throughput against concurrency for TP4 and TP8, on both request shapes,
plus the TP8/TP4 ratio so the crossover point is explicit.

The figure makes one claim only: TP8 does not beat TP4 at every operating
point. It is not an efficiency-per-GPU comparison -- per GPU, TP4 leads at
every concurrency in this sweep, and the appendix Pareto figures cover that
view.

Usage:
    python make_tp_crossover.py --input serving_sweep_56cell.csv --output-dir .
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# gpt-oss-120B MoE down-projection contracts over K = 2880, sharded by TP.
# This is the kernel-shape cost of raising TP: each rank sees a shallower K.
DOWN_PROJ_K = 2880

SHAPES = [
    (1024, 1024, "1K/1K (decode-heavy)"),
    (8192, 1024, "8K/1K (prefill-heavy)"),
]
TP_PAIR = (4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("serving_sweep_56cell.csv"),
                        help="56-cell sweep CSV published with the article")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load(path: Path) -> dict:
    """Return {(isl, osl, tp): {concurrency: output_tok_s}}."""
    table: dict = defaultdict(dict)
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (int(row["isl"]), int(row["osl"]), int(row["tp"]))
            table[key][int(row["concurrency"])] = float(row["output_tok_s"])
    return table


def crossover(concs: list[int], ratios: list[float]) -> float | None:
    """First concurrency where the TP8/TP4 ratio crosses 1.0.

    Interpolated in log2(concurrency), which is the axis the sweep samples on.
    """
    for i in range(len(ratios) - 1):
        lo, hi = ratios[i], ratios[i + 1]
        if lo < 1.0 <= hi:
            frac = (1.0 - lo) / (hi - lo)
            x = math.log2(concs[i]) + frac * (math.log2(concs[i + 1]) - math.log2(concs[i]))
            return 2.0 ** x
    return None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = load(args.input)

    fig, axes = plt.subplots(
        2, 2, figsize=(11.2, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.14, "wspace": 0.22},
    )

    for col, (isl, osl, label) in enumerate(SHAPES):
        top, bottom = axes[0][col], axes[1][col]

        series = {}
        for tp in TP_PAIR:
            cells = table[(isl, osl, tp)]
            concs = sorted(cells)
            series[tp] = (concs, [cells[c] for c in concs])

        concs = series[TP_PAIR[0]][0]
        assert concs == series[TP_PAIR[1]][0], "TP4 and TP8 must sample the same ladder"
        ratios = [b / a for a, b in zip(series[TP_PAIR[0]][1], series[TP_PAIR[1]][1])]

        for tp, marker in zip(TP_PAIR, ("o", "s")):
            xs, ys = series[tp]
            top.plot(xs, ys, marker=marker, linewidth=2, markersize=6,
                     label=f"TP{tp}  (down-proj K = {DOWN_PROJ_K // tp}/rank)")

        cross = crossover(concs, ratios)
        if cross is not None:
            for ax in (top, bottom):
                ax.axvspan(concs[0], cross, color="0.85", zorder=0)
                ax.axvline(cross, color="0.35", linestyle="--", linewidth=1.2, zorder=1)
            top.text(concs[0] * 1.35, top.get_ylim()[1] * 0.88,
                     f"crossover at ~{cross:.1f} concurrent\n"
                     f"TP4 ahead below it ({ratios[0]:.2f}x at {concs[0]})",
                     fontsize=9, color="0.25", ha="left", va="center")

        top.set_xscale("log", base=2)
        top.set_xticks(concs)
        top.set_xticklabels([str(c) for c in concs])
        top.set_title(label, fontsize=12)
        top.legend(frameon=False, fontsize=9, loc="lower right")
        top.grid(alpha=0.25, linewidth=0.6)
        if col == 0:
            top.set_ylabel("Output token throughput (tok/s)")

        bottom.plot(concs, ratios, marker="D", markersize=5, linewidth=2, color="tab:purple")
        bottom.axhline(1.0, color="0.35", linewidth=1.0)
        bottom.set_ylim(0.88, max(ratios) + 0.06)
        bottom.set_xlabel("Concurrent requests")
        bottom.grid(alpha=0.25, linewidth=0.6)
        if col == 0:
            bottom.set_ylabel("TP8 / TP4")
        bottom.annotate(f"{ratios[-1]:.2f}x", xy=(concs[-1], ratios[-1]),
                        xytext=(-4, 6), textcoords="offset points",
                        fontsize=9, ha="right", color="tab:purple")

        for ax in (top, bottom):
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    fig.suptitle(
        "TP8 Only Overtakes TP4 Above a Concurrency Threshold\n"
        "gpt-oss-120B W4A8, MI355X, 56-cell serving sweep; shaded region is where TP4 wins",
        fontsize=13, y=0.985,
    )
    fig.savefig(args.output_dir / "tp4-tp8-crossover.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output_dir / 'tp4-tp8-crossover.png'}")


if __name__ == "__main__":
    main()
