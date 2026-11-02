#!/usr/bin/env python3
"""Generate the hero figure: what the trace attributed vs what bounds allow.

Left panel  -- measured per-rank all-reduce kernel duration at one AR site
               (TP8 prefill, iteration 8, layer 1, phase 0, stock-Triton
               baseline). Seven ranks sit inside the collective for ~8 ms
               while rank 0 is still doing host work upstream.
Right panel -- the three timescales that could explain that duration, on a
               log axis: bytes actually on the wire, waiting for the slow
               peer, and the host-side stall.

The left panel is measurement. The right panel mixes one measurement (the
on-wire cost observed once ranks arrive together) with one analytic bound
(payload / link bandwidth) and one separately-observed phenomenon (the
`_fused_add_rmsnorm_pad` stall). The caption must keep those distinct.

Usage:
    python make_hero_attribution.py --input iter8_l1p0_per_rank_ar.csv --output-dir .
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Right-panel quantities, all sourced from the article body -------------
#
# Payload: a 4096-token prefill chunk all-reduces M x 2880 x 2 B = 22.5 MB.
# TP8 loses QuickReduce's INT4 path, so it goes in full BF16; a two-pass
# collective puts 2*(p-1)/p ~= 1.75x of that on the wire, about 39 MB.
WIRE_MB = 39
WIRE_MS = (0.16, 0.26)          # 39 MB at 240 and 150 GB/s effective
WIRE_MEASURED_MS = 0.22         # observed once all 8 ranks arrive together
HOST_STALL_MS = (130.0, 160.0)  # _fused_add_rmsnorm_pad, hits all 8 ranks

SLOW_RANK = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("iter8_l1p0_per_rank_ar.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load(path: Path) -> list[tuple[int, float]]:
    with path.open(newline="") as fh:
        return [(int(r["rank"]), float(r["max_ar_duration_ms"]))
                for r in csv.DictReader(fh)]


def draw_attribution(ax, per_rank: list[tuple[int, float]]) -> float:
    ranks = [r for r, _ in per_rank]
    durs = [d for _, d in per_rank]
    spread = max(durs) - min(durs)

    colors = ["tab:red" if r == SLOW_RANK else "tab:blue" for r in ranks]
    ax.barh(ranks, durs, color=colors, height=0.68)
    for rank, dur in per_rank:
        ax.text(dur + 0.18, rank, f"{dur:.2f}", va="center", fontsize=9,
                color="0.25")

    right = max(durs) * 2.05
    ax.annotate(
        f"R{SLOW_RANK} is still busy with host work upstream,\n"
        f"so it enters the collective last and leaves almost at once",
        xy=(per_rank[SLOW_RANK][1], SLOW_RANK - 0.30), xytext=(0.85, -0.80),
        arrowprops=dict(arrowstyle="->", color="tab:red"),
        fontsize=9, color="tab:red", va="center", ha="left",
    )
    ax.text(max(durs) * 1.28, 4.0,
            f"the other 7 ranks\narrived early and wait\ninside the collective\n\n"
            f"spread = {spread:.2f} ms",
            fontsize=9, color="tab:blue", va="center", ha="left")

    ax.set_yticks(ranks)
    ax.set_yticklabels([f"R{r}" for r in ranks])
    ax.set_ylim(7.6, -1.15)
    ax.set_xlabel("All-reduce kernel duration (ms)")
    ax.set_title("What the trace attributed\nTP8 prefill, iter 8, layer 1, "
                 "phase 0", fontsize=11)
    ax.set_xlim(0, right)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return spread


def draw_bounds(ax, spread: float) -> None:
    bands = [
        (WIRE_MS, f"Bytes on the wire\n{WIRE_MB} MB two-pass @ 150-240 GB/s",
         "tab:green"),
        ((spread * 0.985, spread * 1.015), "Waiting for the slow peer\n"
         "cross-rank spread at this AR", "tab:blue"),
        (HOST_STALL_MS, "Host-side stall\n_fused_add_rmsnorm_pad, all 8 ranks",
         "tab:red"),
    ]
    for y, ((lo, hi), label, color) in enumerate(bands):
        ax.barh(y, hi - lo, left=lo, height=0.5, color=color)
        text = f"{lo:g}-{hi:g} ms" if hi / lo > 1.05 else f"{lo / 0.985:.2f} ms"
        ax.text(hi * 1.25, y, text, va="center", fontsize=9, color="0.25")
        ax.text(lo * 0.82, y, label, va="center", ha="right", fontsize=9)

    ax.plot([WIRE_MEASURED_MS], [0], marker="|", markersize=16,
            markeredgewidth=2, color="0.15")
    ax.text(WIRE_MEASURED_MS, 0.34, f"measured {WIRE_MEASURED_MS} ms once\n"
            "all 8 ranks arrive together", ha="center", va="top",
            fontsize=8, color="0.15")

    floor = WIRE_MS[1]
    ax.annotate(f"~{spread / floor:.0f}x the transfer cost",
                xy=(spread, 1), xytext=(spread, 1.42), fontsize=9,
                color="tab:blue", ha="center")
    ax.annotate(f"~{HOST_STALL_MS[0] / floor:.0f}x",
                xy=(HOST_STALL_MS[0], 2), xytext=(HOST_STALL_MS[0], 2.42),
                fontsize=9, color="tab:red", ha="center")

    ax.set_xscale("log")
    ax.set_xlim(0.02, 4000)
    ax.set_ylim(2.85, -0.62)
    ax.set_yticks([])
    ax.set_xlabel("Time (ms, log scale)")
    ax.set_title("What the numbers allow it to be\nOnly the green band is "
                 "data movement", fontsize=11)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_rank = load(args.input)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9),
                             gridspec_kw={"width_ratios": [1.0, 1.25],
                                          "wspace": 0.10})
    spread = draw_attribution(axes[0], per_rank)
    draw_bounds(axes[1], spread)

    fig.suptitle("Attribution Is Not Causality\n"
                 "The profiler correctly reports time inside the all-reduce; "
                 "almost none of it is data movement",
                 fontsize=13, y=1.06)
    fig.savefig(args.output_dir / "hero-trace-vs-bound.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output_dir / 'hero-trace-vs-bound.png'}")


if __name__ == "__main__":
    main()
