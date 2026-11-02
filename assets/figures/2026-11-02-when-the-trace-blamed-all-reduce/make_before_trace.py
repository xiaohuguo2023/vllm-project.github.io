#!/usr/bin/env python3
"""Generate the "before" trace figure: eight ranks, one all-reduce site.

This is a rendering of a real Kineto clip, not a mock-up. The input CSV is
extracted from the strict eight-rank clip of iteration 8, layer 1, phase 0 of
the stock-Triton (newbaseline) TP8 prefill trace -- the same window published
as `iter8_L1P0_8ranks_strict.json.gz` for opening in ui.perfetto.dev. Every
bar is a GPU kernel with its measured start and duration; nothing is
schematic.

What it shows:
  * all eight ranks leave the previous all-reduce together at 0.49 ms;
  * seven of them run the MoE block, reach the next all-reduce by ~1.2 ms,
    and then sit inside it;
  * rank 0 goes GPU-idle three times for 7.24 ms in total while the host
    catches up, and only reaches the collective at 8.95 ms;
  * all eight all-reduces return within 2 us of each other at ~9.117 ms.

The seven long blue bars are therefore not transfer time. They are the same
transfer, preceded by a wait whose length was set on rank 0's host thread.

Note on scope: the 130-160 ms `_fused_add_rmsnorm_pad` host stall discussed
elsewhere in the article is a different, separately observed phenomenon and
does not appear in this clip. The gaps annotated here are the ones actually
present in this window.

Regenerate the input CSV from the raw trace with:
    python3 - <<'EOF'
    import gzip, json, csv
    d = json.load(gzip.open(
        "reports/newbaseline_prefill/iter8_L1P0_8ranks_strict.json.gz", "rt"))
    ...  # see the article's data bundle
    EOF

Usage:
    python make_before_trace.py --input iter8_l1p0_gantt.csv --output-dir .
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

CATEGORY_STYLE = {
    "allreduce": ("tab:blue", "QuickReduce two-shot all-reduce"),
    "moe": ("tab:orange", "MoE grouped GEMM / SwiGLU / scatter"),
    "gemm": ("#5b6b7a", "dense GEMM (Tensile)"),
    "attention": ("tab:purple", "attention"),
    "other": ("#b9c2cb", "routing, norm, elementwise"),
}

ALIGN_IN_MS = 0.4855      # every rank leaves the previous all-reduce here
ALIGN_OUT_MS = 9.117      # every rank returns from this all-reduce here
SLOW_RANK = 0
GAP_MIN_MS = 0.4          # only annotate GPU-idle gaps longer than this
ZOOM = (8.55, 9.33)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("iter8_l1p0_gantt.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load(path: Path) -> dict[int, list[dict]]:
    by_rank: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            by_rank[int(row["rank"])].append({
                "start": float(row["start_ms"]),
                "dur": float(row["dur_ms"]),
                "cat": row["category"],
            })
    for rank in by_rank:
        by_rank[rank].sort(key=lambda e: e["start"])
    return dict(by_rank)


def idle_gaps(events: list[dict], min_ms: float) -> list[tuple[float, float]]:
    gaps, end = [], 0.0
    for e in events:
        if e["start"] - end > min_ms:
            gaps.append((end, e["start"]))
        end = max(end, e["start"] + e["dur"])
    return gaps


def draw_gantt(ax, by_rank: dict[int, list[dict]], xlim: tuple[float, float],
               bar_h: float = 0.56, gap_hatch: bool = True,
               headroom: float = 0.55) -> None:
    for rank, events in sorted(by_rank.items()):
        for e in events:
            color = CATEGORY_STYLE[e["cat"]][0]
            ax.add_patch(Rectangle(
                (e["start"], rank - bar_h / 2), max(e["dur"], 0.004), bar_h,
                facecolor=color, edgecolor="white", linewidth=0.35, zorder=3,
            ))
        if gap_hatch and rank == SLOW_RANK:
            for lo, hi in idle_gaps(events, GAP_MIN_MS):
                ax.add_patch(Rectangle(
                    (lo, rank - bar_h / 2), hi - lo, bar_h,
                    facecolor="none", edgecolor="tab:red", linewidth=1.1,
                    hatch="///", zorder=4,
                ))

    ax.set_yticks(sorted(by_rank))
    ax.set_yticklabels([f"R{r}" for r in sorted(by_rank)])
    ax.set_ylim(len(by_rank) - 0.45, -headroom)
    ax.set_xlim(*xlim)
    ax.grid(axis="x", alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_rank = load(args.input)

    gaps = idle_gaps(by_rank[SLOW_RANK], GAP_MIN_MS)
    total_gap = sum(hi - lo for lo, hi in gaps)
    slow_ar = max((e for e in by_rank[SLOW_RANK] if e["cat"] == "allreduce"),
                  key=lambda e: e["start"])
    waiter_ar = max((e for e in by_rank[1] if e["cat"] == "allreduce"),
                    key=lambda e: e["dur"])

    fig, (ax, axz) = plt.subplots(
        2, 1, figsize=(13.4, 8.8),
        gridspec_kw={"height_ratios": [2.5, 1.05], "hspace": 0.40},
    )

    # --- full window --------------------------------------------------------
    draw_gantt(ax, by_rank, (-0.12, 9.62), headroom=1.55)
    for x, text, ha in ((ALIGN_IN_MS, "all 8 ranks leave the previous\nall-reduce together here", "left"),
                        (ALIGN_OUT_MS, "all 8 all-reduces return\nwithin 2 us of each other", "right")):
        ax.axvline(x, color="0.25", linestyle="--", linewidth=1.2, zorder=5)
        ax.text(x + (0.10 if ha == "left" else -0.10), -1.05, text,
                fontsize=9, color="0.25", ha=ha, va="center", zorder=6)

    for lo, hi in gaps:
        ax.text((lo + hi) / 2, SLOW_RANK, f"{hi - lo:.2f}\nms", fontsize=8.5,
                color="tab:red", ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85))

    ax.text(waiter_ar["start"] + waiter_ar["dur"] / 2, 4.0,
            f"the other 7 ranks sit inside the collective for "
            f"{waiter_ar['dur']:.2f} ms, waiting for a peer that has not arrived",
            fontsize=10.5, color="white", ha="center", va="center",
            fontweight="bold", zorder=6)
    ax.set_title("What the \"before\" trace looks like with all eight ranks aligned\n"
                 "gpt-oss-120B W4A8, TP8 prefill, stock-Triton baseline, "
                 "iteration 8, layer 1, phase 0\n"
                 f"R0 is GPU-idle for {' + '.join(f'{hi - lo:.2f}' for lo, hi in gaps)}"
                 f" = {total_gap:.2f} ms of host gap, then clears the collective in "
                 f"{slow_ar['dur'] * 1000:.0f} us",
                 fontsize=12)
    ax.set_xlabel("Time within the clipped window (ms)")

    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white")
               for c, _ in CATEGORY_STYLE.values()]
    handles.append(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="tab:red",
                             hatch="///"))
    labels = [l for _, l in CATEGORY_STYLE.values()] + ["GPU idle on R0 (host gap)"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.045),
               ncol=6, frameon=False, fontsize=9, handlelength=1.6)

    # --- zoom on the arrival ------------------------------------------------
    draw_gantt(axz, by_rank, ZOOM, gap_hatch=False, headroom=1.45)
    axz.axvline(ALIGN_OUT_MS, color="0.25", linestyle="--", linewidth=1.2, zorder=5)
    axz.set_title(f"Zoom on the arrival: {ZOOM[0]:.2f} to {ZOOM[1]:.2f} ms",
                  fontsize=11, pad=26)
    axz.set_xlabel("Time within the clipped window (ms)")
    axz.annotate(
        f"R0 still running MoE work; it enters\nthe collective at "
        f"{slow_ar['start']:.2f} ms, {waiter_ar['dur'] - slow_ar['dur']:.1f} ms "
        f"after its peers",
        xy=(slow_ar["start"], SLOW_RANK - 0.30), xytext=(8.575, -1.02),
        fontsize=9, color="tab:red", ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="tab:red"), zorder=7,
    )
    axz.text(ALIGN_OUT_MS + 0.012, -1.02,
             "all 8 return here; everything to the\nright is the next layer's work",
             fontsize=9, color="0.25", ha="left", va="center", zorder=7)

    fig.savefig(args.output_dir / "prefill-before-trace.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output_dir / 'prefill-before-trace.png'}")
    print(f"  R0 idle gaps: {[f'{lo:.2f}-{hi:.2f}' for lo, hi in gaps]} "
          f"total {total_gap:.2f} ms")


if __name__ == "__main__":
    main()
