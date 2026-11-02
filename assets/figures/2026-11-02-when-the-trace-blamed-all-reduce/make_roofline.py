#!/usr/bin/env python3
"""Generate the gpt-oss-120B W4A8 MoE Roofline figure for MI355X.

The input YAML must contain one mapping per measured kernel shape with at least:
  M: int
  N: int
  K: int
  best_us: float

The measured y-axis values come from best_us. The x-axis arithmetic intensity is
modelled, not measured HBM traffic. By default, this script reproduces the
assumptions used by the supplied draft figure:

  * M is the number of pre-routing tokens in the step.
  * top_k = 4, so routed assignments = M * 4.
  * active experts = min(total_experts, routed assignments), an upper-bound
    distinct-expert assumption.
  * MXFP4 weights cost 0.5 B/element plus one E8M0 scale byte per 32 weights.
  * FP8 activations cost 1 B/element; BF16 outputs cost 2 B/element.
  * Each active expert's weights are read once.
  * The operative W4A8 compute roof is the FP8 MFMA rate.

Publication checks:
  1. Confirm whether M in the benchmark YAML means pre-routing tokens or routed
     rows. If it already means routed rows, pass --m-semantics assignments.
  2. Publish the exact YAML/CSV input with the article.
  3. Treat the 8 TB/s and 5 PFLOP/s lines as specification roofs unless measured
     sustainable roofs are supplied separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


DEFAULT_SHAPES = {
    (5760, 2880): ("MoE gate/up (N=5760, K=2880)", "o", "#c1272d"),
    (2880, 2880): ("MoE down (N=2880, K=2880)", "s", "#1f5fa8"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="YAML file containing M, N, K, and best_us rows")
    parser.add_argument("--output", type=Path, default=Path("fig_roofline.png"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument(
        "--m-semantics",
        choices=("tokens", "assignments"),
        default="tokens",
        help="Whether each row's M is pre-routing tokens or already routed rows",
    )
    parser.add_argument(
        "--active-expert-model",
        choices=("upper-bound", "uniform-expected"),
        default="upper-bound",
        help=("upper-bound reproduces min(E, assignments); uniform-expected "
              "uses the expected number of unique experts under uniform routing"),
    )
    parser.add_argument("--hbm-tb-s", type=float, default=8.0)
    parser.add_argument("--peak-fp8-pf", type=float, default=5.0)
    parser.add_argument("--peak-fp4-pf", type=float, default=10.1)
    parser.add_argument("--peak-bf16-pf", type=float, default=2.5)
    parser.add_argument("--hide-context-roofs", action="store_true",
                        help="Hide FP4 and BF16 context lines")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input YAML not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, list):
        raise ValueError("Input YAML must contain a list of mappings")

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} is not a mapping")
        missing = {"M", "N", "K", "best_us"} - row.keys()
        if missing:
            raise ValueError(f"Row {index} is missing fields: {sorted(missing)}")
        normalized = {
            "M": int(row["M"]),
            "N": int(row["N"]),
            "K": int(row["K"]),
            "best_us": float(row["best_us"]),
        }
        if min(normalized["M"], normalized["N"], normalized["K"]) <= 0:
            raise ValueError(f"Row {index} has a non-positive dimension")
        if normalized["best_us"] <= 0:
            raise ValueError(f"Row {index} has non-positive best_us")
        rows.append(normalized)
    return rows


def routed_assignments(m: int, top_k: int, semantics: str) -> int:
    return m * top_k if semantics == "tokens" else m


def active_expert_count(assignments: int, experts: int, model: str) -> float:
    if model == "upper-bound":
        return float(min(experts, assignments))
    # Expected number of occupied bins for independent uniform routing.
    return experts * (1.0 - ((experts - 1.0) / experts) ** assignments)


def metrics(
    m: int,
    n: int,
    k: int,
    best_us: float,
    *,
    top_k: int,
    experts: int,
    m_semantics: str,
    active_expert_model: str,
) -> tuple[float, float]:
    assignments = routed_assignments(m, top_k, m_semantics)
    flops = 2.0 * assignments * n * k
    active_experts = active_expert_count(assignments, experts, active_expert_model)

    weight_bytes = active_experts * n * k * (0.5 + 1.0 / 32.0)
    activation_bytes = assignments * k * 1.0
    output_bytes = assignments * n * 2.0

    arithmetic_intensity = flops / (weight_bytes + activation_bytes + output_bytes)
    achieved_flops_s = flops / (best_us * 1e-6)
    return arithmetic_intensity, achieved_flops_s


def matching_rows(rows: Iterable[dict[str, Any]], n: int, k: int) -> list[dict[str, Any]]:
    return sorted((row for row in rows if row["N"] == n and row["K"] == k),
                  key=lambda row: row["M"])


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)

    hbm_bw = args.hbm_tb_s * 1e12
    peak_fp8 = args.peak_fp8_pf * 1e15
    peak_fp4 = args.peak_fp4_pf * 1e15
    peak_bf16 = args.peak_bf16_pf * 1e15
    ridge = peak_fp8 / hbm_bw

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    ai_axis = np.logspace(-0.5, 4.0, 400)
    ax.plot(ai_axis, hbm_bw * ai_axis / 1e12, color="#444444", linewidth=2.0,
            label=f"HBM3E specification roof ({args.hbm_tb_s:g} TB/s)")
    ax.axhline(peak_fp8 / 1e12, color="#c1272d", linewidth=1.6,
               label=f"Operative FP8/W4A8 specification roof ({args.peak_fp8_pf:g} PF)")
    if not args.hide_context_roofs:
        ax.axhline(peak_fp4 / 1e12, color="#888888", linestyle="--", linewidth=1.4,
                   label=f"FP4 context roof ({args.peak_fp4_pf:g} PF)")
        ax.axhline(peak_bf16 / 1e12, color="#888888", linestyle=":", linewidth=1.4,
                   label=f"BF16 context roof ({args.peak_bf16_pf:g} PF)")

    ax.axvline(ridge, color="#c1272d", linestyle=(0, (1, 3)), linewidth=1.0, alpha=0.7)
    ax.annotate(f"specification ridge\nAI={ridge:.0f} FLOP/B",
                (ridge, 3.2), fontsize=8, ha="center", va="bottom")

    plotted = 0
    for (n, k), (label, marker, color) in DEFAULT_SHAPES.items():
        shape_rows = matching_rows(rows, n, k)
        if not shape_rows:
            continue
        points = []
        for row in shape_rows:
            ai, achieved = metrics(
                row["M"], n, k, row["best_us"],
                top_k=args.top_k,
                experts=args.experts,
                m_semantics=args.m_semantics,
                active_expert_model=args.active_expert_model,
            )
            points.append((row["M"], ai, achieved))

        ax.plot([point[1] for point in points],
                [point[2] / 1e12 for point in points],
                marker=marker, color=color, linewidth=1.4, markersize=6,
                label=label, zorder=5)
        plotted += 1

        if (n, k) == (5760, 2880):
            for m, x, y in points:
                if m == 16:
                    ax.annotate("M=16\ndecode regime",
                                (x, y / 1e12), fontsize=8,
                                xytext=(10, -4), textcoords="offset points")
                elif m == 8192:
                    ax.annotate("M=8192\nprefill regime",
                                (x, y / 1e12), fontsize=8, ha="right",
                                xytext=(-8, 6), textcoords="offset points")

    if plotted == 0:
        raise ValueError("No rows matched the expected gate/up or down shapes")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2, 2e4)
    ax.set_ylim(8, 1.2e4)
    ax.set_xlabel("Modelled arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Achieved throughput from measured best_us (TFLOP/s)")
    ax.set_title("gpt-oss-120B W4A8 MoE GEMM Roofline — MI355X\n"
                 "TP1 kernel sweep: specification roofs with measured kernel points")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.output}")

    print("\nKey points for gate/up (N=5760, K=2880):")
    for row in matching_rows(rows, 5760, 2880):
        if row["M"] not in (16, 64, 8192):
            continue
        ai, achieved = metrics(
            row["M"], 5760, 2880, row["best_us"],
            top_k=args.top_k,
            experts=args.experts,
            m_semantics=args.m_semantics,
            active_expert_model=args.active_expert_model,
        )
        roof = hbm_bw * ai if ai < ridge else peak_fp8
        roof_name = "memory" if ai < ridge else "FP8/W4A8 compute"
        print(
            f"  M={row['M']:>5}: AI={ai:8.1f} FLOP/B, "
            f"achieved={achieved/1e12:7.1f} TFLOP/s, "
            f"{100.0*achieved/roof:4.1f}% of {roof_name} specification roof, "
            f"best_us={row['best_us']:.1f}"
        )


if __name__ == "__main__":
    main()
