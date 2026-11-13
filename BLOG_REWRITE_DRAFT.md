---
layout: post
title: "When the Trace Blamed All-Reduce: A Bound-Driven vLLM Investigation on AMD MI355X"
date: '2026-11-02'
author: AMD
summary: "How profiling, communication bounds, and Roofline analysis exposed host-side stalls, then guided HIP graph and W4A8 MoE optimizations for vLLM on AMD Instinct MI355X."
image: /assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/fig_roofline.png
math: true
tags:
  - performance
  - hardware
  - rocm
  - moe
---

A TP8 `rocprof` trace suggested that QuickReduce was taking about half of the prefill time. This made all-reduce look like the main bottleneck.

Before changing the collective, we checked how long the data transfer should take. The all-reduce moved about 39 MB of estimated wire traffic, which should take roughly 0.16-0.26 ms at the measured bandwidth. However, the trace showed a 7.81 ms delay between ranks. Most of that time was therefore spent waiting, not transferring data.

Repeated runs confirmed that the problem was outside GPU computation. GPU compute time varied by less than 0.7%, but throughput dropped from 53.3 to 42.9 requests/s. Incomplete HIP graph capture and CPU contention created GPU idle bubbles lasting 130-160 ms.

We replaced unstable kernel paths with AITER kernels and captured more serving shapes in HIP graphs. This removed the large host stalls. After re-profiling, the main GPU bottleneck was the W4A8 grouped MoE GEMM. Roofline analysis showed that decode was limited mainly by HBM bandwidth, while prefill had more room to improve compute efficiency.

Across the matched benchmark cells, the complete set of changes improved geometric-mean throughput by 1.63x for decode-heavy traffic and 1.53x for prefill-heavy traffic. The best individual cells improved by 2.3-2.4x.

The main lesson is a workflow that developers can reuse:

> **Profile -> Bound -> Evaluate -> Fix -> Re-profile**

A long all-reduce range does not prove that all-reduce is the bottleneck. It may include peer waits, host dispatch delays, or skew from earlier kernels—not just data transfer.

[`gpt-oss-120b`](https://openai.com/index/introducing-gpt-oss/) shows this distinction clearly. The model has 36 layers, 128 experts, and activates four experts per token. The AMD checkpoint used here stores MoE weights in MXFP4 and uses FP8 activations, which we refer to as W4A8. In serving, this creates two different operating regimes:

- **Decode:** Each expert receives few tokens. Large expert weights are streamed from HBM with little reuse, so arithmetic intensity is low and the expert GEMM is memory-bandwidth-bound.
- **Prefill:** Many tokens reuse each weight tile. Arithmetic intensity rises, and the same expert GEMM can become compute-bound.
- **Tensor parallelism:** Each layer also contains blocking collectives. Decode executes about 73 small all-reduces per token step, making synchronization latency visible even when the payload is tiny.
- **Higher TP:** Sharding reduces the per-GPU contraction dimension. That can make GEMMs less efficient, increase padding overhead, and change which memory layout wins.

A single trace therefore exposes host scheduling, collectives, memory bandwidth, matrix throughput, padding, and kernel dispatch.

<p align="center"><img src="/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/hero-trace-vs-bound.png" alt="Left: per-rank all-reduce kernel duration at one TP8 prefill all-reduce, showing seven ranks at about 8 ms and rank 0 at 0.17 ms. Right: the same duration compared on a log axis against bytes-on-the-wire, cross-rank wait, and the host-side stall" width="80%"></p>

*The profiler correctly shows that these ranks spend about 8 ms inside the all-reduce. The left panel shows one measured all-reduce site (TP8 prefill, iteration 8, layer 1, phase 0, stock-Triton baseline). Seven ranks arrive early and block, while rank 0 finishes upstream host work and passes through in 0.17 ms. The right panel compares that duration with three possible explanations. Only the green band represents data movement, and it is roughly two orders of magnitude smaller than the other two. The panel combines three types of data. The wire band is an analytic bound for 39 MB of two-pass BF16 traffic at 150-240 GB/s effective, plus one 0.22 ms measurement taken after all eight ranks arrived. The blue band is the measured cross-rank spread at this site. The red band is the separately observed `_fused_add_rmsnorm_pad` host stall, which affects all eight ranks instead of staggering them. The stalled rank and iteration vary across runs, but the host stall does not.*

The generation script and its input data are included with the article: [`make_hero_attribution.py`](/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/make_hero_attribution.py) and [`iter8_l1p0_per_rank_ar.csv`](/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/iter8_l1p0_per_rank_ar.csv).


## Experimental settings

In our experiments, we used a single node with eight [AMD Instinct MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html) GPUs. The benchmark covered TP1, TP2, TP4, and TP8 at concurrency levels from 4 to 256, with ISL/OSL as 1K/1K and 1K/8K. Regarding profiling, we collected full `rocprof` traces at TP4 and TP8, including per-rank GPU and host timelines. We report cross-platform B200 numbers only in the appendix and treat them as a dated software snapshot, not as a general statement about either platform. 

### Roofline analysis models

We use three models to set performance bounds:

- **Kernel and HBM Roofline:** Compare a GPU kernel's arithmetic intensity with the compute and memory-bandwidth limits. For the W4A8 expert GEMM, the MI355X specification roofs are 5.0 PFLOP/s and 8 TB/s, giving a ridge point of about 625 FLOP/byte.
- **Collective latency and bandwidth:** Separate startup and synchronization latency, data-transfer time, and time spent waiting for other ranks or host progress. Bandwidth matters more for large prefill messages, while latency dominates tiny decode messages.
- **Host timeline and control experiments:** Diagnose host dispatch gaps by comparing GPU time with end-to-end time, repeated runs, captured and eager paths, CPU placement, and per-rank arrival times.

<p align="center"><img src="/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/workflow.png" alt="Flow diagram of the five-step loop Measure, Bound, Falsify, Fix, Re-profile, with the Bound step expanding into three bounding models: kernel Roofline, collective latency-bandwidth, and host timeline with control experiments" width="80%"></p>

*The Bound step uses three models, each for a different mechanism. The Roofline model applies to kernels, the latency-bandwidth model applies to collectives, and host dispatch stalls require controlled comparisons because they have no useful GPU ceiling. The collective model was critical here because the profiler range includes wait time. A long collective range therefore does not show that the interconnect was busy for the entire interval. After one bottleneck is fixed, re-profiling reveals the next mechanism and the model needed to analyze it.*

## When the profiling trace blamed all-reduce

This figure shows the initial hypothesis that the later analysis disproved. The mismatch was clearer in our first TP8 prefill traces. The collective ranges accounted for roughly half of the profiled step, and the slowest rank spent about twice as long there as the fastest, corresponding to approximately 61% cross-rank imbalance.

We nearly started rewriting the all-reduce. Before doing that, we estimated the expected transfer time from the payload size and bandwidth.

<p align="center"><img src="/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/fig_bottleneck_reframed.png" alt="Initial TP8 decode attribution: 174 us in the all-reduce range and 80 us in other GPU compute" width="80%"></p>

*Initial TP8 decode attribution. The all-reduce segment is the duration of the profiler range, including synchronization wait; it is not a decomposition of wire time. In a clean run, collectives accounted for about 14% of decode kernel time.*


## Bound the transfer component

For a 4096-token prefill chunk, the all-reduce payload is $$4096 \times 2880 \times 2\ \text{bytes} \approx 22.5\ \text{MB}.$$ At TP8, a two-pass all-reduce moves approximately $$2\frac{p-1}{p} = 2\frac{7}{8} = 1.75$$ times the payload, or about 39 MB of estimated wire traffic. A clean two-pass collective on this system sustained roughly 150-240 GB/s of effective bandwidth. At that rate, the byte-transfer component is approximately $$\frac{39\ \text{MB}}{150\text{-}240\ \text{GB/s}} \approx 0.16\text{-}0.26\ \text{ms}.$$

The profiled cross-rank stagger was 7.81 ms. Even using the conservative end of the bandwidth range, the stagger was about 30 times larger than the transfer component. Startup latency matters, but it does not turn 39 MB into 7.81 ms of link activity on a path that reaches much higher bandwidth in a clean run.

The calculation supports a specific conclusion:

> **For this payload and this execution window, bandwidth could not explain the observed rank spread. The collective was spending most of the interval waiting, not transferring.**

## Evaluate the first explanation

We then ran the same serving command twice, back-to-back, and decomposed each iteration. The results changed the diagnosis:
- Per-iteration GPU compute repeated within **0.7%**.
- End-to-end throughput changed by about **24%**, from **53.3 to 42.9 requests/s**.
- The slow run contained **130-160 ms host gaps** that affected all eight ranks at once.
- The gaps were centered near `_fused_add_rmsnorm_pad`, not inside a long-running data-transfer kernel.

On a machine without a noisy neighboring workload, the same configuration showed none of the catastrophic stalls. The worst host gap was 2.7 ms and the median was 32 us.

Stable GPU time and unstable wall-clock time indicate that the missing time is outside GPU execution. The per-rank timeline showed how the host delay looked like a communication problem: one rank arrived late at a blocking all-reduce, while the other ranks waited inside the collective range.

## Fix: incomplete HIP graph coverage plus CPU contention

The serving path used a fixed graph-capture window. Some shapes were captured into a HIP graph, while other shapes fell back to eager, per-kernel dispatch. When a shape took the eager path, the CPU had to submit a longer launch chain. A contended host could interrupt that chain, leaving the GPUs idle.

The blocking collective exposed the arrival-time skew. The profiler attribution was correct—time elapsed inside the all-reduce range—but our first explanation of the cause was wrong.

The fix had two parts:
1. **Use stable AITER paths for the affected operations.** Replacing the unstable Triton paths with AITER kernels reduced execution variation and made graph capture more consistent.
2. **Widen graph capture and compile more serving shapes.** The `widegraph` configuration captured a larger region and reduced eager fallbacks.

What changed after the fix:
- The AITER kernel-path change reduced bottleneck-rank time getting a **1.54x** improvement.
- The sample iteration's cross-rank stagger reduced about **460x**.
- Wider graph capture reduced the remaining communication-imbalance band from **22.5% to 6-12%**.

The 460x reduction in rank stagger after changing the kernel path and graph capture shows that the original interval was not bandwidth-bound. No interconnect optimization can make a truly bandwidth-limited 39 MB transfer 460 times faster. The stage ladder below uses **cumulative bottleneck-rank time over the selected profiled window**. It does not decompose collective transfer time or represent an additive end-to-end throughput ablation.

<p align="center"><img src="/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/fig_stage_ladder_reframed.png" alt="TP8 prefill stage ladder from the baseline through the AITER kernel path and wider HIP graph capture" width="80%"></p>

*The AITER kernel-path change produces the large reduction in cumulative bottleneck-rank time, from 6.11 s to 3.97 s. Wider HIP graph capture changes total GPU time only slightly in this window, but removes the catastrophic host-side tail stalls and tightens cross-rank consistency. The percentages are the study's cross-rank imbalance summaries for each stage.*


## Re-profile: the real bottleneck was W4A8 MoE

Once the host-side noise was removed, the largest GPU block became clear: the grouped expert GEMMs in the W4A8 MoE path. We then used a conventional compute/HBM roofline model to analyze the kernel.

### One kernel, two operating regimes

The figure below combines **measured tuned-kernel times** with **modelled arithmetic intensity**. The y-axis comes from the best measured `best_us` value for each TensorAtlas W4A8 MoE-GEMM shape. The x-axis is calculated by the supplied script rather than measured from HBM counters.

The plot is a **specification Roofline with measured kernel points**; it does not measure HBM traffic directly. At low `M`, routing collisions could activate fewer experts than the upper-bound model assumes, shifting the point to the right. The decode conclusion still holds because the point remains far below the 625 FLOP/byte specification ridge, but the exact arithmetic-intensity value is not a hardware-counter measurement. Because the result is close to the bandwidth roof, faster arithmetic alone cannot create a large gain. Decode optimization should focus on reducing bytes, improving memory access, avoiding padding, and removing latency around the GEMM.

<p align="center"><img src="/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/fig_roofline.png" alt="W4A8 MoE GEMM specification Roofline with measured TP1 tuned-kernel points" width="80%"></p>

*W4A8 MoE GEMM Roofline for the full-model TP1 gate/up and down shapes. The sloped 8 TB/s line and horizontal 5.0 PFLOP/s line are MI355X specification roofs; the red and blue points use measured tuned AITER kernel times. The gray FP4 and BF16 lines are context only. The x-axis follows the byte model described above.*

At high `M`, representative of prefill modelled arithmetic intensity rose to approximately **680 FLOP/byte**. In this model, the point sits just beyond the 625 FLOP/byte specification ridge. the representative of prefill measured throughput was **1.19 PFLOP/s**, about **24% of the 5.0 PFLOP/s specification compute roof**.

This does not prove that every production prefill shape is unambiguously compute-bound: routing balance, padding, metadata, and TP-specific short-K shapes can move the operating point. It does show that the large-M TP1 kernel sweep has substantially more compute headroom than the decode regime. Tile shape, pipeline stages, occupancy, scale handling, and short-K behavior are therefore worth tuning.

### Optimization 1: tune grouped GEMM by serving shape

The original dispatch did not consistently select efficient tile sizes and pipeline stages for the gpt-oss expert shapes on gfx950. We added per-shape configurations and hardened the fallback path in AITER. The AITER changes added tuned `(block_m, N, K)` entries for the relevant gfx950 shapes and improved fallback selection so an untuned shape did not silently borrow a configuration that was unsafe or inefficient for CDNA4 scale handling.

### Optimization 2: fix output allocation before measuring speed

The same AITER work also fixed a correctness issue. The kernel allocated output storage using the logical, unpadded N while computing tiles over padded N. When `block_n` did not divide the logical dimension, the last tile could write beyond the allocated buffer. The fix sizes the output buffer to padded N while preserving the logical output shape presented to the caller. Correctness constrains performance tuning: a kernel configuration is not a valid optimization if it relies on undefined memory behavior. We do not count the allocation fix as an isolated performance gain.

### Optimization 3: gate MX-scale swizzling by tensor parallelism

CDNA4 can use a swizzled MX-scale layout that is efficient for some shapes. Swizzling, however, imposes block-size and padding constraints. Tensor parallelism shrinks the per-GPU contraction dimension:
- **TP4:** K = 720
- **TP8:** K = 360

At those short K values, padding to the block size required by the swizzled path can cost more than the layout saves. After retuning both paths, the plain strided layout won **23 of 28** TP4/TP8 benchmark cells and improved selected decode-heavy 1K/1K cells by as much as **19%**. The swizzled path remained useful for TP1/TP2 and in a small high-concurrency prefill corner.

The vLLM gate therefore selects:

- swizzle **on** for TP1 and TP2;
- strided layout for TP4 and TP8.

Both the weight-load layout and kernel argument use the same `should_use_cdna4_mx_scale_swizzle()` helper. Centralizing the decision is important because a mismatch can silently corrupt the scale tensor.

<p align="center"><img src="/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/swizzle-gate-heatmap.png" alt="Heatmap of swizzled versus strided MX-scale layout across 28 TP4 and TP8 serving cells, with a diagram of how the BLOCK_K constraint inflates the contraction dimension" width="80%"></p>

*Each cell is a measured A/B test between the two retuned layouts. Tests used the same container image, with the vLLM torch-compile, Triton, and inductor caches purged before each cell. Treating results within 2% as ties, the strided layout wins 23 cells, the swizzled layout wins 3, and 2 are ties. Swizzling trails in all 14 decode-heavy 1K/1K cells, by as much as 18.9%. It retains an advantage only for prefill-heavy 8K/1K traffic at 128 and 256 concurrent requests.*

*The lower panel derives the mechanism from the two tuned dispatch JSONs. Both paths were retuned before the comparison, so this is a layout A/B at matched tuning quality, not a tuned-versus-untuned artefact. The swizzled MX-scale layout makes scale loads more efficient but requires `BLOCK_K` of at least 256. For the decode down-projection, which contracts over K = 2880/TP, that constraint is expensive: at TP8, K = 360, and the tuned swizzled entry steps a 512-element extent against the strided entry's 384, so 152 elements of every pass are masked against 24. Masked elements are work the GPU performs and discards, and here that discarded compute outweighs the scale-load saving. At TP4 both paths land on 768 and mask the same 48 elements; there the layouts differ only in the K step, 256 against 128. The gate follows the measurement: strided at TP4 and TP8, swizzled at TP1 and TP2, where it wins.*

The generation script and its input data are included with the article: [`make_swizzle_gate.py`](/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/make_swizzle_gate.py) and [`swizzle_ab_28cell.csv`](/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/swizzle_ab_28cell.csv).


## What changed in vLLM and AITER

The gains came from multiple layers working together.

| Layer | Change | Purpose |
|---|---|---|
| vLLM runtime | Wider HIP graph capture and more compiled serving shapes | Reduce eager fallback and host launch gaps |
| vLLM dispatch | TP-aware CDNA4 MX-scale swizzle gate | Avoid padding overhead at TP4/TP8 and keep layout decisions consistent |
| AITER kernel path | Stable replacements for affected Triton operations | Reduce variation and improve captured execution |
| AITER grouped GEMM | Per-shape gfx950 W4A8 tuning and hardened fallback | Improve hot expert GEMM shapes and avoid unsafe configuration reuse |
| AITER correctness | Allocate W4A8 MoE output for padded N | Prevent out-of-bounds writes on tail tiles |
| Benchmark operations | CPU isolation/pinning and repeated runs | Separate host noise from GPU execution |

The upstream changes include:

- [AITER PR #3428: W4A8 tuned dispatch, safe fallback, and padded-N output allocation](https://github.com/ROCm/aiter/pull/3428)
- [AITER PR #3580: additional gpt-oss shapes and fallback hardening](https://github.com/ROCm/aiter/pull/3580)
- [vLLM PR #44804: hybrid CDNA4 swizzle gate](https://github.com/vllm-project/vllm/pull/44804)

### Ablation: match each fix to its metric

The optimizations interact, so their gains should not be added. Each change is evaluated with its most relevant metric.

| Stage | Evidence | Observed effect |
|---|---|---|
| Baseline trace | TP8 prefill | 6.11 s bottleneck-rank time; 7.81 ms sample rank stagger; catastrophic host gaps in noisy runs |
| Stable AITER path | Same profiled window | 3.97 s bottleneck-rank time; 0.017 ms sample rank stagger |
| Wider graph capture | Repeated TP8 runs | Residual imbalance reduced to 6-12%; 130-160 ms stalls removed |
| Grouped-GEMM tuning | Kernel dispatch and hot shapes | Better gfx950 tile/stage choices; quantifying the standalone full-sweep contribution requires raw ablation data |
| Swizzle gate | 28 TP4/TP8 cells | Strided layout won 23 cells; up to 19% on selected 1K/1K cells |
| Full stack | 56-cell serving sweep | 1.63x decode-heavy geomean; 1.53x prefill-heavy geomean; up to 2.3-2.4x on individual cells |

<p align="center"><img src="/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/ablation.png" alt="Ablation of the gpt-oss-120B W4A8 MI355X stack, split into three panels by metric: a cumulative profiled-window waterfall, separately measured changes as cards, and the end-to-end 32-cell throughput sweep" width="80%"></p>

*The panels use three different metrics; do not multiply or add their values. **Left:** cumulative bottleneck-rank time in the same profiled TP8 prefill window, with each step applied after the previous one. **Right:** changes measured independently in their own configurations. Cards distinguish these results from the cumulative bars. **Bottom:** the only end-to-end measurement, comparing total token throughput with the previously published MI355X-vLLM results for the 32 of 56 sweep cells that have a published counterpart. The outlined bar marks the cell closest to the decode-heavy geometric mean. No single cell was measured as a per-stage serving-throughput ladder, so the left panel reports profiled-window time rather than tokens per second.*

The generation script and its input data are included with the article: [`make_ablation.py`](/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/make_ablation.py) and [`ablation_fullsweep.csv`](/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/ablation_fullsweep.csv).

---

## End-to-end results

We evaluated TP1, TP2, TP4, and TP8 over concurrency 4-256 for two request shapes, producing 56 matched cells. Every run started from a cold cache with matched input and output lengths.

Against the previously published MI355X-vLLM results on the same serving harness, the optimized stack delivered:

- **1.63x geometric-mean throughput improvement** on decode-heavy 1K/1K traffic;
- **1.53x geometric-mean throughput improvement** on prefill-heavy 8K/1K traffic;
- **2.3-2.4x** on the best individual cells.

These are full-stack results. They combine more complete graph capture, stable AITER paths, grouped-GEMM tuning, correctness hardening, and TP-aware layout selection.

<p align="center"><img src="/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/fig_boost.png" alt="Final full-stack throughput speedup over the previously published MI355X-vLLM baseline" width="80%"></p>

*Final full-stack speedup over the previously published MI355X-vLLM snapshot on matched cells. Dark bars show the geometric mean across cells; light bars show the best individual cell. This chart summarizes the outcome of the complete stack and does not attribute the gain to individual changes.*


### Accuracy

Under the same evaluation configuration, the optimized serving path showed no accuracy regression relative to the unoptimized path.

We publish the evaluation manifest instead of only a headline score. Reproducing a GSM8K result requires the evaluator, metric definition, decoding parameters, and sample count:

| Item | Value |
|---|---|
| Harness | `lm-evaluation-harness` 0.4.9.2 |
| Task | `gsm8k`, task version 3.0, 5-shot, 1319 of 1319 samples |
| Interface | `local-chat-completions` against the served OpenAI-compatible endpoint |
| Decoding | `temperature=0`, `top_p=1`, `max_tokens=1152` |
| Metrics | `exact_match` reported under both `strict-match` and `flexible-extract` |

A representative run on the optimized path scores **0.961 strict-match** and **0.961 flexible-extract**, each with a standard error of 0.005. Across repeated runs, scores range from 0.951 to 0.965. Because that spread is comparable to the standard error, we treat differences within the range as run-to-run noise.

The two metrics agree within 0.001 here. For models that answer in prose, however, `strict-match` can fall sharply while `flexible-extract` remains high. A GSM8K score is therefore ambiguous unless its filter is specified.

We do not use this score as a headline result. It is a regression guard on the serving-path changes, not a claim about the model's reasoning quality.

---

## What generalizes, and what does not

| General method | Case-specific detail |
|---|---|
| Check profiler attribution against a physical bound | 8 TB/s HBM and 5 PFLOP/s W4A8 roof on MI355X |
| Use an alpha-beta model for collectives | QuickReduce thresholds and TP8 two-stage behavior |
| Compare GPU time with end-to-end time | `_fused_add_rmsnorm_pad` and this graph-capture window |
| Repeat identical runs under controlled CPU placement | The particular noisy-neighbor event in this system |
| Re-profile after every major fix | The exact sequence: communication -> graph capture -> AITER -> MoE GEMM -> swizzle |
| Treat prefill and decode as different operating regimes | gpt-oss top-4, 128-expert W4A8 shapes |
| Re-evaluate kernel layouts when TP changes shape | K=720 at TP4 and K=360 at TP8 |
| Separate correctness fixes from performance claims | The padded-N output allocation bug |

---

## Serving recipe

The following command reproduces the optimized serving path used for these measurements. Because nightly images change, pin the image digest and source commit SHAs when publishing results.

```bash
# vLLM 0.23.1-dev + AITER 0.1.16 + Triton 3.6.0 used for these measurements
IMAGE=vllm/vllm-openai-rocm:nightly-34b560b725b401f33477464163244b4744d1c40f
MODEL=amd/gpt-oss-120b-w-mxfp4-a-fp8

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_TRITON_ROPE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export AMDGCN_USE_BUFFER_OPS=0

vllm serve "$MODEL" \
  --tensor-parallel-size 8 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN \
  --compilation-config '{"pass_config": {"fuse_rope_kvcache": true}}' \
  --block-size 64 \
  --gpu-memory-utilization 0.95 \
  --no-enable-prefix-caching \
  --reasoning-parser openai_gptoss
```

`--reasoning-parser openai_gptoss` is required for the vLLM 0.23.x configuration used here. Without the appropriate gpt-oss parser, Harmony-channel output may not appear in the expected chat-completions content field, invalidating an accuracy test.

The TP-aware CDNA4 MX-scale layout decision is internal. No user flag is required: TP1/TP2 select the swizzled path, while TP4/TP8 select the strided path.

`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` enables the quantized collective path only when its dispatch conditions are met. It does not reduce every payload. In the profiled 4096-token TP8 prefill case, the threshold condition did not select the INT4 path.

---

## Conclusion

The first trace told a convincing story: all-reduce appeared to dominate, so communication looked like the bottleneck. A simple bound showed that the bytes could not account for the observed delay. A control experiment then located the missing time on the host. Fixing HIP graph coverage and unstable execution paths removed the false bottleneck and exposed the real one: W4A8 grouped MoE GEMMs that were memory-bound in decode and below the compute roof in prefill.

The sequence matters more than any single kernel change.

- **Profiler ranges tell us where time was attributed.**
- **Performance bounds tell us whether that attribution can explain the runtime.**
- **Controlled experiments establish the cause.**
- **Re-profiling reveals the next bottleneck.**

The combined vLLM and AITER changes improved the MI355X serving sweep by 1.63x geometric mean on decode-heavy traffic and 1.53x on prefill-heavy traffic, with all changes upstreamed or represented by upstream work. More importantly, the investigation produced a reusable method for the next model, the next kernel, and the next accelerator:

> **Profile. Bound. Evaluate. Fix. Re-profile.**

---

## Appendix A: Dated cross-platform snapshot

For context only, we place the optimized MI355X results beside dated B200 vLLM and TensorRT-LLM snapshots of the same model. These are not matched runs: the snapshots were collected separately, on different hardware, under an older software stack, as the versions below show. Read them as a dated reference point rather than a controlled comparison.

On decode-heavy 1K/1K traffic, the reported peak per-GPU throughputs were:

- MI355X: **36.5k token/s/GPU**
- B200 with vLLM: **18.3k token/s/GPU**
- B200 with TensorRT-LLM: **28.0k token/s/GPU**

On prefill-heavy 8K/1K traffic:

- MI355X: **50.1k token/s/GPU**
- B200 with vLLM: **51.5k token/s/GPU**
- MI355X was **0.81x** the reported B200 TensorRT-LLM result.

These numbers come from the [InferenceX dashboard](https://inferencex.semianalysis.com/) configurations `gptoss-fp4-mi355x-vllm`, `gptoss-fp4-b200-vllm`, and `gptoss-fp4-b200-trt`. They used vLLM 0.22.0 and TensorRT-LLM 1.3.0rc14 and were last published on 2026-07-06.

<p align="center"><img src="/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/pareto_combined_1K1K.png" alt="Dated cross-platform throughput-per-GPU versus interactivity snapshot for 1K input and 1K output" width="80%"></p>

*Decode-heavy 1K/1K snapshot. The connected lines show the reported nondominated frontier; faint points show the remaining measured cells. The small numerals identify tensor-parallel degree. Interactivity is expressed as tokens/s/user, derived from median TPOT. Per-GPU normalization is useful for this plot but is not a complete node-cost comparison.*

<p align="center"><img src="/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/pareto_combined_8K1K.png" alt="Dated cross-platform throughput-per-GPU versus interactivity snapshot for 8K input and 1K output" width="80%"></p>

*Prefill-heavy 8K/1K snapshot, shown with the same conventions. These figures belong in the appendix because their interpretation depends on the exact collection date, framework versions, host setup, TP degree, and benchmark harness.*

---

## Acknowledgements

We would like to thank everyone who contributed to this collaboration,
including Xiaohu Guo, Chang Liu, Chun Fan, Rohan Potdar, Lixun Zhang, Peng Sun, and Andy
Luo.

## Resources

- [vLLM](https://github.com/vllm-project/vllm)
- [AMD AITER](https://github.com/ROCm/aiter)
- [Roofline figure generator](/assets/figures/2026-11-02-when-the-trace-blamed-all-reduce/make_roofline.py)
- [InferenceX dashboard](https://inferencex.semianalysis.com/) (source of the cross-platform snapshot in Appendix A)
- [AMD Instinct MI355X specifications](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)
- [AMD W4A8 gpt-oss-120b checkpoint](https://huggingface.co/amd/gpt-oss-120b-w-mxfp4-a-fp8)
- [OpenAI gpt-oss model overview](https://openai.com/index/introducing-gpt-oss/)
- [AITER PR #3428](https://github.com/ROCm/aiter/pull/3428)
- [AITER PR #3580](https://github.com/ROCm/aiter/pull/3580)
- [vLLM PR #44804](https://github.com/vllm-project/vllm/pull/44804)

## Benchmark disclaimer

Measurements were run on an AMD Instinct™ MI355X platform using the configuration below.

**Hardware configuration**

8× AMD Instinct™ MI355X GPUs (gfx950) with 2× AMD EPYC™ 9575F 64-Core processors.

**Software configuration**

Ubuntu 22.04.5 LTS, ROCm/HIP runtime 7.2.53211, vLLM 0.23.1rc1.dev1120+g0f0f28b53, PyTorch 2.11.0+gitd0c8b1f, Transformers 5.13.1, Python 3.12.13, AITER 0.1.16, Triton 3.6.0.

Results apply to the hardware, software, model, request shapes, tensor-parallel degrees, and measurement procedures described here. Differences in server configuration, firmware, framework and vLLM versions, kernel selection, clocks, drivers, optimizations, or concurrent system activity can materially change performance. Cross-platform numbers are dated snapshots and should not be generalized beyond the matched cells and software versions disclosed with the final dataset.
