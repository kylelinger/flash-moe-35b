# Benchmark Results

**Hardware:** Apple M4, 16 GB RAM, 256 GB NVMe SSD
**Model:** Qwen3.5-35B-A3B-4bit (mlx-community)
**Config:** K=4 experts, 40 layers, 4-bit quantized
**Date:** 2026-04-09

## Generation Speed

### Short Context (< 100 tokens)

| Tokens | Total Time | Gen Time | Gen Speed | TTFT | Peak Speed |
|--------|-----------|----------|-----------|------|------------|
| 5 | 2.6s | 0.9s | 4.43 tok/s | 1720 ms | 5.72 tok/s |
| 50 | 8.8s | 7.4s | 6.61 tok/s | 1393 ms | 10.27 tok/s |

**50-token detailed per-token timing (ms):**
```
175, 154, 176, 151, 131, 124, 178, 209, 159, 177,  (gen 1-10)
158, 163, 166, 160, 177, 139, 144, 137, 166, 153,  (gen 11-20)
157, 170, 164, 143, 160, 138, 142, 145, 147, 130,  (gen 21-30)
140, 128, 158, 150, 137, 122, 108, 107, 97,  148,  (gen 31-40)
133, 124, 167, 153, 175, 147, 146, 229, 149         (gen 41-49)
```

Minimum: 97 ms/tok (10.27 tok/s) — after page cache warm-up
Maximum: 229 ms/tok (4.36 tok/s) — page cache miss
Average: 150 ms/tok (6.61 tok/s)

### Long Context

| Context | Prefill Time | Prefill Rate | Gen Speed | Gen Tokens |
|---------|-------------|-------------|-----------|------------|
| 7 tokens | 1.4s | 5.0 tok/s | 6.61 tok/s | 50 |
| 2K tokens | 221.7s | 9.1 tok/s | 4.65 tok/s | 5 |
| 8.2K tokens | 950.8s | 8.6 tok/s | 4.00 tok/s | 20 |
| 8.2K tokens (2nd) | 876.4s | 9.4 tok/s | 4.02 tok/s | 10 |

## Comparison: Flash-MoE vs SwiftLM

### Generation Speed

| Engine | Config | Gen Speed | vs Baseline |
|--------|--------|-----------|-------------|
| SwiftLM | K=8 (default) | 0.95 tok/s | 1.0x |
| SwiftLM | K=4 (pruned) | 1.72 tok/s | 1.8x |
| **Flash-MoE** | **K=4** | **6.61 tok/s** | **7.0x** |

### Root Cause of SwiftLM's Slowness

SwiftLM's `SwitchLayers.swift` performs **960 serial GPU synchronizations per token**:
- 40 layers × 8 experts × 3 projections (gate/up/down)
- Each sync: ~0.4ms SSD read + ~0.1ms GPU compute + ~0.1ms sync overhead = 0.6ms
- Total: 960 × 0.6ms = 576ms minimum per token

Flash-MoE eliminates this by:
1. Batching all K expert GPU dispatches into a single command buffer (CMD3)
2. Deferring CMD3 commit — no CPU wait for GPU expert completion
3. Parallel pread via 4 GCD threads — all K experts loaded simultaneously
4. GPU-side combine — expert output weighting + residual + norm all on GPU

### Memory Usage

| Engine | Config | MEM_DEMAND | OS_RAM |
|--------|--------|------------|--------|
| SwiftLM K=8 | default | 11 GB | 8 GB |
| SwiftLM K=4 | pruned | 3.5 GB | 3 GB |
| Flash-MoE K=4 | optimized | ~2 GB | ~1.5 GB |

### Prefill Speed

| Engine | Config | Prefill Rate | Notes |
|--------|--------|-------------|-------|
| SwiftLM K=4 | 100K ctx | 33.3 tok/s | Batch MLX attention |
| Flash-MoE K=4 | 8K ctx | 8.6 tok/s | Sequential per-token |

> Flash-MoE prefill is slower because it processes tokens one-by-one through the full
> forward pass. SwiftLM uses MLX's batched attention which is more efficient for prefill.
> Generation speed is the primary optimization target for interactive use.

## Expert Pruning Analysis

| Config | Gen Speed | Quality | MEM_DEMAND |
|--------|-----------|---------|------------|
| K=8 (original) | 0.95 tok/s | Baseline | 11 GB |
| K=6 | — | Good | ~7 GB |
| K=4 | 6.61 tok/s | No visible degradation | ~3.5 GB |
| K=3 | — | Quality collapse (per Flash-MoE paper) | — |

K=4 provides the best speed/quality/memory tradeoff for 16 GB machines.

## Output Quality Samples

### Short Context
**Prompt:** "Explain quantum computing in simple terms"
**Output (50 tokens):**
> Imagine a computer like the one you use right now. It processes information using **bits**. You can think of a bit as a light switch: it's either **off** (0) or **on** (1)

### Long Context (8K tokens)
**Prompt:** Repeated "The quick brown fox jumps over the lazy dog" × 2000, then "What animal jumps?"
**Output:** "The quick brown fox" — correctly identified the animal from the repeated text.

## System Info

```
Hardware:    Apple M4, 10-core GPU, 16 GB unified memory
SSD:         256 GB NVMe (~5.5 GB/s sequential read)
OS:          macOS 15.x (Darwin 25.3.0)
Compiler:    Apple clang 17.0.0
Metal:       GPU shaders compiled at runtime (no Xcode required)
Model size:  ~18 GB total (1.4 GB non-expert + 17 GB packed experts)
```
