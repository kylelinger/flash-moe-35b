# Technical Architecture

## Model: Qwen3.5-35B-A3B-4bit

### Overview

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 2048 |
| Number of layers | 40 |
| Attention heads | 16 |
| KV heads | 2 (GQA) |
| Head dimension | 256 |
| Experts per layer | 256 |
| Active experts (K) | 4 (pruned from 8) |
| MoE intermediate | 512 |
| Shared expert intermediate | 512 |
| Vocab size | 248,320 |
| Quantization | 4-bit, group_size=64 |

### Layer Types

- **Linear attention (GatedDeltaNet)**: Layers 0-2, 4-6, 8-10, 12-14, 16-18, 20-22, 24-26, 28-30, 32-34, 36-38 (30 layers)
  - Conv1d (kernel=4) + gated delta recurrence
  - Recurrent state: [32 × 128 × 128] float per layer (2 MB)
  - Conv state: [3 × 8192] float per layer (96 KB)
  - No KV cache needed (constant memory per layer)

- **Full attention**: Layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 (10 layers)
  - Standard QKV + scaled dot product + RoPE
  - KV cache: grows with context length
  - RoPE theta: 10,000,000

### Expert Weight Layout

Each expert is stored as a contiguous 1,769,472-byte binary block:

```
Offset      Size      Component
0           524,288   gate_proj.weight  (4-bit packed uint32)
524,288     32,768    gate_proj.scales  (bf16)
557,056     32,768    gate_proj.biases  (bf16)
589,824     524,288   up_proj.weight
1,114,112   32,768    up_proj.scales
1,146,880   32,768    up_proj.biases
1,179,648   524,288   down_proj.weight
1,703,936   32,768    down_proj.scales
1,736,704   32,768    down_proj.biases
─────────────────────
Total:      1,769,472 bytes per expert
```

Per-layer file: 256 experts × 1,769,472 = 452,984,832 bytes (~432 MB)
Total experts: 40 layers × 432 MB = ~17 GB

## Inference Pipeline

### Per-Token Forward Pass (3 Command Buffers per Layer)

```
For each layer (40 total):

  CMD1: Attention Input Projections
  ├── GPU: Q/K/V projections (dequant matvec)
  ├── For full attn: Q=[16×256×2], K=[2×256], V=[2×256]
  └── For linear attn: QKV=[8192], Z=[4096], beta=[32], alpha=[32]
  [commit + wait]

  CPU: Attention Compute
  ├── Full attn: RoPE → scaled dot product → softmax → weighted sum
  └── Linear attn: conv1d → delta recurrence → gated output

  CMD2: Post-Attention + Routing + Shared Expert (8 encoders, 1 commit)
  ├── GPU: o_proj (attn output → hidden dim)
  ├── GPU: residual_add (hidden += attn_output)
  ├── GPU: rms_norm (post-attention)
  ├── GPU: routing gate (hidden → 256 expert scores)
  ├── GPU: shared_expert gate_proj (hidden → 512)
  ├── GPU: shared_expert up_proj (hidden → 512)
  └── GPU: shared_expert_gate (hidden → 1 score)
  [commit + wait]

  CPU: Top-K Routing + Parallel Expert I/O
  ├── softmax(256 scores) → top-K=4 expert indices + weights
  └── 4 × parallel pread (GCD dispatch_apply):
      ├── thread 0: pread(layer_fd, expert[k0], 1.77MB, offset0)
      ├── thread 1: pread(layer_fd, expert[k1], 1.77MB, offset1)
      ├── thread 2: pread(layer_fd, expert[k2], 1.77MB, offset2)
      └── thread 3: pread(layer_fd, expert[k3], 1.77MB, offset3)

  CMD3: Expert Forwards + Combine (DEFERRED — async commit, NO wait)
  ├── GPU: 4 × expert forward (gate+up dequant matvec → SwiGLU → down dequant matvec)
  ├── GPU: shared expert SwiGLU + down_proj
  ├── GPU: moe_combine_residual (weighted sum + residual + shared gate → hidden)
  └── GPU: rms_norm (input norm for next layer → buf_input)
  [commit, NO wait — GPU queue serializes automatically]

  → Return immediately. Next layer's CMD1 can submit to GPU queue.
    GPU processes CMD3(layer N) then CMD1(layer N+1) in order.
```

### Why This Is Fast

**SwiftLM's approach (960 serial syncs/token):**
```
For each layer, for each expert k (K=8):
  for each projection (gate, up, down):
    CPU: pread expert weight (0.4ms)
    GPU: dispatch matvec (0.1ms)
    CPU: Stream.gpu.synchronize() ← BLOCKS (0.1ms)
Total: 40 × 8 × 3 × 0.6ms = 576ms/token
```

**Flash-MoE's approach (3 cmd buffers/layer, async):**
```
For each layer:
  CMD1: batch all attention projections (1 commit, 1 wait)
  CPU:  attention compute (runs while GPU finishes CMD1)
  CMD2: batch post-attn + routing + shared expert (1 commit, 1 wait)
  CPU:  top-K + 4 parallel preads (runs while GPU finishes CMD2)
  CMD3: batch all K expert forwards + combine (1 commit, NO wait)
Total: 40 × ~4ms = 160ms/token
```

Key difference: **960 sync points → 80 sync points** (2 per layer instead of 24), plus parallel I/O and deferred GPU compute.

## Metal Shader Overview

All in `shaders.metal`, compiled at runtime via `MTLDevice newLibraryWithSource:`:

| Kernel | Purpose | Used In |
|--------|---------|---------|
| `dequant_matvec_4bit_v3` | Fast 4-bit dequant matrix-vector multiply (shared mem) | All projections |
| `dequant_matvec_4bit_fast` | Large matrix matvec (> 4096 input dim) | lm_head |
| `swiglu_kernel` | SiLU(gate) × up activation | Expert + shared expert |
| `rms_norm_sum_sq` | RMS norm reduction (sum of squares) | Layer norms |
| `rms_norm_apply_bf16` | RMS norm apply with bf16 weights | Layer norms |
| `residual_add` | Element-wise addition | Residual connections |
| `moe_combine_residual` | Weighted expert sum + residual + shared gate | MoE output |
| `delta_net_step` | Gated delta-net recurrence update | Linear attention |
| `gpu_attn_scores` | Q×K^T dot products for full attention | Full attention |
| `gpu_attn_apply` | Softmax attention × V | Full attention |
| `rope_rotate` | RoPE positional encoding | Full attention |

## Memory Layout

### Resident in RAM (~1.5 GB)
- Non-expert weights: 1.38 GB (mmap'd, shared Metal buffer)
- Delta-net state: 65.9 MB (30 layers × 2 MB each)
- KV cache buffers: 168 MB (10 full-attn layers × 16.8 MB pre-allocated)
- Metal buffers: ~50 MB (expert I/O, scratch, attention)

### On SSD (~17 GB)
- Packed expert files: 40 × 432 MB
- Accessed via `pread()` per token, ~7 MB/token (4 experts × 1.77 MB each)
- OS page cache provides ~32 GB/s for repeated experts (vs 5.5 GB/s cold SSD)
