# Development Progress

## Project Timeline

### Phase 1: Analysis & Root Cause (2026-04-09)

- [x] Identify SwiftLM slowness: 0.95 tok/s with Qwen3.5-35B-A3B K=8
- [x] Read Flash-MoE paper (397B model at 5.74 tok/s on M3 Max)
- [x] Root cause analysis: 960 serial `Stream.gpu.synchronize()` calls per token in `SwitchLayers.swift`
- [x] Calculate theoretical overhead: 40 layers × 8 experts × 3 projections × 0.6ms = 576ms/token

### Phase 2: Expert Pruning K=8→K=4 (2026-04-09)

- [x] Modify `config.json`: `num_experts_per_tok` 8 → 4
- [x] Restart SwiftLM, validate output quality
- [x] Benchmark: 0.95 → 1.72 tok/s (+81%)
- [x] Memory reduction: MEM_DEMAND 11 GB → 3.5 GB (-68%)
- [x] 100K context test: 99,193 tokens prefilled at 33.3 tok/s

### Phase 3: Flash-MoE Engine Adaptation (2026-04-09)

- [x] Download Flash-MoE source from GitHub
- [x] Create `extract_weights_35b.py` for non-expert weight extraction
- [x] Create `repack_experts_35b.py` for expert binary repacking
- [x] Extract non-expert weights: 1,378.9 MB → `model_weights.bin`
- [x] Repack 40 layers × 256 experts: ~17 GB → `packed_experts/`
- [x] Adapt `infer.m` architecture constants (20+ changes):
  - HIDDEN_DIM: 4096 → 2048
  - NUM_LAYERS: 60 → 40
  - NUM_ATTN_HEADS: 32 → 16
  - NUM_EXPERTS: 512 → 256
  - NUM_EXPERTS_PER_TOK: 10 → 4
  - MOE_INTERMEDIATE: 1024 → 512
  - LINEAR_NUM_V_HEADS: 64 → 32
  - EXPERT_SIZE: 7,077,888 → 1,769,472
  - NUM_FULL_ATTN_LAYERS: 15 → 10
  - NUM_LINEAR_LAYERS: 45 → 30
  - All hardcoded 4-bit expert offsets → #define constants
  - Delta-net state buffer sizes
  - Conv state buffer sizes
  - MODEL_PATH_DEFAULT → kylexu's model path
- [x] Export `vocab.bin` with BPE byte→UTF-8 decoding
- [x] Export `tokenizer.bin` for prompt encoding
- [x] Build with `make infer` (no Xcode needed)
- [x] First successful inference: "Hello" → ", high school science"
- [x] 50-token benchmark: **6.61 tok/s** (peak 10.27 tok/s)

### Phase 4: HTTP Server & Long Context (2026-04-09)

- [x] Start HTTP server on port 5414 (OpenAI-compatible API)
- [x] Fix vocab.bin BPE decoding (Ġ→space, Ċ→newline)
- [x] Verify SSE streaming works correctly
- [x] Test: "Say hello in 3 languages" → English/Español/Français
- [x] Attempt prefill expert skip optimization
- [x] Revert: linear attention recurrent state requires expert computation
- [x] Long context test: 2K tokens → prefill 221s, gen 4.65 tok/s
- [x] Long context test: 8.2K tokens → prefill 951s, gen 4.00 tok/s
- [x] Long context test: 8.2K tokens (2nd run) → prefill 876s, gen 4.02 tok/s

### Phase 5: Documentation & Packaging (2026-04-09)

- [x] Create project directory structure
- [x] Write README.md with quick start guide
- [x] Write DEPLOY_GUIDE.md with step-by-step instructions
- [x] Write ARCHITECTURE.md with technical deep-dive
- [x] Write benchmark results with detailed measurements
- [x] Write PROGRESS.md (this file)
- [ ] Initialize git repository
- [ ] Push to GitHub

## Key Decisions

### Why K=4 instead of K=8?
- K=4 gives 81% speed improvement with no visible quality loss
- Flash-MoE paper shows K=3 causes quality collapse
- K=4 cuts memory demand from 11 GB to 3.5 GB — critical for 16 GB machines
- Expert I/O per token: 4 × 1.77 MB = 7 MB (vs 14 MB for K=8)

### Why not skip experts during prefill?
- Attempted but reverted: linear attention (GatedDeltaNet) has recurrent state
- The delta-net state at each position depends on the hidden state quality
- Feeding shared-expert-only hidden states corrupts the recurrence
- Different from standard transformer KV cache which is position-independent
- Result: model output degraded to garbage when experts were skipped

### Why not use SwiftLM source modification?
- SwiftLM is distributed as a pre-built binary (release b222)
- Source compilation requires Xcode IDE (~35 GB download)
- The root cause (serial GPU sync in `SwitchLayers.swift`) is deep in the MLX framework
- Flash-MoE's Objective-C/Metal engine was faster to adapt and gave better results

## Open Tasks

- [ ] Prefill optimization: batch attention for non-MoE layers during prefill
- [ ] 2-bit expert requantization (44% I/O reduction per Flash-MoE paper)
- [ ] Expert prediction (speculative routing from previous token's activations)
- [ ] Multi-turn conversation session persistence
- [ ] Custom system prompt support (from ~/.flash-moe/system.md)
- [ ] Benchmark on M1/M2/M3 variants
- [ ] Compare with llama.cpp Qwen3.5 support (if available)
