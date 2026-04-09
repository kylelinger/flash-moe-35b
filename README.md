# Flash-MoE 35B: Fast Local Inference for Qwen3.5-35B-A3B on Apple Silicon

High-performance inference engine for **Qwen3.5-35B-A3B-4bit** on Apple Silicon Macs, adapted from the [Flash-MoE](https://github.com/danveloper/flash-moe) paper's Metal/Objective-C engine.

Achieves **6.6 tok/s generation** (peak 10.3 tok/s) on M4 with 16GB RAM — **7x faster** than SwiftLM's SSD-streaming mode.

## Performance

| Metric | Flash-MoE 35B | SwiftLM K=4 | Speedup |
|--------|--------------|-------------|---------|
| Generation (short) | 6.61 tok/s | 1.72 tok/s | **3.8x** |
| Generation (8K ctx) | 4.00 tok/s | — | — |
| Peak speed | 10.27 tok/s | — | **6.0x** |
| Prefill | 8.6 tok/s | 33.3 tok/s | 0.26x |

> Prefill is slower because Flash-MoE processes tokens sequentially (no batched attention).
> Generation is dramatically faster due to the deferred GPU pipeline eliminating serial synchronization.

## Architecture

**Qwen3.5-35B-A3B-4bit** — 40-layer Mixture-of-Experts transformer:
- Hidden size: 2048, Head dim: 256
- 256 experts/layer, top-K=4 active (pruned from K=8)
- 30 linear attention layers (GatedDeltaNet) + 10 full attention layers
- 4-bit quantized, group_size=64
- Total model size: ~18 GB (experts on SSD, ~1.4 GB non-expert weights in RAM)

## Key Optimizations

1. **Deferred CMD3 Pipeline** — GPU expert computation is submitted async; next layer's attention projections start immediately without waiting. Eliminates the 960 serial `GPU.synchronize()` calls per token that plague SwiftLM.

2. **Parallel pread** — 4 pthreads via GCD `dispatch_apply` load K expert weights simultaneously from SSD (~5.5 GB/s per thread, ~20 GB/s aggregate with page cache).

3. **Trust the OS Page Cache** — No application-level LRU cache. Removes memory compressor thrashing that causes 38% overhead on 16GB machines.

4. **Expert Pruning K=8→K=4** — No measurable quality loss; halves expert I/O per token; reduces memory demand from 11 GB to 3.5 GB.

5. **GPU-side Combine** — MoE output combination (weighted expert sum + residual + RMS norm) runs entirely on GPU in CMD3, eliminating CPU readback between layers.

## Requirements

- macOS 14+ (Sonoma or later)
- Apple Silicon Mac (M1/M2/M3/M4, any variant)
- 16 GB RAM minimum (tested on M4 16GB)
- ~20 GB free disk space (model weights)
- Python 3.9+ (for weight preparation scripts)
- Xcode Command Line Tools (`xcode-select --install`)
- No Xcode IDE required (Metal shaders compile at runtime)

## Quick Start

### 1. Download the Model

```bash
pip3 install huggingface_hub
huggingface-cli download mlx-community/Qwen3.5-35B-A3B-4bit --local-dir ~/models/Qwen3.5-35B-A3B-4bit
```

Or use the snapshot path if already cached:
```bash
export MODEL_PATH=~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/<hash>
```

### 2. Prepare Weights

```bash
# Extract non-expert weights (~1.4 GB)
python3 scripts/extract_weights_35b.py --model $MODEL_PATH --output weights/

# Export tokenizer and vocabulary
python3 scripts/export_tokenizer.py $MODEL_PATH/tokenizer.json weights/tokenizer.bin
python3 scripts/export_vocab.py $MODEL_PATH/tokenizer.json weights/vocab.bin

# Repack expert weights (~17 GB, takes ~5 min)
python3 scripts/repack_experts_35b.py --model $MODEL_PATH --output $MODEL_PATH/packed_experts/
```

### 3. Build

```bash
make infer
```

### 4. Run

**Single prompt:**
```bash
./infer --prompt "Explain quantum computing" --tokens 100 --k 4
```

**HTTP Server (OpenAI-compatible API):**
```bash
# 后台启动
nohup ./infer --serve 5414 --k 4 > /tmp/flash-moe-35b-server.log 2>&1 &
echo "PID: $!"

# 等待就绪 (~5s)
until curl -s http://localhost:5414/health > /dev/null 2>&1; do sleep 1; done
echo "Service ready!"
```

**测试：**
```bash
curl http://localhost:5414/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-35b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'
```

**停止服务：**
```bash
kill $(pgrep -f "infer --serve")
```

**检查服务状态：**
```bash
curl -s http://localhost:5414/health   # API 健康检查
ps aux | grep "infer --serve"          # 查看进程
tail -5 /tmp/flash-moe-35b-server.log  # 查看日志
```

> 详细的服务管理说明、Python/JS 调用示例见 [docs/DEPLOY_GUIDE.md](docs/DEPLOY_GUIDE.md)

## Project Structure

```
flash-moe-35b/
├── infer.m              # Main inference engine (Objective-C/Metal, ~7K lines)
├── shaders.metal        # GPU kernels (4-bit dequant matvec, SwiGLU, RMS norm, etc.)
├── tokenizer.h          # C BPE tokenizer implementation
├── Makefile             # Build system (clang + Metal framework)
├── scripts/
│   ├── extract_weights_35b.py   # Extract non-expert weights from safetensors
│   ├── repack_experts_35b.py    # Repack experts into per-layer binary files
│   ├── export_tokenizer.py      # Export BPE tokenizer to binary format
│   └── export_vocab.py          # Export vocab with BPE byte decoding
├── benchmarks/
│   └── results.md               # Detailed benchmark results
├── docs/
│   ├── DEPLOY_GUIDE.md          # Step-by-step deployment guide
│   ├── ARCHITECTURE.md          # Technical architecture deep-dive
│   └── PROGRESS.md              # Development progress and task log
└── README.md
```

## HTTP API

The server implements an OpenAI-compatible API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (SSE streaming) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

### Request Format

```json
{
  "model": "qwen3.5-35b",
  "messages": [
    {"role": "user", "content": "Your message here"}
  ],
  "max_tokens": 100,
  "stream": true
}
```

## CLI Options

```
./infer [OPTIONS]

  --prompt TEXT         Input prompt text
  --tokens N            Max tokens to generate (default: 20)
  --k N                 Active experts per layer (default: 4)
  --serve PORT          Run HTTP server on PORT
  --model PATH          Model directory path
  --weights PATH        model_weights.bin path
  --manifest PATH       model_weights.json path
  --vocab PATH          vocab.bin path
  --timing              Print per-layer timing breakdown
  --freq                Track expert activation frequency
```

## Acknowledgments

- [Flash-MoE paper](https://github.com/danveloper/flash-moe) by Daniel Woods — the original 397B inference engine and optimization insights
- [Qwen3.5](https://github.com/QwenLM/Qwen3) by Alibaba — the model architecture
- [MLX Community](https://huggingface.co/mlx-community) — 4-bit quantized model weights

## License

MIT License — See individual files for attribution.
