# Deployment Guide: Flash-MoE 35B on Apple Silicon

Step-by-step guide to deploy Qwen3.5-35B-A3B-4bit with the Flash-MoE inference engine on Apple Silicon Macs.

## Prerequisites

### Hardware
- Apple Silicon Mac (M1/M2/M3/M4, any variant)
- 16 GB RAM minimum
- 20+ GB free disk space

### Software
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Python dependencies
pip3 install safetensors huggingface_hub
```

## Step 1: Download the Model

```bash
# Option A: Using huggingface-cli (recommended)
pip3 install huggingface_hub
huggingface-cli download mlx-community/Qwen3.5-35B-A3B-4bit

# The model will be cached at:
# ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/

# Find the snapshot path:
export MODEL_PATH=$(python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('mlx-community/Qwen3.5-35B-A3B-4bit', local_files_only=True)
print(path)
")
echo "Model at: $MODEL_PATH"
```

```bash
# Option B: Direct download to a specific directory
huggingface-cli download mlx-community/Qwen3.5-35B-A3B-4bit --local-dir ~/models/Qwen3.5-35B-A3B-4bit
export MODEL_PATH=~/models/Qwen3.5-35B-A3B-4bit
```

## Step 2: Apply Expert Pruning (K=8 → K=4)

Edit the model's `config.json` to reduce active experts from 8 to 4:

```bash
# Backup original config
cp $MODEL_PATH/config.json $MODEL_PATH/config.json.bak

# Change num_experts_per_tok from 8 to 4
python3 -c "
import json
cfg = json.load(open('$MODEL_PATH/config.json'))
print(f'Original K={cfg[\"num_experts_per_tok\"]}')
cfg['num_experts_per_tok'] = 4
json.dump(cfg, open('$MODEL_PATH/config.json', 'w'), indent=2)
print('Updated K=4')
"
```

## Step 3: Prepare Weights

### 3a. Extract Non-Expert Weights

```bash
mkdir -p weights/
python3 scripts/extract_weights_35b.py \
  --model $MODEL_PATH \
  --output weights/

# Output: weights/model_weights.bin (~1.4 GB)
#         weights/model_weights.json (weight manifest)
#         weights/expert_index.json (expert component offsets)
```

### 3b. Export Tokenizer and Vocabulary

```bash
python3 scripts/export_tokenizer.py \
  $MODEL_PATH/tokenizer.json \
  weights/tokenizer.bin

python3 scripts/export_vocab.py \
  $MODEL_PATH/tokenizer.json \
  weights/vocab.bin
```

### 3c. Repack Expert Weights

This creates per-layer binary files for efficient pread I/O:

```bash
python3 scripts/repack_experts_35b.py \
  --model $MODEL_PATH \
  --output $MODEL_PATH/packed_experts/

# Creates 40 files: layer_00.bin through layer_39.bin
# Each ~432 MB, total ~17 GB
# Takes approximately 5 minutes
```

## Step 4: Build

```bash
make infer
```

Build output: `./infer` (~140 KB binary). Metal shaders compile at runtime on first launch (~260ms, then cached by the OS).

## Step 5: Run

### Single Prompt

```bash
cd weights/  # or wherever model_weights.bin is
../infer --model $MODEL_PATH --prompt "Hello, world!" --tokens 50 --k 4
```

### HTTP Server (Production)

```bash
cd weights/
../infer --model $MODEL_PATH --serve 5414 --k 4
```

Server starts and pre-caches the system prompt (~3s). Then it's ready for requests:

```bash
# Health check
curl http://localhost:5414/health

# Chat completion
curl http://localhost:5414/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "max_tokens": 100
  }'
```

### OpenAI SDK Compatible

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5414/v1", api_key="unused")
response = client.chat.completions.create(
    model="qwen3.5-35b",
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
    max_tokens=200,
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Troubleshooting

### "Cannot open vocab vocab.bin"
The inference engine looks for `vocab.bin`, `tokenizer.bin`, and `model_weights.bin` in the current working directory. Either:
- `cd` to the weights directory before running
- Use `--vocab`, `--weights`, `--manifest` flags to specify paths

### Slow First Token
The first token after startup is slower (~1.7s) because:
1. Metal shader compilation (~260ms, first run only)
2. System prompt prefill (~3s for 21 tokens)
3. Expert weight page cache is cold

Subsequent tokens reach full speed (6+ tok/s) after page cache warms up.

### Out of Memory
If the process gets killed (signal 9):
- Reduce K from 4 to 3 (quality may degrade)
- Close other memory-intensive apps
- The engine needs ~2 GB RAM + SSD bandwidth for experts

### Garbled Output
If the HTTP API returns BPE-encoded tokens (like `Ġ` for space):
- Ensure `vocab.bin` was generated with `export_vocab.py` (includes BPE byte decoding)
- Don't use `tokenizer.bin` as `vocab.bin` — they are different formats

## File Locations Summary

| File | Location | Size | Description |
|------|----------|------|-------------|
| model_weights.bin | weights/ | 1.4 GB | Non-expert weights (mmap'd) |
| model_weights.json | weights/ | 246 KB | Weight tensor manifest |
| vocab.bin | weights/ | 2.2 MB | Decoded vocabulary for output |
| tokenizer.bin | weights/ | 7.8 MB | BPE tokenizer for input |
| packed_experts/ | $MODEL_PATH/ | 17 GB | Per-layer expert binaries |
| config.json | $MODEL_PATH/ | 2 KB | Model architecture config |
