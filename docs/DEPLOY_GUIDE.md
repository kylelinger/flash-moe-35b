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

---

## Step 5: 服务管理 (启动/停止/状态检查)

### 启动服务

**前台运行（调试用）：**
```bash
cd weights/
../infer --model $MODEL_PATH --serve 5414 --k 4
```

**后台运行（推荐）：**
```bash
cd weights/
nohup ../infer --model $MODEL_PATH --serve 5414 --k 4 > /tmp/flash-moe-35b-server.log 2>&1 &
echo "Server PID: $!"
```

**等待服务就绪：**

启动后需要等待系统提示词预填充完成（约 3-5 秒），可通过日志确认：
```bash
# 查看启动日志，等待 "System prompt cached" 出现
tail -f /tmp/flash-moe-35b-server.log
# 看到以下输出说明服务就绪：
# [serve] System prompt cached: 21 tokens prefilled
# Ctrl+C 退出 tail
```

或用 health check 轮询：
```bash
# 等待服务就绪
until curl -s http://localhost:5414/health > /dev/null 2>&1; do sleep 1; done
echo "Service ready!"
```

### 检查服务状态

```bash
# 方法1: Health check API
curl -s http://localhost:5414/health
# 返回: {"status":"ok","model":"qwen3.5-397b-a17b"}

# 方法2: 查看进程
ps aux | grep "infer --serve" | grep -v grep

# 方法3: 查看日志
tail -20 /tmp/flash-moe-35b-server.log
```

### 停止服务

```bash
# 方法1: 通过 PID 停止（推荐）
kill $(pgrep -f "infer --serve")

# 方法2: 如果知道 PID
kill <PID>

# 确认已停止
ps aux | grep "infer --serve" | grep -v grep
# 应无输出
```

### 重启服务

```bash
# 先停止
kill $(pgrep -f "infer --serve") 2>/dev/null
sleep 2

# 再启动
cd weights/
nohup ../infer --model $MODEL_PATH --serve 5414 --k 4 > /tmp/flash-moe-35b-server.log 2>&1 &
echo "Restarted, PID: $!"

# 等待就绪
until curl -s http://localhost:5414/health > /dev/null 2>&1; do sleep 1; done
echo "Service ready!"
```

### 端口冲突处理

如果端口 5414 被占用：
```bash
# 查看谁占用了端口
lsof -i :5414

# 换一个端口启动
../infer --model $MODEL_PATH --serve 8080 --k 4
```

---

## Step 6: 使用服务

### 6a. 命令行 curl 测试

```bash
# 基础对话
curl http://localhost:5414/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "你好，请用中文介绍一下自己"}],
    "max_tokens": 200
  }'

# 查看可用模型
curl http://localhost:5414/v1/models
```

### 6b. Python 调用（OpenAI SDK 兼容）

```bash
pip3 install openai
```

```python
from openai import OpenAI

# 连接本地 Flash-MoE 服务
client = OpenAI(
    base_url="http://localhost:5414/v1",
    api_key="unused"  # 本地服务无需 API key
)

# 非流式调用
response = client.chat.completions.create(
    model="qwen3.5-35b",
    messages=[
        {"role": "user", "content": "什么是量子计算？用简单的话解释"}
    ],
    max_tokens=200,
    stream=False
)
# 注意：非流式模式下也返回 SSE 格式，需要解析

# 流式调用（推荐）
response = client.chat.completions.create(
    model="qwen3.5-35b",
    messages=[
        {"role": "user", "content": "写一首关于春天的诗"}
    ],
    max_tokens=200,
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### 6c. JavaScript/Node.js 调用

```javascript
const response = await fetch('http://localhost:5414/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'qwen3.5-35b',
    messages: [{ role: 'user', content: 'Hello!' }],
    max_tokens: 100
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const text = decoder.decode(value);
  // 解析 SSE 格式: "data: {...}\n\n"
  for (const line of text.split('\n')) {
    if (line.startsWith('data: ') && line !== 'data: [DONE]') {
      const data = JSON.parse(line.slice(6));
      const content = data.choices?.[0]?.delta?.content;
      if (content) process.stdout.write(content);
    }
  }
}
```

### 6d. 单次命令行推理（不启动服务）

```bash
cd weights/
../infer --model $MODEL_PATH --prompt "Explain relativity" --tokens 100 --k 4
```

---

## API 参考

### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint, 支持 SSE 流式输出。

**请求体：**
```json
{
  "model": "qwen3.5-35b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Your question here"}
  ],
  "max_tokens": 100,
  "stream": true
}
```

**响应格式（SSE 流）：**
```
data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### GET /v1/models

返回可用模型列表。

### GET /health

返回服务健康状态：`{"status":"ok","model":"..."}`

---

## 与 SwiftLM 的关系

本项目**替代** SwiftLM 来驱动 Qwen3.5-35B-A3B-4bit 模型：

| 对比 | SwiftLM | Flash-MoE |
|------|---------|-----------|
| 端口 | 5413 | 5414 |
| 生成速度 | 1.72 tok/s | 6.61 tok/s |
| 引擎 | Swift/MLX (预编译) | Objective-C/Metal (源码编译) |
| Expert 加载 | 逐个串行 + GPU 同步 | 并行 pread + 延迟 GPU |
| 依赖 | SwiftLM 二进制 | 仅 clang + Metal framework |

**如果不再需要 SwiftLM，停止它：**
```bash
kill $(pgrep -f SwiftLM)
```

---

## Troubleshooting

### "Cannot open vocab vocab.bin"
推理引擎在当前工作目录查找 `vocab.bin`、`tokenizer.bin`、`model_weights.bin`。可以：
- `cd` 到 weights 目录再运行
- 用 `--vocab`、`--weights`、`--manifest` 参数指定路径

### 首 Token 较慢
启动后第一个 token 较慢（~1.7s）：
1. Metal shader 编译（~260ms，仅首次）
2. System prompt 预填充（~3s，21 tokens）
3. Expert 权重页面缓存为冷缓存

后续 token 在页面缓存预热后达到全速（6+ tok/s）。

### 进程被 Kill（信号 9）
这是 macOS 的内存压力终止（OOM kill）：
- 关闭其他占内存的应用
- 确认没有同时运行 SwiftLM 和 Flash-MoE
- 引擎需要 ~2 GB RAM + SSD 带宽

### 输出乱码
如果 API 返回 BPE 编码 token（如 `Ġ` 代替空格）：
- 确保 `vocab.bin` 由 `export_vocab.py` 生成（包含 BPE 字节解码）
- 不要把 `tokenizer.bin` 当 `vocab.bin` 用——格式不同

### 长上下文响应慢
Prefill 阶段是逐 token 处理，8K token 上下文需要约 15 分钟。这是当前的限制，生成阶段不受影响（仍然 4-6 tok/s）。

---

## 文件位置总览

| 文件 | 位置 | 大小 | 说明 |
|------|------|------|------|
| model_weights.bin | weights/ | 1.4 GB | 非 expert 权重（mmap 加载） |
| model_weights.json | weights/ | 246 KB | 权重张量清单 |
| vocab.bin | weights/ | 2.2 MB | 解码词表（输出用） |
| tokenizer.bin | weights/ | 7.8 MB | BPE 分词器（输入用） |
| packed_experts/ | $MODEL_PATH/ | 17 GB | 按层打包的 expert 二进制文件 |
| config.json | $MODEL_PATH/ | 2 KB | 模型架构配置 |
| /tmp/flash-moe-35b-server.log | /tmp/ | — | 服务运行日志 |
