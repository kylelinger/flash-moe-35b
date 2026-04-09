#!/usr/bin/env python3
"""Export vocab.bin for the C inference engine's decode_token().

Binary format:
  uint32 num_entries
  uint32 max_id
  For each entry (0..max_id):
    uint16 byte_len
    char[byte_len] UTF-8 string

BPE tokens use a byte-level encoding where printable/safe bytes map to themselves
and others map to Unicode chars starting at U+0100. This script reverses that
mapping so the C code gets clean UTF-8 strings.
"""
import json, struct, sys, os

def bytes_to_unicode():
    """GPT-2 style byte-to-unicode mapping (same as used by Qwen tokenizers)."""
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): bytes([b]) for b, c in zip(bs, cs)}

BYTE_DECODER = bytes_to_unicode()

def decode_bpe_token(token_str):
    """Convert BPE token string back to raw bytes."""
    try:
        return b''.join(BYTE_DECODER.get(c, c.encode('utf-8')) for c in token_str)
    except Exception:
        return token_str.encode('utf-8', errors='replace')

model_path = os.path.expanduser(
    '~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit'
    '/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec'
)
tok_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(model_path, 'tokenizer.json')
out_path = sys.argv[2] if len(sys.argv) > 2 else 'vocab.bin'

with open(tok_path, 'r', encoding='utf-8') as f:
    t = json.load(f)

vocab = t['model']['vocab']  # str -> int
added = {tok['content']: tok['id'] for tok in t['added_tokens']}

# Merge all tokens
all_tokens = {}
for s, i in vocab.items():
    # Decode BPE byte-level encoding for regular vocab tokens
    all_tokens[i] = decode_bpe_token(s)
for s, i in added.items():
    # Added/special tokens are stored as-is (not byte-level encoded)
    all_tokens[i] = s.encode('utf-8')

max_id = max(all_tokens.keys())
num_entries = max_id + 1

with open(out_path, 'wb') as f:
    f.write(struct.pack('<I', num_entries))
    f.write(struct.pack('<I', max_id))
    for i in range(num_entries):
        b = all_tokens.get(i, b'')
        f.write(struct.pack('<H', len(b)))
        if b:
            f.write(b)

sz = os.path.getsize(out_path)
print(f"Exported {out_path}: {num_entries} entries, {sz/1024/1024:.1f} MB")

# Verify a few known tokens
test_cases = {9419: "Hello", 271: "\n\n", 264: " a"}
for tid, expected in test_cases.items():
    actual = all_tokens.get(tid, b'').decode('utf-8', errors='replace')
    status = "OK" if actual == expected else f"MISMATCH (got {repr(actual)})"
    print(f"  token {tid}: {repr(actual)} {status}")
