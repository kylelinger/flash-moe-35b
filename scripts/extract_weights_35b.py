#!/usr/bin/env python3
"""
extract_weights_35b.py — Extract all non-expert weights from Qwen3.5-35B-A3B-4bit
into a single binary file for the C inference engine.

Adapted from extract_weights.py (397B) for the 35B model.
"""

import json
import struct
import sys
import os
import argparse
import time
from pathlib import Path
from collections import defaultdict
import re


def parse_safetensors_header(filepath):
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def main():
    parser = argparse.ArgumentParser(description='Extract non-expert weights (35B)')
    parser.add_argument('--model', type=str,
                        default=os.path.expanduser(
                            '~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit'
                            '/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec'),
                        help='Path to model directory')
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory')
    args = parser.parse_args()

    model_path = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = model_path / 'model.safetensors.index.json'
    if not index_path.exists():
        print(f"ERROR: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(index_path) as f:
        idx = json.load(f)

    weight_map = idx['weight_map']

    expert_pattern = re.compile(r'\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$')
    vision_pattern = re.compile(r'^(vision_tower|model\.visual)')

    tensors_to_extract = {}
    skipped_expert = 0
    skipped_vision = 0

    for name, filename in weight_map.items():
        if vision_pattern.match(name):
            skipped_vision += 1
            continue
        if expert_pattern.search(name):
            skipped_expert += 1
            continue
        tensors_to_extract[name] = filename

    print(f"Model: {model_path}")
    print(f"Total weights: {len(weight_map)}")
    print(f"Skipped vision: {skipped_vision}, expert: {skipped_expert}")
    print(f"Extracting: {len(tensors_to_extract)} tensors")

    by_file = defaultdict(list)
    for name, filename in tensors_to_extract.items():
        by_file[filename].append(name)

    print("\nParsing safetensors headers...")
    header_cache = {}
    for filename in sorted(by_file.keys()):
        filepath = model_path / filename
        header_cache[filename] = parse_safetensors_header(str(filepath))

    def sanitize_name(name):
        if name.startswith("language_model."):
            return name[len("language_model."):]
        return name

    all_tensors = []
    for name in sorted(tensors_to_extract.keys()):
        san_name = sanitize_name(name)
        all_tensors.append((san_name, name, tensors_to_extract[name]))

    # Read actual model config
    config_path = model_path / 'config.json'
    with open(config_path) as f:
        model_config = json.load(f)
    tc = model_config.get('text_config', model_config)

    layer_types = tc.get('layer_types', [])

    bin_path = output_dir / 'model_weights.bin'
    manifest = {
        "model": str(model_path),
        "num_tensors": len(all_tensors),
        "tensors": {},
        "config": {
            "hidden_size": tc['hidden_size'],
            "num_hidden_layers": tc['num_hidden_layers'],
            "num_attention_heads": tc['num_attention_heads'],
            "num_key_value_heads": tc['num_key_value_heads'],
            "head_dim": tc['head_dim'],
            "vocab_size": tc['vocab_size'],
            "rms_norm_eps": tc.get('rms_norm_eps', 1e-6),
            "num_experts": tc['num_experts'],
            "num_experts_per_tok": tc['num_experts_per_tok'],
            "moe_intermediate_size": tc['moe_intermediate_size'],
            "shared_expert_intermediate_size": tc['shared_expert_intermediate_size'],
            "full_attention_interval": tc['full_attention_interval'],
            "linear_num_value_heads": tc['linear_num_value_heads'],
            "linear_num_key_heads": tc['linear_num_key_heads'],
            "linear_key_head_dim": tc['linear_key_head_dim'],
            "linear_value_head_dim": tc['linear_value_head_dim'],
            "linear_conv_kernel_dim": tc['linear_conv_kernel_dim'],
            "partial_rotary_factor": tc.get('rope_parameters', {}).get('partial_rotary_factor', 0.25),
            "rope_theta": tc.get('rope_parameters', {}).get('rope_theta', 10000000.0),
            "layer_types": layer_types,
        }
    }

    print(f"\nWriting {bin_path}...")
    t0 = time.time()
    offset = 0
    total_bytes = 0
    ALIGN = 64

    with open(bin_path, 'wb') as out_f:
        for i, (san_name, orig_name, filename) in enumerate(all_tensors):
            filepath = model_path / filename
            header, data_start = header_cache[filename]

            if orig_name not in header:
                print(f"  WARNING: {orig_name} not found in {filename}, skipping")
                continue

            meta = header[orig_name]
            tensor_offsets = meta['data_offsets']
            byte_len = tensor_offsets[1] - tensor_offsets[0]
            shape = meta['shape']
            dtype = meta['dtype']

            if offset % ALIGN != 0:
                pad = ALIGN - (offset % ALIGN)
                out_f.write(b'\x00' * pad)
                offset += pad

            with open(filepath, 'rb') as sf:
                sf.seek(data_start + tensor_offsets[0])
                data = sf.read(byte_len)

            out_f.write(data)

            manifest["tensors"][san_name] = {
                "offset": offset,
                "size": byte_len,
                "shape": shape,
                "dtype": dtype,
            }

            offset += byte_len
            total_bytes += byte_len

            if (i + 1) % 50 == 0 or i == len(all_tensors) - 1:
                print(f"  [{i+1}/{len(all_tensors)}] {total_bytes / 1e6:.1f} MB written")

    elapsed = time.time() - t0
    print(f"\nDone: {total_bytes / 1e6:.1f} MB in {elapsed:.1f}s")
    print(f"Binary: {bin_path} ({os.path.getsize(bin_path) / 1e6:.1f} MB)")

    json_path = output_dir / 'model_weights.json'
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {json_path}")

    # Also generate expert_index.json for repack_experts
    print("\nGenerating expert_index.json...")
    expert_reads = {}
    expert_count = 0
    for name, filename in weight_map.items():
        if not expert_pattern.search(name):
            continue
        san_name = sanitize_name(name)
        # Parse: model.layers.X.mlp.switch_mlp.{gate_proj|up_proj|down_proj}.{weight|scales|biases}
        m = re.match(r'.*layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$', name)
        if not m:
            continue
        layer_idx = m.group(1)
        proj = m.group(2)
        comp = m.group(3)
        comp_name = f"{proj}.{comp}"

        filepath = model_path / filename
        header, data_start = header_cache.get(filename) or parse_safetensors_header(str(filepath))
        if filename not in header_cache:
            header_cache[filename] = (header, data_start)

        if name not in header:
            continue

        meta = header[name]
        tensor_offsets = meta['data_offsets']
        byte_len = tensor_offsets[1] - tensor_offsets[0]
        shape = meta['shape']

        num_experts = tc['num_experts']
        expert_size = byte_len // num_experts
        expert_stride = expert_size

        if layer_idx not in expert_reads:
            expert_reads[layer_idx] = {}

        expert_reads[layer_idx][comp_name] = {
            "file": filename,
            "abs_offset": data_start + tensor_offsets[0],
            "expert_size": expert_size,
            "expert_stride": expert_stride,
            "shape": shape,
            "dtype": meta['dtype'],
        }
        expert_count += 1

    expert_index = {
        "model_path": str(model_path),
        "num_layers": tc['num_hidden_layers'],
        "num_experts": tc['num_experts'],
        "expert_reads": expert_reads,
    }

    ei_path = output_dir / 'expert_index.json'
    with open(ei_path, 'w') as f:
        json.dump(expert_index, f, indent=2)
    print(f"Expert index: {ei_path} ({expert_count} components across {len(expert_reads)} layers)")

    # Summary
    categories = defaultdict(lambda: {"count": 0, "bytes": 0})
    for san_name, info in manifest["tensors"].items():
        if "embed_tokens" in san_name:
            cat = "embedding"
        elif "norm.weight" in san_name and "layers." not in san_name:
            cat = "final_norm"
        elif "lm_head" in san_name:
            cat = "lm_head"
        elif "input_layernorm" in san_name or "post_attention_layernorm" in san_name:
            cat = "layer_norms"
        elif "linear_attn" in san_name:
            cat = "linear_attention"
        elif "self_attn" in san_name:
            cat = "full_attention"
        elif "mlp.gate." in san_name:
            cat = "routing_gate"
        elif "shared_expert." in san_name:
            cat = "shared_expert"
        elif "shared_expert_gate" in san_name:
            cat = "shared_expert_gate"
        else:
            cat = "other"
        categories[cat]["count"] += 1
        categories[cat]["bytes"] += info["size"]

    print("\nWeight categories:")
    for cat in sorted(categories.keys()):
        info = categories[cat]
        print(f"  {cat:25s}: {info['count']:4d} tensors, {info['bytes']/1e6:8.1f} MB")


if __name__ == '__main__':
    main()
