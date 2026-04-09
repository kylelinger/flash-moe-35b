#!/usr/bin/env python3
"""Repack expert weights for Qwen3.5-35B-A3B-4bit into contiguous per-layer binary files.

Creates one binary file per layer: packed_experts/layer_XX.bin
Each file = 256 experts x 1,769,472 bytes = ~432 MB

Expert layout (4-bit, group_size=64):
  gate_proj: [512, 2048] -> weight=524288, scales=32768, biases=32768
  up_proj:   [512, 2048] -> weight=524288, scales=32768, biases=32768
  down_proj: [2048, 512] -> weight=524288, scales=32768, biases=32768
  Total per expert: 1,769,472 bytes

Usage:
    python repack_experts_35b.py --index expert_index.json
    python repack_experts_35b.py --index expert_index.json --layers 0-4
    python repack_experts_35b.py --index expert_index.json --dry-run
"""

import argparse
import json
import os
import time
import sys

# 35B expert layout: hidden=2048, moe_intermediate=512, 4-bit, group_size=64
# gate_proj: [512, 2048] at 4-bit
#   weight: 512*2048*4/8 = 524288 bytes (U32 packed)
#   scales: 512*2048/64*2 = 32768 bytes (BF16)
#   biases: 32768 bytes (BF16)
# up_proj: same as gate_proj
# down_proj: [2048, 512] at 4-bit
#   weight: 2048*512*4/8 = 524288 bytes
#   scales: 2048*512/64*2 = 32768 bytes
#   biases: 32768 bytes

COMPONENTS = [
    {"name": "gate_proj.weight",  "offset": 0,        "size": 524288, "dtype": "U32",  "shape": [512, 512]},
    {"name": "gate_proj.scales",  "offset": 524288,    "size": 32768,  "dtype": "BF16", "shape": [512, 32]},
    {"name": "gate_proj.biases",  "offset": 557056,    "size": 32768,  "dtype": "BF16", "shape": [512, 32]},
    {"name": "up_proj.weight",    "offset": 589824,    "size": 524288, "dtype": "U32",  "shape": [512, 512]},
    {"name": "up_proj.scales",    "offset": 1114112,   "size": 32768,  "dtype": "BF16", "shape": [512, 32]},
    {"name": "up_proj.biases",    "offset": 1146880,   "size": 32768,  "dtype": "BF16", "shape": [512, 32]},
    {"name": "down_proj.weight",  "offset": 1179648,   "size": 524288, "dtype": "U32",  "shape": [2048, 128]},
    {"name": "down_proj.scales",  "offset": 1703936,   "size": 32768,  "dtype": "BF16", "shape": [2048, 8]},
    {"name": "down_proj.biases",  "offset": 1736704,   "size": 32768,  "dtype": "BF16", "shape": [2048, 8]},
]

EXPERT_SIZE = 1769472   # bytes per expert
NUM_EXPERTS = 256
NUM_LAYERS = 40
LAYER_SIZE = NUM_EXPERTS * EXPERT_SIZE  # 452,984,832 bytes (~432 MB)


def parse_layers(spec):
    if spec is None or spec == 'all':
        return list(range(NUM_LAYERS))
    layers = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_index(index_path):
    with open(index_path) as f:
        idx = json.load(f)
    return idx['expert_reads'], idx['model_path']


def verify_component_sizes(expert_reads):
    expected = {c['name']: c['size'] for c in COMPONENTS}
    for layer_key, comps in expert_reads.items():
        for comp_name, info in comps.items():
            if comp_name not in expected:
                print(f"WARNING: unknown component {comp_name}")
                continue
            if info['expert_size'] != expected[comp_name]:
                print(f"MISMATCH: layer {layer_key}, {comp_name}: "
                      f"index={info['expert_size']}, expected={expected[comp_name]}")
                return False
    print("Component sizes verified OK")
    return True


def open_source_files(expert_reads, model_path, layers):
    needed_files = set()
    for layer_idx in layers:
        layer_key = str(layer_idx)
        if layer_key not in expert_reads:
            continue
        for info in expert_reads[layer_key].values():
            needed_files.add(info['file'])

    fds = {}
    for fname in sorted(needed_files):
        path = os.path.join(model_path, fname)
        fds[fname] = os.open(path, os.O_RDONLY)
    print(f"Opened {len(fds)} source files")
    return fds


def repack_layer(layer_idx, expert_reads, model_path, fds, output_dir, dry_run=False):
    layer_key = str(layer_idx)
    if layer_key not in expert_reads:
        print(f"  Layer {layer_idx}: NOT FOUND, skipping")
        return 0, 0.0

    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if dry_run:
        for expert_idx in range(NUM_EXPERTS):
            for comp in COMPONENTS:
                info = layer_info[comp['name']]
                _ = info['abs_offset'] + expert_idx * info['expert_stride']
        print(f"  Layer {layer_idx:2d}: DRY RUN OK — {LAYER_SIZE:,} bytes")
        return LAYER_SIZE, 0.0

    t0 = time.monotonic()
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, LAYER_SIZE)

    read_plan = []
    for expert_idx in range(NUM_EXPERTS):
        for comp in COMPONENTS:
            info = layer_info[comp['name']]
            src_fd = fds[info['file']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * EXPERT_SIZE + comp['offset']
            read_plan.append((src_fd, src_offset, dst_offset, comp['size']))

    read_plan.sort(key=lambda x: (x[0], x[1]))

    bytes_written = 0
    for src_fd, src_offset, dst_offset, size in read_plan:
        data = os.pread(src_fd, size, src_offset)
        if len(data) != size:
            raise IOError(f"Short read: expected {size}, got {len(data)}")
        os.pwrite(fd_out, data, dst_offset)
        bytes_written += size

    os.close(fd_out)
    elapsed = time.monotonic() - t0
    return bytes_written, elapsed


def verify_layer(layer_idx, expert_reads, model_path, fds, output_dir):
    layer_key = str(layer_idx)
    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if not os.path.exists(out_path):
        print(f"  Layer {layer_idx}: packed file not found")
        return False

    fd_packed = os.open(out_path, os.O_RDONLY)
    mismatches = 0
    for expert_idx in [0, 1, 127, 255]:
        for comp in COMPONENTS:
            info = layer_info[comp['name']]
            src_fd = fds[info['file']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * EXPERT_SIZE + comp['offset']

            original = os.pread(src_fd, comp['size'], src_offset)
            packed = os.pread(fd_packed, comp['size'], dst_offset)

            if original != packed:
                print(f"  MISMATCH: layer {layer_idx}, expert {expert_idx}, {comp['name']}")
                mismatches += 1

    os.close(fd_packed)
    if mismatches == 0:
        print(f"  Layer {layer_idx}: PASSED (experts 0, 1, 127, 255)")
    return mismatches == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True, help='Path to expert_index.json')
    parser.add_argument('--layers', default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verify-only', type=int, default=None)
    args = parser.parse_args()

    print("Loading expert index...")
    expert_reads, model_path = load_index(args.index)
    print(f"Model: {model_path}")
    print(f"Layers: {len(expert_reads)}")

    if not verify_component_sizes(expert_reads):
        sys.exit(1)

    output_dir = os.path.join(model_path, "packed_experts")
    os.makedirs(output_dir, exist_ok=True)

    if args.verify_only is not None:
        layers = [args.verify_only]
    else:
        layers = parse_layers(args.layers)

    print(f"Processing layers: {layers[0]}-{layers[-1]} ({len(layers)} layers)")

    fds = open_source_files(expert_reads, model_path, layers)

    if args.verify_only is not None:
        verify_layer(args.verify_only, expert_reads, model_path, fds, output_dir)
        for fd in fds.values():
            os.close(fd)
        return

    t_start = time.monotonic()
    total_written = 0

    for i, layer_idx in enumerate(layers):
        bytes_written, elapsed = repack_layer(
            layer_idx, expert_reads, model_path, fds, output_dir, args.dry_run
        )
        total_written += bytes_written

        if not args.dry_run and bytes_written > 0:
            throughput = bytes_written / elapsed / (1024**3) if elapsed > 0 else 0
            overall_elapsed = time.monotonic() - t_start
            eta = (len(layers) - i - 1) * (overall_elapsed / (i + 1))
            print(f"  Layer {layer_idx:2d}: {bytes_written/1024**2:.0f} MB in {elapsed:.1f}s "
                  f"({throughput:.1f} GB/s) | ETA: {eta:.0f}s")

            if not verify_layer(layer_idx, expert_reads, model_path, fds, output_dir):
                sys.exit(1)

    for fd in fds.values():
        os.close(fd)

    total_elapsed = time.monotonic() - t_start
    if not args.dry_run and total_written > 0:
        print(f"\nDONE: {total_written/1024**3:.1f} GB in {total_elapsed:.1f}s")
        print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
