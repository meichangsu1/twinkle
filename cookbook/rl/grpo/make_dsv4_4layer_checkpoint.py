#!/usr/bin/env python3
"""Create a four-layer DeepSeek-V4 checkpoint without loading the full model."""

import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
MTP_RE = re.compile(r"(?:^|\.)(?:mtp|nextn_predict_layers)(?:\.|$)")
DSPARK_RE = re.compile(r"(?:^|\.)dspark(?:\.|_|$)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--layers", type=int, default=4)
    return parser.parse_args()


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def keep_weight(name, layers):
    if MTP_RE.search(name) or DSPARK_RE.search(name):
        return False
    match = LAYER_RE.search(name)
    return match is None or int(match.group(1)) < layers


def source_shards(source):
    for stem in ("model", "quant_model_weights"):
        index_path = source / f"{stem}.safetensors.index.json"
        if index_path.is_file():
            index = read_json(index_path)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Invalid weight_map: {index_path}")
            shards = sorted(set(weight_map.values()))
            if any(not (source / name).is_file() for name in shards):
                raise FileNotFoundError(f"Index references a missing shard: {index_path}")
            return stem, shards
        single = source / f"{stem}.safetensors"
        if single.is_file():
            return stem, [single.name]
        shards = sorted(path.name for path in source.glob(f"{stem}-*.safetensors"))
        if shards:
            return stem, shards
    raise FileNotFoundError(f"No safetensors checkpoint found in {source}")


def copy_auxiliary_files(source, destination):
    excluded = {"config.json", "quant_model_description.json"}
    for entry in source.iterdir():
        if (entry.name in excluded or entry.name.endswith(".safetensors.index.json")
                or entry.is_file() and entry.suffix == ".safetensors"):
            continue
        target = destination / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        elif entry.is_file():
            shutil.copy2(entry, target)


def convert_config(source, destination, layers):
    config = read_json(source / "config.json")
    original_layers = int(config["num_hidden_layers"])
    if not 0 < layers <= original_layers:
        raise ValueError(f"--layers must be in [1, {original_layers}]")
    config["num_hidden_layers"] = layers
    config["num_nextn_predict_layers"] = 0
    config["n_mtp_layers"] = 0
    for name in ("layer_types", "compress_ratios", "mlp_layer_types"):
        value = config.get(name)
        if isinstance(value, list):
            config[name] = value[:layers]
    for name in list(config):
        if name.startswith("dspark_"):
            config.pop(name)
    write_json(destination / "config.json", config)
    return original_layers


def filter_quant_description(source, destination, layers):
    path = source / "quant_model_description.json"
    if not path.is_file():
        return
    description = read_json(path)
    filtered = {
        key: value
        for key, value in description.items()
        if not isinstance(key, str) or keep_weight(key, layers)
    }
    write_json(destination / path.name, filtered)


def convert_weights(source, destination, layers):
    stem, shards = source_shards(source)
    output_names = []
    output_keys = []
    output_sizes = []

    for source_name in shards:
        source_path = source / source_name
        with safe_open(str(source_path), framework="pt", device="cpu") as handle:
            keys = [key for key in handle.keys() if keep_weight(key, layers)]
            if not keys:
                continue
            tensors = {key: handle.get_tensor(key) for key in keys}
            size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        output_names.append("")
        output_keys.append(keys)
        output_sizes.append(size)
        output_names[-1] = f"{stem}-{len(output_names):05d}-of-PLACEHOLDER.safetensors"
        save_file(tensors, str(destination / output_names[-1]), metadata={"format": "pt"})
        del tensors

    count = len(output_names)
    if count == 0:
        raise RuntimeError("No tensors were retained")
    weight_map = {}
    for index, temporary_name in enumerate(output_names, start=1):
        final_name = f"{stem}-{index:05d}-of-{count:05d}.safetensors"
        (destination / temporary_name).rename(destination / final_name)
        for key in output_keys[index - 1]:
            weight_map[key] = final_name
    write_json(
        destination / f"{stem}.safetensors.index.json",
        {"metadata": {"total_size": sum(output_sizes)}, "weight_map": weight_map},
    )
    return len(weight_map), sum(output_sizes), count


def main():
    args = parse_args()
    source = args.source.expanduser().resolve(strict=True)
    destination = args.destination.expanduser().resolve()
    if source == destination:
        raise ValueError("source and destination must differ")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.mkdir(parents=True)
    try:
        original_layers = convert_config(source, destination, args.layers)
        copy_auxiliary_files(source, destination)
        filter_quant_description(source, destination, args.layers)
        tensors, size, shards = convert_weights(source, destination, args.layers)
        manifest = {
            "source": str(source),
            "num_hidden_layers": args.layers,
            "source_num_hidden_layers": original_layers,
            "mtp_removed": True,
            "dspark_removed": True,
            "output_tensors": tensors,
            "output_shards": shards,
            "total_size": size,
        }
        write_json(destination / "four_layer_manifest.json", manifest)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    print(f"Created {args.layers}-layer checkpoint: {destination}")
    print(f"Tensors: {tensors}, shards: {shards}, size: {size / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
