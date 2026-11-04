#!/usr/bin/env python
"""Check that torch.compile keeps the served actions, on held-out windows with the same noise.

Compares the compiled actions with the plain ones, against the gap from new noise as a scale, and
prints the distance of both to the demonstration and the latency of both.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SEED = 100008


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wan", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/marvin/train_joint.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    from hydra.utils import instantiate
    from vpp2.cli import load_config
    from vpp2.marvin_policy import MarvinPolicy, PromptTable, compile_model, load_runtime

    cfg = load_config(args.config, [
        f"paths.wan={args.wan}", "paths.video_init=unused", "paths.action_init=unused",
        f"paths.data={args.prepared}", f"paths.media={args.media_root}",
        f"data.train.metadata_path={args.prepared / 'val.csv'}", "data.train.is_training_set=false",
    ])
    dataset = instantiate(cfg.data.train)
    model, processor, step = load_runtime(args.checkpoint, config=args.config, wan=args.wan,
                                          prepared=args.prepared, device=args.device)
    print(f"loaded joint weights from step {step}", flush=True)
    policy = MarvinPolicy(model, processor, PromptTable(args.prepared), steps=args.steps)

    lengths = np.asarray(dataset.trajectory_lengths)
    prefix = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    rng = np.random.default_rng(SEED)
    windows = []
    for row_index in rng.choice(len(lengths), args.windows, replace=False):
        step_index = int(rng.integers(0, max(1, int(lengths[row_index]) - 32)))
        windows.append(dataset._get(int(prefix[row_index]) + step_index))

    def sample_all(offset):
        chunks, seconds = [], []
        for index, item in enumerate(windows):
            started = time.perf_counter()
            chunks.append(policy.sample(item["condition_video"], item["proprio"], item["context"], SEED + index + offset))
            seconds.append(time.perf_counter() - started)
        return chunks, float(np.mean(seconds[1:]))

    plain, plain_seconds = sample_all(0)
    other, _ = sample_all(args.windows)
    compile_model(model)
    started = time.perf_counter()
    policy.sample(windows[0]["condition_video"], windows[0]["proprio"], windows[0]["context"], SEED)
    print(f"compiled in {time.perf_counter() - started:.0f} s", flush=True)
    compiled, compiled_seconds = sample_all(0)

    rows = []
    for item, a, b, c in zip(windows, plain, compiled, other):
        valid = ~item["action_is_pad"].numpy().astype(bool)
        rows.append(dict(
            compiled_l1=float((b - a)[valid].abs().mean()),
            other_noise_l1=float((c - a)[valid].abs().mean()),
            plain_to_truth_l1=float((a - item["action"])[valid].abs().mean()),
            compiled_to_truth_l1=float((b - item["action"])[valid].abs().mean()),
        ))
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    summary.update(compiled_l1_max_window=max(row["compiled_l1"] for row in rows),
                   plain_seconds=plain_seconds, compiled_seconds=compiled_seconds)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
