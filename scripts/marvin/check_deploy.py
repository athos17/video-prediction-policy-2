#!/usr/bin/env python
"""Check that the Marvin server feeds the model what training and evaluation fed it.

Rebuilds held-out windows from the same clips through the wire format (three 320x240 PNGs, a 62-D
state) and compares them with the dataset's tensors. Prints the pixel difference of the condition,
the proprioception difference and the denormalization round trip. With --checkpoint it also samples
both inputs with the same noise and compares the actions, against the hold-pose gap as a scale.
"""

import argparse
import base64
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SEED = 100008


def to_png(view):
    import cv2

    ok, encoded = cv2.imencode(".png", cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(encoded).decode("ascii")


def wire_images(canvas):
    """The client's three views from one 880x480 training canvas, through real PNG decoding."""
    import cv2
    from vpp2.marvin_protocol import decode_observation

    views = (canvas[:, :640], canvas[:240, 640:], canvas[240:, 640:])
    keys = ("head_camera", "left_wrist_camera", "right_wrist_camera")
    images = {key: to_png(cv2.resize(view, (320, 240), interpolation=cv2.INTER_AREA))
              for key, view in zip(keys, views)}
    state = np.zeros(62)
    state[3:9] = state[34:40] = [1, 0, 0, 0, 1, 0]
    return decode_observation(dict(images=images, prompt="check", state=state.tolist()))[0]["images"]


def read_frames(path, indices):
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames, position = {}, 0
    wanted = set(indices)
    while len(frames) < len(wanted):
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"{path} ended at {position}")
        if position in wanted:
            frames[position] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        position += 1
    capture.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--wan", type=Path, default=Path("unused"))
    parser.add_argument("--config", type=Path, default=Path("configs/marvin/train_joint.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    import torch
    from hydra.utils import instantiate
    from vpp2.cli import load_config
    from vpp2.marvin_policy import (
        MarvinPolicy,
        NUM_VIDEO_FRAMES,
        PromptTable,
        denormalize_action,
        load_processor,
        load_runtime,
        normalize_state,
        prepare_frame,
    )
    from vpp2.marvin_protocol import condition_steps

    csv = args.prepared / "val.csv"
    cfg = load_config(args.config, [
        f"paths.wan={args.wan}", "paths.video_init=unused", "paths.action_init=unused",
        f"paths.data={args.prepared}", f"paths.media={args.media_root}",
        f"data.train.metadata_path={csv}", "data.train.is_training_set=false",
    ])
    dataset = instantiate(cfg.data.train)
    processor = load_processor(cfg, args.prepared)
    prompts = PromptTable(args.prepared)
    model = None
    if args.checkpoint:
        model, processor, step = load_runtime(args.checkpoint, config=args.config, wan=args.wan,
                                              prepared=args.prepared, device=args.device)
        print(f"loaded joint weights from step {step}", flush=True)
    policy = MarvinPolicy(model, processor, prompts, steps=args.steps)

    lengths = np.asarray(dataset.trajectory_lengths)
    prefix = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    rng = np.random.default_rng(SEED)
    rows = []
    for row_index in rng.choice(len(lengths), args.windows, replace=False):
        row = dataset.metadata.iloc[int(row_index)]
        step = int(rng.integers(0, max(1, int(lengths[row_index]) - 32)))
        item = dataset._get(int(prefix[row_index]) + step)
        if (int(item["trajectory_idx"]), int(item["step_idx"])) != (int(row_index), step):
            raise RuntimeError("The dataset resolved another window")
        if int(item["video_num_frames"]) != NUM_VIDEO_FRAMES:
            raise ValueError(f"Expected {NUM_VIDEO_FRAMES} video frames, got {int(item['video_num_frames'])}")
        steps = condition_steps(step)
        offsets = [0, *(np.maximum(item["condition_history_offsets"].numpy() + step, 0).tolist())]
        if steps != offsets:
            raise ValueError(f"History grid differs: {steps} vs {offsets}")
        canvases = read_frames(args.media_root / row.video_path, steps)
        wire = torch.stack([prepare_frame(wire_images(canvases[s])) for s in steps], dim=1)
        table = pd.read_parquet(args.media_root / row.parquet_path)
        state = np.asarray(table["observation.state"].iloc[step], dtype=np.float32)
        proprio = normalize_state(processor, state)
        context = prompts.context(row.instruction)
        stored, mask = dataset._load_text_context(row)
        stored[~mask] = 0.0
        truth = denormalize_action(processor, item["action"], item["proprio"])
        real = np.stack(table["action"].iloc[step : step + 32].to_numpy()).astype(np.float32)
        result = dict(
            case=f"ep{row.episode_index:03d}_seg{row.seg_id:03d}_s{step}",
            condition_l1_255=float((wire - item["condition_video"]).abs().mean()) * 127.5,
            head_l1_255=float((wire - item["condition_video"])[..., :302].abs().mean()) * 127.5,
            wrist_l1_255=float((wire - item["condition_video"])[..., 303:].abs().mean()) * 127.5,
            proprio_max_error=float((proprio - item["proprio"]).abs().max()),
            context_max_error=float((context.float() - stored.float()).abs().max()),
            denormalize_max_error=float(np.abs(truth[: len(real)] - real).max()),
        )
        if model is not None:
            seed = SEED + len(rows)
            reference = policy.sample(item["condition_video"], item["proprio"], item["context"], seed)
            served = policy.sample(wire, proprio, context, seed)
            valid = ~item["action_is_pad"].numpy().astype(bool)
            other = policy.sample(item["condition_video"], item["proprio"], item["context"], seed + 1)
            result.update(
                action_l1=float((served - reference)[valid].abs().mean()),
                other_noise_l1=float((other - reference)[valid].abs().mean()),
                to_truth_l1=float((reference - item["action"])[valid].abs().mean()),
                served_to_truth_l1=float((served - item["action"])[valid].abs().mean()),
            )
        rows.append(result)
        print(json.dumps(result), flush=True)
    keys = [key for key in rows[0] if key != "case"]
    print(json.dumps({key: float(np.mean([row[key] for row in rows])) for key in keys}, indent=2))


if __name__ == "__main__":
    main()
