#!/usr/bin/env python
"""Counterfactual instruction following of the actions of a Marvin joint checkpoint.

Uses the windows and candidates of gate1.py. For every candidate prompt and seed the policy
samples the 32-step action chunk from the same history, proprioception and initial noise.

  sample  Write one npz per window. Run one process per GPU with --rank/--world.
  report  Compare the chunks with the demonstration and with each other.

Without --checkpoint, it evaluates the initial weights given by --video and --action.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gate1 import bootstrap, build_cases, rate  # noqa: E402

HORIZON = 32
# [left EEF xyz+rot6d (9), left hand (22), right EEF xyz+rot6d (9), right hand (22)].
XYZ = dict(left=slice(0, 3), right=slice(31, 34))
OTHER = dict(left="right", right="left")


def load_runtime(args):
    import torch
    from hydra.utils import instantiate
    from vpp2.cli import load_config

    cfg = load_config(args.config, [
        f"paths.wan={args.wan}",
        f"paths.video_init={args.video}",
        f"paths.action_init={args.action}",
        f"paths.data={args.prepared}",
        f"paths.media={args.media_root}",
        f"data.train.metadata_path={args.csv}",
        "data.train.is_training_set=false",
        "model.video_dit_config.use_gradient_checkpointing=false",
        "model.action_dit_config.use_gradient_checkpointing=false",
    ])
    dataset = instantiate(cfg.data.train)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=f"cuda:{args.gpu}")
    if args.checkpoint:
        step = model.load_checkpoint(str(args.checkpoint))
        print(f"loaded joint weights from step {step}", flush=True)
    model.eval().requires_grad_(False)
    return model, dataset


def denormalize(dataset, action, proprio):
    from vpp2.evaluation import _ActionMetrics

    class Owner:
        pass

    owner = Owner()
    owner.dataset = dataset
    return _ActionMetrics._denormalize_action(owner, action, proprio).numpy()


def sample(args):
    import torch

    directory = args.output / f"action_{args.tag}"
    directory.mkdir(parents=True, exist_ok=True)
    model, dataset = load_runtime(args)
    prefix = np.concatenate([[0], np.cumsum(dataset.trajectory_lengths)[:-1]])
    cases = build_cases(args.csv, args.phases, args.max_candidates, args.episodes, args.seed)
    print(f"{len(cases)} cases, this rank takes every {args.world}th", flush=True)
    cases = cases[args.rank :: args.world]
    for case in cases[: args.limit] if args.limit else cases:
        path = directory / f"{case['case']}.npz"
        if path.exists():
            raise FileExistsError(path)
        item = dataset._get(int(prefix[case["row"]]) + case["step"])
        if (int(item["trajectory_idx"]), int(item["step_idx"])) != (case["row"], case["step"]):
            raise RuntimeError(f"Dataset resolved another window for {case['case']}")
        chunks = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for candidate in case["candidates"]:
                context, mask = dataset._load_text_context(dataset.metadata.iloc[candidate["row"]])
                context[~mask] = 0.0
                for offset in range(args.seeds):
                    chunk = model.infer_action(
                        prompt=None,
                        input_image=item["video"][:, 0].unsqueeze(0),
                        action_horizon=HORIZON,
                        condition_video=item["condition_video"],
                        proprio=item["proprio"][0],
                        context=context,
                        context_mask=torch.ones_like(mask),
                        num_video_frames=int(item["video_num_frames"]),
                        num_inference_steps=args.steps,
                        seed=args.seed + offset,
                    )["action"]
                    chunks.append(chunk.detach().to("cpu", torch.float32).reshape(HORIZON, -1))
        proprio = item["proprio"]
        pad = item["action_is_pad"].cpu().numpy().astype(bool)
        np.savez(
            path,
            normalized=torch.stack(chunks).numpy().reshape(len(case["candidates"]), args.seeds, HORIZON, -1),
            actions=np.stack([denormalize(dataset, chunk, proprio) for chunk in chunks]).reshape(
                len(case["candidates"]), args.seeds, HORIZON, -1),
            truth_normalized=item["action"].cpu().float().numpy(),
            truth=denormalize(dataset, item["action"], proprio),
            pad=pad,
            case=json.dumps(case),
        )
        print(case["case"], flush=True)


def travel(chunk, hand, valid):
    """Largest distance of one hand's end effector from where this chunk starts."""
    position = chunk[valid][:, XYZ[hand]]
    return float(np.linalg.norm(position - position[0], axis=1).max())


def endpoint_gap(a, b, valid):
    """Distance between the final end-effector positions of two chunks, worst arm."""
    last = np.flatnonzero(valid)[-1]
    return max(float(np.linalg.norm(a[last, XYZ[hand]] - b[last, XYZ[hand]])) for hand in XYZ)


def case_row(path):
    with np.load(path) as data:
        normalized, actions = data["normalized"], data["actions"]
        truth_normalized, truth, pad = data["truth_normalized"], data["truth"], data["pad"]
        case = json.loads(str(data["case"]))
    valid = ~pad
    hands = [candidate["hand"] for candidate in case["candidates"]]
    count, seeds = normalized.shape[:2]
    distance = np.abs(normalized[:, :, valid] - truth_normalized[valid]).mean(axis=(2, 3)).mean(axis=1)
    same = [index for index, hand in enumerate(hands) if hand == hands[0]]
    row = dict(
        case=case["case"], episode=case["episode"], seg_id=case["seg_id"], phase=case["phase"],
        closest=float(distance.argmin() == 0), closest_chance=1 / count, l1=float(distance[0]),
        truth_hand=float(travel(truth, hands[0], valid) > travel(truth, OTHER[hands[0]], valid)),
    )
    if len(same) > 1:
        row["object"] = float(same[int(distance[same].argmin())] == 0)
        row["object_chance"] = 1 / len(same)
    if len(same) < count:
        row["hand"] = float(hands[int(distance.argmin())] == hands[0])
        row["hand_chance"] = len(same) / count
    moves = {"demonstrated": [], "same_hand": [], "other_hand": []}
    gaps = {"same_hand": [], "other_hand": [], "seed": []}
    rises = []
    for seed in range(seeds):
        for index, hand in enumerate(hands):
            chunk = actions[index, seed]
            kind = "demonstrated" if index == 0 else "same_hand" if hand == hands[0] else "other_hand"
            moves[kind].append(float(travel(chunk, hand, valid) > travel(chunk, OTHER[hand], valid)))
            if index:
                gaps[kind].append(endpoint_gap(actions[0, seed], chunk, valid))
            if kind == "other_hand":
                rises.append(float(travel(chunk, hand, valid) > travel(actions[0, seed], hand, valid)))
    for index in range(count):
        for a in range(seeds):
            for b in range(a + 1, seeds):
                gaps["seed"].append(endpoint_gap(actions[index, a], actions[index, b], valid))
    for kind, values in moves.items():
        if values:
            row[f"moves_{kind}"] = float(np.mean(values))
    for kind, values in gaps.items():
        if values:
            row[f"gap_{kind}"] = float(np.mean(values))
    if rises:
        row["other_hand_rises"] = float(np.mean(rises))
    return row


def mean_of(rows, key):
    values = [row[key] for row in rows if key in row]
    return float(np.mean(values)) if values else None


def report(args):
    summary = {}
    for directory in sorted(args.output.glob("action_*")):
        paths = sorted(directory.glob("*.npz"))
        if not paths:
            continue
        rows = [case_row(path) for path in paths]
        groups = [f"{row['episode']}_{row['seg_id']}" for row in rows]
        item = dict(
            cases=len(rows),
            closest=rate(rows, "closest", "closest_chance"),
            object=rate(rows, "object", "object_chance"),
            hand=rate(rows, "hand", "hand_chance"),
            l1_demonstrated=mean_of(rows, "l1"),
            l1_ci95=bootstrap([row["l1"] for row in rows], groups),
            truth_commanded_hand_moves_more=mean_of(rows, "truth_hand"),
        )
        for key in ("moves_demonstrated", "moves_same_hand", "moves_other_hand", "other_hand_rises",
                    "gap_same_hand", "gap_other_hand", "gap_seed"):
            item[key] = mean_of(rows, key)
        for phase in sorted({row["phase"] for row in rows}):
            item[f"closest_phase_{phase}"] = rate(
                [row for row in rows if row["phase"] == phase], "closest", "closest_chance")
        summary[directory.name] = item
        (args.output / f"{directory.name}_cases.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "action_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("sample", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag")
    parser.add_argument("--checkpoint", type=Path, help="Joint weights; omit to test the initial weights")
    parser.add_argument("--video", type=Path, help="Video weights the joint run started from")
    parser.add_argument("--action", type=Path, help="Action weights the joint run started from")
    parser.add_argument("--config", type=Path, default=Path("configs/marvin/train_joint.yaml"))
    parser.add_argument("--wan", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--csv", type=Path, help="Defaults to val.csv under --prepared")
    parser.add_argument("--episodes", type=int, help="Sample this many episodes from the CSV")
    parser.add_argument("--phases", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--steps", type=int, default=10, help="Action denoising steps, as in evaluation")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=100008)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    args = parser.parse_args()
    if args.command == "report":
        return report(args)
    if not all((args.tag, args.video, args.action, args.wan, args.prepared, args.media_root)):
        parser.error("sample needs --tag --video --action --wan --prepared --media-root")
    args.csv = args.csv or args.prepared / "val.csv"
    return sample(args)


if __name__ == "__main__":
    main()
