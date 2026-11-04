#!/usr/bin/env python
"""Counterfactual instruction following of a history-conditioned Marvin Video checkpoint.

Each case is one window of a PICK subtask whose episode still has other objects on the
table. The candidates are the demonstrated (hand, object) and the ones picked later.

  score     Denoising error of the real future under every candidate prompt, with shared
            noise and timesteps. The demonstrated prompt should explain it best.
  generate  Sample a future per candidate and seed, as the policy would (no CFG).
  report    Summarize both and draw contact sheets.

Run one process per GPU with --rank/--world, then `report` once on the output directory.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

CONDITION_LATENTS = 3
CONDITION_FRAMES = 9
FUTURE_NATIVE_STEPS = 64
# 880x480 composite resized to 416x240: the wrists fill x >= 302.5, left on top.
WRIST_X = 306
LEFT_WRIST = (2, 118)
RIGHT_WRIST = (122, 238)
SHEET_FRAMES = (8, 12, 16, 20, 24)
SHEET_CELL = (208, 120)


def build_cases(csv, phases, max_candidates, episodes=None, seed=0):
    """Return windows; candidate 0 is demonstrated, the rest are still on the table."""
    frame = pd.read_csv(csv, keep_default_na=False, na_values=[""])
    available = sorted(frame.episode_index.unique().tolist())
    if episodes is not None and episodes < len(available):
        available = sorted(np.random.default_rng(seed).choice(available, episodes, replace=False))
    picks = frame[(frame.segment_type == "PICK") & frame.episode_index.isin(available)]
    cases = []
    for episode, group in picks.groupby("episode_index"):
        rows = list(group.sort_values("source_start").itertuples())
        for position, row in enumerate(rows):
            candidates, seen = [], set()
            for later in rows[position:]:
                key = (later.hand, later.object)
                if key not in seen and len(candidates) < max_candidates:
                    seen.add(key)
                    candidates.append(dict(row=int(later.Index), hand=later.hand, object=later.object))
            if len(candidates) < 2:
                continue
            for phase in phases:
                step = int(phase * row.grasp)
                if step + FUTURE_NATIVE_STEPS >= row.num_frames:
                    continue
                cases.append(dict(
                    case=f"ep{episode:03d}_seg{row.seg_id:03d}_p{int(phase * 100):03d}",
                    episode=int(episode), seg_id=int(row.seg_id), phase=float(phase), step=step,
                    row=int(row.Index), grasp=int(row.grasp), candidates=candidates,
                ))
    return cases


def load_runtime(args):
    import torch
    from hydra.utils import instantiate
    from vpp2.cli import load_config
    from vpp2.models.wan21_14b.vpp2 import VPP2

    cfg = load_config(args.config, [
        f"paths.wan={args.wan}",
        f"paths.video_init={args.checkpoint}",
        f"paths.data={args.prepared}",
        f"paths.media={args.media_root}",
        f"data.train.metadata_path={args.csv}",
        "data.train.is_training_set=false",
        "model.video_dit_config.use_gradient_checkpointing=false",
    ])
    dataset = instantiate(cfg.data.train)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=f"cuda:{args.gpu}")
    if args.joint:
        model.dit.load_state_dict(VPP2._load_video_checkpoint_state(args.joint), strict=True)
        print(f"loaded Video weights from {args.joint}", flush=True)
    model.eval().requires_grad_(False)
    return model, dataset


def case_batch(dataset, case, device):
    """Load a window exactly as training does, without history augmentation."""
    import torch

    prefix = np.concatenate([[0], np.cumsum(dataset.trajectory_lengths)[:-1]])
    sample = dataset[int(prefix[case["row"]]) + case["step"]]
    if (int(sample["trajectory_idx"]), int(sample["step_idx"])) != (case["row"], case["step"]):
        raise RuntimeError(f"Dataset resolved another window for {case['case']}")
    if bool(sample["image_is_pad"][CONDITION_FRAMES:].any()):
        raise RuntimeError(f"Future frames are padded for {case['case']}")
    keys = ("video", "video_num_frames", "context", "context_mask", "image_is_pad", "condition_video")
    batch = {key: sample[key].unsqueeze(0).to(device) for key in keys}
    contexts = []
    for candidate in case["candidates"]:
        context, mask = dataset._load_text_context(dataset.metadata.iloc[candidate["row"]])
        context[~mask] = 0.0
        contexts.append(context)
    return batch, torch.stack(contexts).to(device)


def truth_frames(batch):
    import torch

    truth = batch["video"][0].permute(1, 2, 3, 0).float()
    return ((truth.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()


def sample_video(model, batch, context, num_frames, steps, seed):
    import torch

    condition = batch["condition_video"]
    result = model.infer(
        prompt=None,
        input_image=condition[:, :, 0],
        num_frames=num_frames,
        context=context[None].to(model.torch_dtype),
        context_mask=torch.ones((1, context.shape[0]), dtype=torch.bool, device=model.device),
        num_inference_steps=steps,
        seed=seed,
        condition_video=condition,
    )["video"]
    return np.stack([np.asarray(frame) for frame in result])


def shard(args):
    cases = build_cases(args.csv, args.phases, args.max_candidates, args.episodes, args.seed)
    print(f"{len(cases)} cases, this rank takes every {args.world}th", flush=True)
    selected = cases[args.rank :: args.world]
    return selected[: args.limit] if args.limit else selected


def score(args):
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"score_{args.tag}_{args.rank}.jsonl"
    if path.exists():
        raise FileExistsError(path)
    model, dataset = load_runtime(args)
    device = model.device
    scheduler = model.train_video_scheduler
    sigmas = scheduler._phi((torch.arange(args.timesteps, dtype=torch.float32) + 0.5) / args.timesteps,
                            scheduler.shift)
    with path.open("w") as output, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, case in enumerate(shard(args)):
            batch, contexts = case_batch(dataset, case, device)
            inputs = model.build_inputs(batch)
            latents = inputs["input_latents"]
            count = len(contexts)
            generator = torch.Generator(device=device).manual_seed(args.seed + args.rank * 100000 + index)
            errors = []
            for sigma in sigmas:
                timestep = (sigma * scheduler.num_train_timesteps).to(device=device, dtype=latents.dtype)
                noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                noisy = scheduler.add_noise(latents, noise, timestep)
                target = scheduler.training_target(latents, noise, timestep)
                prediction = model.dit(
                    x=noisy.expand(count, -1, -1, -1, -1),
                    timestep=timestep.expand(count),
                    context=contexts.to(latents.dtype),
                    context_mask=torch.ones(contexts.shape[:2], dtype=torch.bool, device=device),
                    action=None,
                    clip_fea=inputs["clip_fea"].expand(count, -1, -1),
                    condition_latents=inputs["condition_latents"].expand(count, -1, -1, -1, -1),
                )
                squared = (prediction.float() - target.float()).pow(2)
                errors.append(squared[:, :, CONDITION_LATENTS:].mean(dim=(1, 2, 3, 4)).cpu().tolist())
            output.write(json.dumps({**case, "sigmas": sigmas.tolist(), "errors": errors}) + "\n")
            output.flush()
            print(f"[{index}] {case['case']} candidates={count}", flush=True)


def generate(args):
    import torch

    directory = args.output / f"generate_{args.tag}"
    directory.mkdir(parents=True, exist_ok=True)
    model, dataset = load_runtime(args)
    device = model.device
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for case in shard(args):
            path = directory / f"{case['case']}.npz"
            if path.exists():
                raise FileExistsError(path)
            batch, contexts = case_batch(dataset, case, device)
            truth = truth_frames(batch)
            videos = []
            for candidate, context in enumerate(contexts):
                for offset in range(args.seeds):
                    # Every candidate shares the initial noise of its seed.
                    videos.append(sample_video(model, batch, context, truth.shape[0], args.steps, args.seed + offset))
                print(f"{case['case']} candidate {candidate}", flush=True)
            shape = (len(contexts), args.seeds, *videos[0].shape)
            np.savez(path, truth=truth, videos=np.stack(videos).reshape(shape),
                     case=json.dumps(case))


def bootstrap(values, groups, seed=0, draws=2000):
    """95% interval over segments, because the windows of one segment are correlated."""
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    members = [values[groups == group] for group in np.unique(groups)]
    rng = np.random.default_rng(seed)
    means = [
        np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))]).mean()
        for _ in range(draws)
    ]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def rate(rows, key, chance_key):
    rows = [row for row in rows if row.get(key) is not None]
    if not rows:
        return None
    values = [row[key] for row in rows]
    groups = [f"{row['episode']}_{row['seg_id']}" for row in rows]
    return dict(
        cases=len(rows), rate=float(np.mean(values)), ci95=bootstrap(values, groups),
        chance=float(np.mean([row[chance_key] for row in rows])),
    )


def score_row(case):
    errors = np.asarray(case["errors"])
    mean = errors.mean(axis=0)
    hands = [candidate["hand"] for candidate in case["candidates"]]
    same_hand = [i for i, hand in enumerate(hands) if hand == hands[0]]
    row = dict(
        case=case["case"], episode=case["episode"], seg_id=case["seg_id"], phase=case["phase"],
        top1=float(mean.argmin() == 0), chance=1 / len(mean),
        margin=float((np.delete(mean, 0).mean() - mean[0]) / mean[0]),
        object_top1=None, object_chance=None, hand_top1=None, hand_chance=None,
    )
    if len(same_hand) > 1:
        row["object_top1"] = float(same_hand[int(mean[same_hand].argmin())] == 0)
        row["object_chance"] = 1 / len(same_hand)
    other_hand = [i for i in range(len(mean)) if i not in same_hand]
    if other_hand:
        row["hand_top1"] = float(hands[int(mean.argmin())] == hands[0])
        row["hand_chance"] = len(same_hand) / len(mean)
    return row


def summarize_scores(rows):
    summary = dict(
        cases=len(rows),
        top1=rate(rows, "top1", "chance"),
        object_top1=rate(rows, "object_top1", "object_chance"),
        hand_top1=rate(rows, "hand_top1", "hand_chance"),
        relative_margin=float(np.mean([row["margin"] for row in rows])),
    )
    summary["by_phase"] = {
        str(phase): rate([row for row in rows if row["phase"] == phase], "top1", "chance")
        for phase in sorted({row["phase"] for row in rows})
    }
    return summary


def wrist_motion(video):
    """Mean future change of each wrist panel from the current frame: (left, right)."""
    future = video[CONDITION_FRAMES:].astype(np.float32)
    current = video[CONDITION_FRAMES - 1].astype(np.float32)
    change = np.abs(future - current).mean(axis=(0, 3))
    return (float(change[LEFT_WRIST[0] : LEFT_WRIST[1], WRIST_X:].mean()),
            float(change[RIGHT_WRIST[0] : RIGHT_WRIST[1], WRIST_X:].mean()))


def commanded_wrist_moves_more(video, hand):
    left, right = wrist_motion(video)
    return float((left > right) == (hand == "left"))


def generated_row(path):
    with np.load(path) as data:
        truth, videos, case = data["truth"], data["videos"], json.loads(str(data["case"]))
    future = videos[:, :, CONDITION_FRAMES:].astype(np.float32)
    real = truth[CONDITION_FRAMES:].astype(np.float32)
    distance = np.abs(future - real).mean(axis=(2, 3, 4, 5)).mean(axis=1)
    seeds = videos.shape[1]
    prompt_gap = np.mean([
        np.abs(future[0, s] - future[c, s]).mean()
        for c in range(1, len(future)) for s in range(seeds)
    ])
    seed_gap = np.mean([
        np.abs(future[c, a] - future[c, b]).mean()
        for c in range(len(future)) for a in range(seeds) for b in range(a + 1, seeds)
    ]) if seeds > 1 else float("nan")
    hand = case["candidates"][0]["hand"]
    hands = [candidate["hand"] for candidate in case["candidates"]]
    return dict(
        case=case["case"], episode=case["episode"], seg_id=case["seg_id"], phase=case["phase"],
        top1=float(distance.argmin() == 0), chance=1 / len(distance),
        prompt_gap=float(prompt_gap), seed_gap=float(seed_gap),
        hand_commanded=float(np.mean([
            commanded_wrist_moves_more(videos[c, s], hands[c])
            for c in range(len(hands)) for s in range(seeds)
        ])),
        hand_truth=commanded_wrist_moves_more(truth, hand),
        l1_demonstrated=float(distance[0]),
    )


def draw_case(path):
    with np.load(path) as data:
        truth, videos, case = data["truth"], data["videos"], json.loads(str(data["case"]))
    labels = [f"{case['case']}  ground truth: {case['candidates'][0]['hand']} {case['candidates'][0]['object']}"]
    rows = [truth]
    for index, candidate in enumerate(case["candidates"]):
        mark = "demonstrated" if index == 0 else "counterfactual"
        labels.append(f"{mark}: {candidate['hand']} {candidate['object']}")
        rows.append(videos[index, 0])
    width, height = SHEET_CELL
    sheet = Image.new("RGB", (width * len(SHEET_FRAMES), (height + 16) * len(rows)), "white")
    pen = ImageDraw.Draw(sheet)
    for row, (label, video) in enumerate(zip(labels, rows)):
        top = row * (height + 16)
        pen.text((3, top + 2), label, fill="black")
        for column, frame in enumerate(SHEET_FRAMES):
            image = Image.fromarray(video[frame]).resize(SHEET_CELL, Image.Resampling.BICUBIC)
            sheet.paste(image, (column * width, top + 16))
    return sheet


def report(args):
    summary = {}
    for tag in sorted({path.stem[6:].rsplit("_", 1)[0] for path in args.output.glob("score_*.jsonl")}):
        rows = []
        for path in sorted(args.output.glob(f"score_{tag}_[0-9]*.jsonl")):
            rows += [score_row(json.loads(line)) for line in path.read_text().splitlines()]
        summary[f"score/{tag}"] = summarize_scores(rows)
    for directory in sorted(args.output.glob("generate_*")):
        paths = sorted(directory.glob("*.npz"))
        rows = [generated_row(path) for path in paths]
        if not rows:
            continue
        ratio = [row["prompt_gap"] / row["seed_gap"] for row in rows]
        summary[directory.name] = dict(
            cases=len(rows),
            closest_to_truth=rate(rows, "top1", "chance"),
            prompt_gap=float(np.mean([row["prompt_gap"] for row in rows])),
            seed_gap=float(np.mean([row["seed_gap"] for row in rows])),
            prompt_over_seed_median=float(np.median(ratio)),
            commanded_wrist_moves_more=float(np.mean([row["hand_commanded"] for row in rows])),
            commanded_wrist_moves_more_truth=float(np.mean([row["hand_truth"] for row in rows])),
            l1_demonstrated=float(np.mean([row["l1_demonstrated"] for row in rows])),
        )
        (args.output / f"{directory.name}_cases.json").write_text(json.dumps(rows, indent=2) + "\n")
        sheets = args.output / "sheets" / directory.name
        sheets.mkdir(parents=True, exist_ok=True)
        for path in paths:
            draw_case(path).save(sheets / f"{path.stem}.jpg", quality=88)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("score", "generate", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", help="Names this checkpoint and split in the output")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--joint", type=Path, help="Replace the Video weights with those of this joint checkpoint")
    parser.add_argument("--config", type=Path, default=Path("configs/marvin/train_video.yaml"))
    parser.add_argument("--wan", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--csv", type=Path, help="Defaults to val.csv under --prepared")
    parser.add_argument("--episodes", type=int, help="Sample this many episodes from the CSV")
    parser.add_argument("--phases", type=float, nargs="+", default=[0.0, 0.5],
                        help="Window starts as a fraction of the way to the grasp")
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30, help="Sampling steps for generate")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=100008)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    args = parser.parse_args()
    if args.command == "report":
        return report(args)
    if not all((args.tag, args.checkpoint, args.wan, args.prepared, args.media_root)):
        parser.error(f"{args.command} needs --tag --checkpoint --wan --prepared --media-root")
    args.csv = args.csv or args.prepared / "val.csv"
    return score(args) if args.command == "score" else generate(args)


if __name__ == "__main__":
    main()
