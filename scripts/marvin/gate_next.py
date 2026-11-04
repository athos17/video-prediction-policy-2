#!/usr/bin/env python
"""Can a Marvin joint checkpoint go for the next object from the current observation?

Each case is a PICK A directly followed by a different PICK B in one episode. From the first
window of A, the policy receives the instruction of A and of B. The future video (2.13 s) and
action chunk (1.07 s) of each prompt are compared with the real starts of A and B. The chunk
ends well before any grasp, so the actions are also compared with where B was really grasped.

  action  Sample the action chunks. Run one process per GPU with --rank/--world.
  video   Sample the videos in the same way.
  report  Summarize both and write one video and one trajectory plot per case.

The real B starts from another scene and arm pose, so generations are compared in pairs against
the same reference, and trajectories are compared as displacements from their own start.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from gate1 import CONDITION_FRAMES, FUTURE_NATIVE_STEPS, WRIST_X, case_batch, rate, sample_video, truth_frames
from gate1 import load_runtime as load_video_runtime
from gate1_web import encode
from gate_action import XYZ, denormalize, sample_chunk
from gate_action import load_runtime as load_action_runtime

A, B = 0, 1
CHANCE = 0.5
# The current frame and the future, as in the Gate 1 site.
FIRST_FRAME = CONDITION_FRAMES - 1
LABEL_HEIGHT = 18
PANEL = 440
COLORS = ("#2a78c8", "#d9622b")


def build_pairs(csv):
    frame = pd.read_csv(csv, keep_default_na=False, na_values=[""])
    pairs = []
    for episode, group in frame[frame.segment_type == "PICK"].groupby("episode_index"):
        rows = list(group.sort_values("source_start").itertuples())
        for a, b in zip(rows, rows[1:]):
            if a.source_end != b.source_start or (a.hand, a.object) == (b.hand, b.object):
                continue
            if min(a.num_frames, b.num_frames) <= FUTURE_NATIVE_STEPS:
                continue
            pairs.append(dict(
                case=f"ep{episode:03d}_seg{a.seg_id:03d}_to_seg{b.seg_id:03d}",
                episode=int(episode), seg_id=int(a.seg_id), step=0, row=int(a.Index), grasp=int(a.grasp),
                same_hand=bool(a.hand == b.hand),
                candidates=[dict(row=int(row.Index), hand=row.hand, object=row.object, grasp=int(row.grasp))
                            for row in (a, b)],
            ))
    return pairs


def shard(args):
    pairs = build_pairs(args.csv)
    print(f"{len(pairs)} pairs, this rank takes every {args.world}th", flush=True)
    selected = pairs[args.rank :: args.world]
    return selected[: args.limit] if args.limit else selected


def window(dataset, prefix, row, step):
    item = dataset._get(int(prefix[row]) + step)
    if (int(item["trajectory_idx"]), int(item["step_idx"])) != (row, step):
        raise RuntimeError(f"Dataset resolved another window for row {row} step {step}")
    return item


def sample_actions(args):
    import torch

    directory = args.output / f"action_{args.tag}"
    directory.mkdir(parents=True, exist_ok=True)
    model, dataset = load_action_runtime(args)
    prefix = np.concatenate([[0], np.cumsum(dataset.trajectory_lengths)[:-1]])
    for pair in shard(args):
        path = directory / f"{pair['case']}.npz"
        if path.exists():
            raise FileExistsError(path)
        item = window(dataset, prefix, pair["row"], 0)
        real, goals, pads = [], [], []
        for candidate in pair["candidates"]:
            start = window(dataset, prefix, candidate["row"], 0)
            real.append(denormalize(dataset, start["action"], start["proprio"]))
            pads.append(start["action_is_pad"].cpu().numpy().astype(bool))
            grasp = window(dataset, prefix, candidate["row"], candidate["grasp"])
            goals.append(denormalize(dataset, grasp["action"], grasp["proprio"])[0])
        actions = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for candidate in pair["candidates"]:
                context, mask = dataset._load_text_context(dataset.metadata.iloc[candidate["row"]])
                context[~mask] = 0.0
                for offset in range(args.seeds):
                    chunk = sample_chunk(model, item, context, mask, args.steps, args.seed + offset)
                    actions.append(denormalize(dataset, chunk, item["proprio"]))
        np.savez(
            path, actions=np.stack(actions).reshape(2, args.seeds, *actions[0].shape), real=np.stack(real),
            goals=np.stack(goals), pad=np.stack(pads).any(axis=0), case=json.dumps(pair),
        )
        print(pair["case"], flush=True)


def sample_videos(args):
    import torch

    directory = args.output / f"video_{args.tag}"
    directory.mkdir(parents=True, exist_ok=True)
    model, dataset = load_video_runtime(args)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for pair in shard(args):
            path = directory / f"{pair['case']}.npz"
            if path.exists():
                raise FileExistsError(path)
            batch, contexts = case_batch(dataset, pair, model.device)
            next_start = dict(pair, row=pair["candidates"][B]["row"], candidates=pair["candidates"][B:])
            real = [truth_frames(batch), truth_frames(case_batch(dataset, next_start, model.device)[0])]
            videos = [
                sample_video(model, batch, context, real[A].shape[0], args.steps, args.seed + offset)
                for context in contexts for offset in range(args.seeds)
            ]
            np.savez(path, videos=np.stack(videos).reshape(2, args.seeds, *videos[0].shape), real=np.stack(real),
                     case=json.dumps(pair))
            print(pair["case"], flush=True)


def displacement(chunk, hand, valid):
    position = chunk[valid][:, XYZ[hand]]
    return position - position[0]


def cosine(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def action_row(path):
    with np.load(path) as data:
        actions, real, goals, pad = data["actions"], data["real"], data["goals"], data["pad"]
        pair = json.loads(str(data["case"]))
    valid = ~pad
    hands = [candidate["hand"] for candidate in pair["candidates"]]
    per_seed = []
    for seed in range(actions.shape[1]):
        chunks = actions[:, seed]
        # Mean distance of prompt p to the real displacement of q, on the arm q moves.
        distance = [[float(np.linalg.norm(
            displacement(chunks[p], hands[q], valid) - displacement(real[q], hands[q], valid), axis=1).mean())
            for q in (A, B)] for p in (A, B)]
        progress, heading = [], []
        for p in (A, B):
            start = chunks[p][valid][0, XYZ[hands[B]]]
            end = chunks[p][valid][-1, XYZ[hands[B]]]
            goal = goals[B, XYZ[hands[B]]]
            progress.append(float(np.linalg.norm(start - goal) - np.linalg.norm(end - goal)))
            heading.append(cosine(end - start, goal - start))
        moved = [float(np.linalg.norm(displacement(chunks[p], hands[B], valid), axis=1).max()) for p in (A, B)]
        last = np.flatnonzero(valid)[-1]
        per_seed.append(dict(
            b_prompt_nearer_real_b=float(distance[B][B] < distance[A][B]),
            a_prompt_nearer_real_a=float(distance[A][A] < distance[B][A]),
            b_output_nearer_real_b=float(distance[B][B] < distance[B][A]),
            a_output_nearer_real_a=float(distance[A][A] < distance[A][B]),
            b_prompt_nearer_goal=float(progress[B] > progress[A]),
            b_arm_moves_more=float(moved[B] > moved[A]),
            distance_b_to_real_b=distance[B][B], distance_a_to_real_b=distance[A][B],
            distance_a_to_real_a=distance[A][A], distance_b_to_real_a=distance[B][A],
            progress_b=progress[B], progress_a=progress[A], heading_b=heading[B], heading_a=heading[A],
            prompt_gap=float(np.linalg.norm(chunks[A][last, XYZ[hands[B]]] - chunks[B][last, XYZ[hands[B]]])),
        ))
    row = {key: float(np.mean([item[key] for item in per_seed])) for key in per_seed[0]}
    for key in ("b_prompt_nearer_real_b", "a_prompt_nearer_real_a", "b_output_nearer_real_b",
                "a_output_nearer_real_a", "b_prompt_nearer_goal", "b_arm_moves_more"):
        row[f"{key}_chance"] = CHANCE
    arm = XYZ[hands[B]]
    last = np.flatnonzero(valid)[-1]
    goal = goals[B, arm]
    row.update(
        case=pair["case"], episode=pair["episode"], seg_id=pair["seg_id"], same_hand=pair["same_hand"],
        real_b_progress=float(np.linalg.norm(real[B][valid][0, arm] - goal) - np.linalg.norm(real[B][last, arm] - goal)),
        start_distance_to_goal=float(np.linalg.norm(actions[B, 0][valid][0, arm] - goal)),
        start_pose_gap=float(np.linalg.norm(real[A][valid][0, arm] - real[B][valid][0, arm])),
        seed_gap=float(np.mean([np.linalg.norm(actions[p, 0][last, arm] - actions[p, 1][last, arm]) for p in (A, B)])),
    )
    return row


def video_row(path):
    with np.load(path) as data:
        videos, real = data["videos"].astype(np.float32), data["real"].astype(np.float32)
        pair = json.loads(str(data["case"]))
    row = dict(case=pair["case"], episode=pair["episode"], seg_id=pair["seg_id"], same_hand=pair["same_hand"])
    for name, width in (("frame", videos.shape[-2]), ("table", WRIST_X)):
        # Prompt p against real q, per seed, over the generated future only.
        distance = np.array([[[
            np.abs(videos[p, seed, CONDITION_FRAMES:, :, :width] - real[q, CONDITION_FRAMES:, :, :width]).mean()
            for q in (A, B)] for p in (A, B)] for seed in range(videos.shape[1])])
        row[f"{name}_b_prompt_nearer_real_b"] = float((distance[:, B, B] < distance[:, A, B]).mean())
        row[f"{name}_a_prompt_nearer_real_a"] = float((distance[:, A, A] < distance[:, B, A]).mean())
        row[f"{name}_b_prompt_nearer_real_b_chance"] = row[f"{name}_a_prompt_nearer_real_a_chance"] = CHANCE
        row[f"{name}_l1_b_to_real_b"] = float(distance[:, B, B].mean())
        row[f"{name}_l1_a_to_real_b"] = float(distance[:, A, B].mean())
        row[f"{name}_l1_a_to_real_a"] = float(distance[:, A, A].mean())
        row[f"{name}_l1_b_to_real_a"] = float(distance[:, B, A].mean())
    future = videos[:, :, CONDITION_FRAMES:]
    row["prompt_gap"] = float(np.abs(future[A] - future[B]).mean())
    row["seed_gap"] = float(np.mean([np.abs(future[p, 0] - future[p, 1]).mean() for p in (A, B)]))
    return row


def summarize(rows):
    summary = dict(cases=len(rows))
    for key in rows[0]:
        if key.endswith("_chance") or key in ("case", "episode", "seg_id", "same_hand"):
            continue
        if f"{key}_chance" in rows[0]:
            summary[key] = rate(rows, key, f"{key}_chance")
        else:
            summary[key] = float(np.mean([row[key] for row in rows]))
    return summary


def grouped(rows):
    groups = dict(all=rows, same_hand=[row for row in rows if row["same_hand"]],
                  other_hand=[row for row in rows if not row["same_hand"]])
    return {name: summarize(group) for name, group in groups.items() if group}


def labeled(frames, text):
    strip = Image.new("RGB", (frames.shape[2], LABEL_HEIGHT), "black")
    ImageDraw.Draw(strip).text((4, 3), text, fill="white")
    strip = np.repeat(np.asarray(strip)[None], frames.shape[0], axis=0)
    return np.concatenate([strip, frames], axis=1)


def describe(candidate):
    return f"{candidate['hand']} {candidate['object']}"


def write_video(path, destination):
    with np.load(path) as data:
        videos, real = data["videos"], data["real"]
        pair = json.loads(str(data["case"]))
    a, b = (describe(candidate) for candidate in pair["candidates"])
    top = np.concatenate([labeled(videos[A, 0, FIRST_FRAME:], f"generated, prompt A: {a}"),
                          labeled(videos[B, 0, FIRST_FRAME:], f"generated, prompt B: {b}")], axis=2)
    bottom = np.concatenate([labeled(real[A, FIRST_FRAME:], f"real A: {a}"),
                             labeled(real[B, FIRST_FRAME:], f"real B: {b}")], axis=2)
    encode(np.concatenate([top, bottom], axis=1), destination)


def draw_panel(actions, real, goals, valid, pair, hand):
    """Top view of one arm: x rises upward and y rises to the left, in a square of equal scale."""
    arm = XYZ[hand]
    points = [actions[:, :, valid][..., arm].reshape(-1, 3), real[:, valid][..., arm].reshape(-1, 3)]
    markers = [(goals[index, arm], COLORS[index], "AB"[index]) for index in (A, B)
               if pair["candidates"][index]["hand"] == hand]
    points = np.concatenate(points + [goal[None] for goal, _, _ in markers])[:, :2]
    center, scale = (points.max(0) + points.min(0)) / 2, max(np.ptp(points, axis=0).max(), 0.05) * 1.15

    def place(xy):
        relative = (np.asarray(xy)[..., :2] - center) / scale
        return np.stack([(0.5 - relative[..., 1]) * PANEL, (0.5 - relative[..., 0]) * PANEL + LABEL_HEIGHT], -1)

    panel = Image.new("RGB", (PANEL, PANEL + LABEL_HEIGHT), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((4, 3), f"{hand} arm, top view, {scale:.2f} m across", fill="black")
    for index in (A, B):
        for seed in range(actions.shape[1]):
            draw.line([tuple(point) for point in place(actions[index, seed][valid][:, arm])], fill=COLORS[index], width=2)
        line = place(real[index][valid][:, arm])
        for start, end in zip(line[::2], line[1::2]):
            draw.line([tuple(start), tuple(end)], fill=COLORS[index], width=1)
    x, y = place(actions[A, 0][valid][0, arm])
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="black")
    for goal, color, name in markers:
        x, y = place(goal)
        draw.rectangle((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)
        draw.text((x + 7, y - 5), f"{name} grasp", fill=color)
    return panel


def write_plot(path, destination):
    with np.load(path) as data:
        actions, real, goals, pad = data["actions"], data["real"], data["goals"], data["pad"]
        pair = json.loads(str(data["case"]))
    a, b = (describe(candidate) for candidate in pair["candidates"])
    sheet = Image.new("RGB", (PANEL * 2 + 4, PANEL + LABEL_HEIGHT * 3), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 3), f"{pair['case']}  blue A: {a}  orange B: {b}", fill="black")
    draw.text((4, LABEL_HEIGHT + 3), "solid: generated under that prompt, dashed: real, dot: observed start", fill="black")
    for column, hand in enumerate(("left", "right")):
        sheet.paste(draw_panel(actions, real, goals, ~pad, pair, hand), (column * (PANEL + 4), LABEL_HEIGHT * 2))
    sheet.save(destination)


def report(args):
    summary = {}
    for directory in sorted(args.output.glob("action_*")):
        if directory.is_dir() and (paths := sorted(directory.glob("*.npz"))):
            rows = [action_row(path) for path in paths]
            summary[directory.name] = grouped(rows)
            (args.output / f"{directory.name}_cases.json").write_text(json.dumps(rows, indent=2) + "\n")
            media = args.output / "media" / directory.name[len("action_") :]
            media.mkdir(parents=True, exist_ok=True)
            for path in paths:
                write_plot(path, media / f"{path.stem}.png")
    for directory in sorted(args.output.glob("video_*")):
        if directory.is_dir() and (paths := sorted(directory.glob("*.npz"))):
            rows = [video_row(path) for path in paths]
            summary[directory.name] = grouped(rows)
            (args.output / f"{directory.name}_cases.json").write_text(json.dumps(rows, indent=2) + "\n")
            media = args.output / "media" / directory.name[len("video_") :]
            media.mkdir(parents=True, exist_ok=True)
            for path in paths:
                write_video(path, media / f"{path.stem}.mp4")
    (args.output / "next_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("action", "video", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag")
    parser.add_argument("--joint", type=Path, help="Joint checkpoint under test")
    parser.add_argument("--video", type=Path, help="Video weights the joint run started from")
    parser.add_argument("--action", type=Path, help="Action weights the joint run started from")
    parser.add_argument("--wan", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--csv", type=Path, help="Defaults to val.csv under --prepared")
    parser.add_argument("--steps", type=int, help="Defaults to 10 for action and 30 for video, as in the other gates")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=100008)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    args = parser.parse_args()
    if args.command == "report":
        return report(args)
    if not all((args.tag, args.joint, args.video, args.action, args.wan, args.prepared, args.media_root)):
        parser.error(f"{args.command} needs --tag --joint --video --action --wan --prepared --media-root")
    args.csv = args.csv or args.prepared / "val.csv"
    if args.command == "action":
        args.checkpoint, args.config = args.joint, Path("configs/marvin/train_joint.yaml")
        args.steps = args.steps or 10
        return sample_actions(args)
    args.checkpoint, args.config = args.video, Path("configs/marvin/train_video.yaml")
    args.steps = args.steps or 30
    return sample_videos(args)


if __name__ == "__main__":
    main()
