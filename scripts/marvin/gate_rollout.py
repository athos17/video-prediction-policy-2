#!/usr/bin/env python
"""Multi-step rollouts of a Marvin joint checkpoint from the first frame of a PICK.

One action chunk covers 1.07 s and a grasp takes about 5 s, so the policy has to run repeatedly.
There is no robot or simulator here. Each cycle the policy samples 32 actions, a Video checkpoint
imagines the frames of those 32 steps, and the next cycle observes those frames and takes the last
commanded action as its proprioception. The imagined frames do not depend on the actions.

Each case is a pair of consecutive PICKs A, B of gate_next.py, rolled out from the first frame of A
under the prompts of A and B. A's grasp is compared with its real trajectory. B really started
later from another pose, so prompt B is compared with where B was really grasped.

  truth    Read the real trajectories and frames, and roll out prompt A on real frames and states.
           This is the upper bound without imagination. Run it first.
  imagine  Run the rollouts with --world-video. One process per GPU with --rank/--world.
  report   Summarize every rollout.
  site     Write one video per case and data.json for gate_rollout_viewer.html.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from gate1 import CONDITION_FRAMES, WRIST_X, bootstrap, sample_video
from gate1 import load_runtime as load_video_runtime
from gate1_web import encode
from gate_action import HORIZON, XYZ, denormalize, sample_chunk
from gate_action import load_runtime as load_action_runtime
from gate_next import A, B, LABEL_HEIGHT, build_pairs, describe, window

# Cycles execute a whole chunk, and frame 8 of a generated future falls on its end.
FRAME_STRIDE = 4
FRAMES_PER_CHUNK = HORIZON // FRAME_STRIDE
HISTORY_FRAMES = 8
HISTORY_STRIDE = 30
AFTER_GRASP = 64
MAX_STEPS = 480
HAND = dict(left=slice(9, 31), right=slice(40, 62))
FPS = 30
# Real commanded hands reach a median of 1.0 of the way to their grasp pose 1 s after it, and
# the arm rises a median of 23 cm in that second.
CLOSED = 0.8
LIFT = 0.05
RADII = (0.03, 0.05, 0.08)


def shard(args):
    pairs = build_pairs(args.csv)
    print(f"{len(pairs)} pairs, this rank takes every {args.world}th", flush=True)
    selected = pairs[args.rank :: args.world]
    return selected[: args.limit] if args.limit else selected


def horizon(pair):
    steps = max(candidate["grasp"] for candidate in pair["candidates"]) + AFTER_GRASP
    return min(math.ceil(steps / HORIZON) * HORIZON, MAX_STEPS)


def normalize_state(dataset, state):
    """Normalize a raw 62-D state as the dataset normalizes proprioception."""
    import torch

    processor = dataset.processor
    key = processor.shape_meta["state"][0]["key"]
    batch = dict(state={key: torch.as_tensor(state, dtype=torch.float32)[None]})
    return processor.normalizer.forward(batch)["state"][key]


def to_frames(tensor):
    """[C,T,H,W] in [-1,1] to [T,H,W,C] uint8."""
    import torch

    return ((tensor.permute(1, 2, 3, 0).float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()


def to_tensor(frame):
    import torch

    return torch.from_numpy(frame).float().permute(2, 0, 1) / 127.5 - 1


def condition_times(step):
    """Native steps of the condition: the first frame, then 8 frames one second apart ending now."""
    times = [0]
    for index in range(HISTORY_FRAMES):
        native = max(step + (index - HISTORY_FRAMES + 1) * HISTORY_STRIDE, 0)
        # Imagined frames exist every 4 steps; training jitters history by up to a step.
        times.append(min((native + FRAME_STRIDE // 2) // FRAME_STRIDE * FRAME_STRIDE, step))
    return times


def build_condition(frames, step):
    import torch

    stack = [to_tensor(frames[time // FRAME_STRIDE]) for time in condition_times(step)]
    return torch.stack(stack, dim=1)[None]


def rollout(policy, world, dataset, item, context, mask, steps, args, seed):
    """One rollout from a real first window: chunks of raw actions and imagined frames."""
    import torch

    start = to_frames(item["condition_video"])[-1]
    frames, actions = [start], []
    proprio = item["proprio"]
    video_frames = int(item["video_num_frames"])
    for cycle in range(steps // HORIZON):
        step = cycle * HORIZON
        condition = build_condition(frames, step)
        observed = dict(video=condition[0], condition_video=condition[0], proprio=proprio,
                        video_num_frames=video_frames)
        chunk = sample_chunk(policy, observed, context, mask, args.action_steps, seed + cycle)
        raw = denormalize(dataset, chunk, proprio)
        actions.append(raw)
        future = sample_video(world, dict(condition_video=condition.to(world.device)),
                              context.to(world.device), video_frames, args.video_steps, seed + cycle)
        frames.extend(future[CONDITION_FRAMES : CONDITION_FRAMES + FRAMES_PER_CHUNK])
        proprio = normalize_state(dataset, raw[-1])
        print(f"  cycle {cycle} step {step}", flush=True)
    return np.concatenate(actions), np.stack(frames)


def real_segment(dataset, prefix, row, steps):
    """Actions and frames of a segment over its first `steps` steps, held at its last frame."""
    length = int(dataset.trajectory_lengths[row])
    actions, states, frames = [], [], []
    valid_actions, valid_frames = [], []
    for step in range(0, steps, HORIZON):
        item = window(dataset, prefix, row, min(step, length - 1))
        video = to_frames(item["video"])
        pad = item["image_is_pad"].cpu().numpy().astype(bool)
        if step == 0:
            frames.append(video[CONDITION_FRAMES - 1 : CONDITION_FRAMES])
            valid_frames.append(np.ones(1, dtype=bool))
        frames.append(video[CONDITION_FRAMES : CONDITION_FRAMES + FRAMES_PER_CHUNK])
        valid_frames.append(~pad[CONDITION_FRAMES : CONDITION_FRAMES + FRAMES_PER_CHUNK] if step < length
                            else np.zeros(FRAMES_PER_CHUNK, dtype=bool))
        actions.append(denormalize(dataset, item["action"], item["proprio"]))
        valid_actions.append(~item["action_is_pad"].cpu().numpy().astype(bool) if step < length
                             else np.zeros(HORIZON, dtype=bool))
    return dict(actions=np.concatenate(actions), frames=np.concatenate(frames),
                valid_actions=np.concatenate(valid_actions), valid_frames=np.concatenate(valid_frames))


def truth(args):
    import torch

    directory = args.output / "truth"
    teacher = args.output / "rollout_real"
    directory.mkdir(parents=True, exist_ok=True)
    teacher.mkdir(parents=True, exist_ok=True)
    policy, dataset = load_action_runtime(args)
    prefix = np.concatenate([[0], np.cumsum(dataset.trajectory_lengths)[:-1]])
    for pair in shard(args):
        path = directory / f"{pair['case']}.npz"
        if path.exists():
            raise FileExistsError(path)
        steps = horizon(pair)
        real = [real_segment(dataset, prefix, candidate["row"], steps) for candidate in pair["candidates"]]
        goals, goal_frames = [], []
        for candidate in pair["candidates"]:
            item = window(dataset, prefix, candidate["row"], candidate["grasp"])
            goals.append(denormalize(dataset, item["action"], item["proprio"])[0])
            goal_frames.append(to_frames(item["video"])[CONDITION_FRAMES - 1])
        np.savez(
            path, actions=np.stack([part["actions"] for part in real]),
            valid_actions=np.stack([part["valid_actions"] for part in real]),
            frames=np.stack([part["frames"] for part in real]),
            valid_frames=np.stack([part["valid_frames"] for part in real]),
            goals=np.stack(goals), goal_frames=np.stack(goal_frames), case=json.dumps(pair),
        )
        length = int(dataset.trajectory_lengths[pair["row"]])
        context, mask = dataset._load_text_context(dataset.metadata.iloc[pair["candidates"][A]["row"]])
        context[~mask] = 0.0
        chunks = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for seed in range(args.seeds):
                actions = []
                for step in range(0, steps, HORIZON):
                    item = window(dataset, prefix, pair["row"], min(step, length - 1))
                    chunk = sample_chunk(policy, item, context, mask, args.action_steps, args.seed + seed * 1000 + step)
                    actions.append(denormalize(dataset, chunk, item["proprio"]))
                chunks.append(np.concatenate(actions))
        np.savez(teacher / f"{pair['case']}.npz", actions=np.stack(chunks)[None], case=json.dumps(pair))
        print(pair["case"], flush=True)


def imagine(args):
    import torch

    directory = args.output / f"rollout_{args.tag}"
    directory.mkdir(parents=True, exist_ok=True)
    policy, dataset = load_action_runtime(args)
    world, _ = load_video_runtime(SimpleNamespace(
        config=Path("configs/marvin/train_video.yaml"), wan=args.wan, checkpoint=args.video,
        joint=args.world_video, prepared=args.prepared, media_root=args.media_root, csv=args.csv, gpu=args.world_gpu,
    ))
    prefix = np.concatenate([[0], np.cumsum(dataset.trajectory_lengths)[:-1]])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for pair in shard(args):
            path = directory / f"{pair['case']}.npz"
            if path.exists():
                raise FileExistsError(path)
            item = window(dataset, prefix, pair["row"], 0)
            steps = horizon(pair)
            actions, frames = [], []
            for candidate in pair["candidates"]:
                context, mask = dataset._load_text_context(dataset.metadata.iloc[candidate["row"]])
                context[~mask] = 0.0
                for seed in range(args.seeds):
                    print(f"{pair['case']} {describe(candidate)} seed {seed}", flush=True)
                    action, frame = rollout(policy, world, dataset, item, context, mask, steps, args,
                                            args.seed + seed * 1000)
                    actions.append(action)
                    frames.append(frame)
            shape = (len(pair["candidates"]), args.seeds)
            np.savez(path, actions=np.stack(actions).reshape(*shape, *actions[0].shape),
                     frames=np.stack(frames).reshape(*shape, *frames[0].shape), case=json.dumps(pair))


def distance_to(trajectory, goal, hand):
    return np.linalg.norm(trajectory[:, XYZ[hand]] - goal[XYZ[hand]], axis=1)


def closure_to(trajectory, goal, hand):
    """How far the hand went from its first pose to the grasp pose: 0 at the start, 1 at the pose."""
    pose = trajectory[:, HAND[hand]]
    direction = goal[HAND[hand]] - pose[0]
    return (pose - pose[0]) @ direction / max(float(direction @ direction), 1e-9)


def grasps(trajectory, goal, hand, radius):
    """Reached the grasp pose, closed the hand there, then lifted. No physics is involved."""
    distance, closure = distance_to(trajectory, goal, hand), closure_to(trajectory, goal, hand)
    height = trajectory[:, XYZ[hand]][:, 2]
    holding = np.flatnonzero((distance <= radius) & (closure >= CLOSED))
    if not len(holding):
        return 0.0
    return float(height[holding[0] :].max() - goal[XYZ[hand]][2] >= LIFT)


def travel(trajectory, hand):
    position = trajectory[:, XYZ[hand]]
    return float(np.linalg.norm(position - position[0], axis=1).max())


def rollout_row(path, truth_directory):
    with np.load(path) as data:
        actions = data["actions"]
        frames = data["frames"] if "frames" in data else None
        pair = json.loads(str(data["case"]))
    with np.load(truth_directory / path.name) as data:
        real, valid = data["actions"], data["valid_actions"]
        goals, goal_frames, real_frames = data["goals"], data["goal_frames"], data["frames"]
        valid_frames = data["valid_frames"]
    candidates = pair["candidates"]
    hands = [candidate["hand"] for candidate in candidates]
    row = dict(case=pair["case"], episode=pair["episode"], seg_id=pair["seg_id"], same_hand=pair["same_hand"],
               goal_gap=float(np.linalg.norm(goals[A, XYZ[hands[A]]] - goals[B, XYZ[hands[B]]])) if pair["same_hand"] else None)
    per_seed = []
    for seed in range(actions.shape[1]):
        item = {}
        for name, prompt in (("a", A), ("b", B)):
            if prompt >= actions.shape[0]:
                continue
            trajectory = actions[prompt, seed]
            hand = hands[prompt]
            own = distance_to(trajectory, goals[prompt], hand)
            item[f"{name}_reach"] = float(own.min())
            item[f"{name}_final_closure"] = float(closure_to(trajectory, goals[prompt], hand).max())
            item[f"{name}_arm_moves_more"] = float(travel(trajectory, hand) > travel(trajectory, "left" if hand == "right" else "right"))
            item[f"{name}_reach_time"] = float(own.argmin() - candidates[prompt]["grasp"]) / FPS
            for radius in RADII:
                label = int(radius * 100)
                item[f"{name}_reach_{label}cm"] = float(own.min() <= radius)
                item[f"{name}_grasp_{label}cm"] = grasps(trajectory, goals[prompt], hand, radius)
        grasp = candidates[A]["grasp"]
        horizon_a = valid[A] & (np.arange(valid.shape[1]) <= grasp)
        item["a_tracking"] = float(np.linalg.norm(
            actions[A, seed][horizon_a][:, XYZ[hands[A]]] - real[A][horizon_a][:, XYZ[hands[A]]], axis=1).mean())
        item["a_hold_tracking"] = float(np.linalg.norm(
            real[A][0, XYZ[hands[A]]] - real[A][horizon_a][:, XYZ[hands[A]]], axis=1).mean())
        if actions.shape[0] > 1:
            for radius in RADII:
                label = int(radius * 100)
                # Controls: does the rollout of the other prompt also count as this grasp?
                item[f"a_prompt_grasps_b_{label}cm"] = grasps(actions[A, seed], goals[B], hands[B], radius)
                item[f"b_prompt_grasps_a_{label}cm"] = grasps(actions[B, seed], goals[A], hands[A], radius)
            item["a_prompt_reach_of_b"] = float(distance_to(actions[A, seed], goals[B], hands[B]).min())
            item["b_prompt_nearer_b"] = float(item["b_reach"] < item["a_prompt_reach_of_b"])
            item["a_prompt_nearer_a"] = float(
                item["a_reach"] < distance_to(actions[B, seed], goals[A], hands[A]).min())
            if pair["same_hand"]:
                item["b_prompt_ends_at_own_goal"] = float(item["b_reach"] < distance_to(actions[B, seed], goals[A], hands[B]).min())
            item["prompt_gap"] = float(np.linalg.norm(
                actions[A, seed][:, XYZ[hands[B]]] - actions[B, seed][:, XYZ[hands[B]]], axis=1).max())
        if frames is not None:
            for name, prompt in (("a", A), ("b", B)):
                video = frames[prompt, seed].astype(np.float32)
                item[f"{name}_video_to_grasp"] = float(np.abs(video - goal_frames[prompt].astype(np.float32)).mean(axis=(1, 2, 3)).min())
            other = lambda prompt, goal: float(np.abs(frames[prompt, seed].astype(np.float32) - goal_frames[goal].astype(np.float32)).mean(axis=(1, 2, 3)).min())
            item["b_video_nearer_b_grasp"] = float(item["b_video_to_grasp"] < other(A, B))
            item["a_video_nearer_a_grasp"] = float(item["a_video_to_grasp"] < other(B, A))
            usable = valid_frames[A]
            error = np.abs(frames[A, seed].astype(np.float32) - real_frames[A].astype(np.float32)).mean(axis=(1, 2, 3))
            for second in (1, 2, 4, 6):
                index = second * FPS // FRAME_STRIDE
                if index < len(error) and usable[index]:
                    item[f"a_video_l1_{second}s"] = float(error[index])
        per_seed.append(item)
    for key in sorted({key for item in per_seed for key in item}):
        values = [item[key] for item in per_seed if key in item]
        row[key] = float(np.mean(values))
    return row


def true_row(path):
    """The same measures on the real trajectories, which calibrate the thresholds."""
    with np.load(path) as data:
        real, valid, goals = data["actions"], data["valid_actions"], data["goals"]
        pair = json.loads(str(data["case"]))
    row = dict(case=pair["case"], episode=pair["episode"], seg_id=pair["seg_id"], same_hand=pair["same_hand"])
    for name, prompt in (("a", A), ("b", B)):
        hand = pair["candidates"][prompt]["hand"]
        trajectory = real[prompt][valid[prompt]]
        row[f"{name}_start_to_goal"] = float(distance_to(trajectory, goals[prompt], hand)[0])
        for radius in RADII:
            row[f"{name}_grasp_{int(radius * 100)}cm"] = grasps(trajectory, goals[prompt], hand, radius)
    return row


def summarize(rows):
    summary = dict(cases=len(rows))
    keys = {key for row in rows for key, value in row.items() if isinstance(value, float)}
    for key in sorted(keys):
        kept = [row for row in rows if row.get(key) is not None]
        values = [row[key] for row in kept]
        summary[key] = dict(mean=float(np.mean(values)), median=float(np.median(values)),
                            ci95=bootstrap(values, [f"{row['episode']}_{row['seg_id']}" for row in kept]))
    return summary


def grouped(rows):
    short = {3, 30, 57}
    groups = dict(
        all=rows, same_hand=[row for row in rows if row["same_hand"]],
        other_hand=[row for row in rows if not row["same_hand"]],
        short_episodes=[row for row in rows if row["episode"] in short],
        long_episodes=[row for row in rows if row["episode"] not in short],
    )
    return {name: summarize(group) for name, group in groups.items() if group}


def report(args):
    truth_directory = args.output / "truth"
    summary = dict(real=grouped([true_row(path) for path in sorted(truth_directory.glob("*.npz"))]))
    for directory in sorted(args.output.glob("rollout_*")):
        if directory.is_dir() and (paths := sorted(directory.glob("*.npz"))):
            rows = [rollout_row(path, truth_directory) for path in paths]
            summary[directory.name] = grouped(rows)
            (args.output / f"{directory.name}_cases.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "rollout_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def stamped(frames, texts):
    """Frames with one line of text above each."""
    strips = []
    for text in texts:
        strip = Image.new("RGB", (frames.shape[2], LABEL_HEIGHT), "black")
        ImageDraw.Draw(strip).text((4, 3), text, fill="white")
        strips.append(np.asarray(strip))
    return np.concatenate([np.stack(strips), frames], axis=1)


def measures(trajectory, goal, hand):
    """Distance and closure at each video frame."""
    index = np.minimum(np.arange(trajectory.shape[0] // FRAME_STRIDE + 1) * FRAME_STRIDE, trajectory.shape[0] - 1)
    return distance_to(trajectory, goal, hand)[index], closure_to(trajectory, goal, hand)[index]


def compose(run, truth_data, pair, seed):
    candidates = pair["candidates"]
    columns = []
    for prompt, candidate in enumerate(candidates):
        name = "AB"[prompt]
        goal = truth_data["goals"][prompt]
        hand = candidate["hand"]
        distance, closure = measures(run["actions"][prompt, seed], goal, hand)
        count = run["frames"].shape[2]
        generated = stamped(run["frames"][prompt, seed], [
            f"imagined {name}: {describe(candidate)} {index * FRAME_STRIDE / FPS:.1f}s "
            f"{distance[index] * 100:.0f}cm hand {closure[index]:.2f}" for index in range(count)])
        real_d, real_c = measures(truth_data["actions"][prompt], goal, hand)
        real = stamped(truth_data["frames"][prompt], [
            f"real {name}: {describe(candidate)} {index * FRAME_STRIDE / FPS:.1f}s "
            f"{real_d[index] * 100:.0f}cm hand {real_c[index]:.2f}"
            + ("" if truth_data["valid_frames"][prompt, index] else " ended") for index in range(count)])
        goal_frame = stamped(np.repeat(truth_data["goal_frames"][prompt][None], count, axis=0),
                             [f"real {name} at its grasp, {candidate['grasp'] / FPS:.1f} s"] * count)
        columns.append(np.concatenate([generated, real, goal_frame], axis=1))
    return np.concatenate(columns, axis=2)


def series(trajectory, valid=None):
    """Both arms' positions at the video rate, in millimetres, with None past the valid range."""
    index = np.arange(0, trajectory.shape[0], FRAME_STRIDE)
    result = {}
    for hand in XYZ:
        position = np.round(trajectory[index][:, XYZ[hand]] * 1000).astype(int).tolist()
        if valid is not None:
            position = [point if valid[i] else None for point, i in zip(position, index)]
        result[hand] = position
    return result


def site(args):
    videos = args.site / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    truth_directory = args.output / "truth"
    worlds = sorted(path.name[len("rollout_") : -len("_cases.json")] for path in args.output.glob("rollout_*_cases.json")
                    if path.name != "rollout_real_cases.json")
    rows = {world: {row["case"]: row for row in json.loads((args.output / f"rollout_{world}_cases.json").read_text())}
            for world in worlds}
    cases = []
    for path in sorted(truth_directory.glob("*.npz")):
        with np.load(path) as data:
            truth_data = {key: data[key] for key in data.files if key != "case"}
            pair = json.loads(str(data["case"]))
        with np.load(args.output / "rollout_real" / path.name) as data:
            teacher = data["actions"]
        hands = [candidate["hand"] for candidate in pair["candidates"]]
        item = dict(
            id=pair["case"], episode=pair["episode"], sameHand=pair["same_hand"],
            prompts=[dict(hand=candidate["hand"], object=candidate["object"], grasp=candidate["grasp"],
                          goal=np.round(truth_data["goals"][index][XYZ[candidate["hand"]]] * 1000).astype(int).tolist(),
                          distance=np.round(measures(truth_data["actions"][index], truth_data["goals"][index], candidate["hand"])[0] * 1000).astype(int).tolist(),
                          closure=np.round(measures(truth_data["actions"][index], truth_data["goals"][index], candidate["hand"])[1], 3).tolist(),
                          real=series(truth_data["actions"][index], truth_data["valid_actions"][index]))
                     for index, candidate in enumerate(pair["candidates"])],
            teacher=series(teacher[A, 0], truth_data["valid_actions"][A]),
            worlds={},
        )
        for world in worlds:
            run_path = args.output / f"rollout_{world}" / path.name
            if not run_path.exists():
                continue
            with np.load(run_path) as data:
                run = dict(actions=data["actions"], frames=data["frames"])
            encode(compose(run, truth_data, pair, 0), videos / f"{world}_{pair['case']}.mp4")
            rollouts = []
            for prompt in (A, B):
                distance, closure = measures(run["actions"][prompt, 0], truth_data["goals"][prompt], hands[prompt])
                rollouts.append(dict(series(run["actions"][prompt, 0]),
                                     distance=np.round(distance * 1000).astype(int).tolist(),
                                     closure=np.round(closure, 3).tolist()))
            metrics = {key: round(value, 4) for key, value in rows[world][pair["case"]].items() if isinstance(value, float)}
            item["worlds"][world] = dict(rollouts=rollouts, metrics=metrics)
        cases.append(item)
        print(pair["case"], flush=True)
    summary = json.loads((args.output / "rollout_summary.json").read_text())
    (args.site / "data.json").write_text(json.dumps(
        dict(worlds=worlds, summary=summary, cases=cases, frameStep=FRAME_STRIDE / FPS),
        ensure_ascii=False, separators=(",", ":")))
    shutil.copy(Path(__file__).with_name("gate_rollout_viewer.html"), args.site / "index.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("truth", "imagine", "report", "site"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--checkpoint", type=Path, help="Joint checkpoint under test")
    parser.add_argument("--video", type=Path, help="Video weights the joint run started from")
    parser.add_argument("--action", type=Path, help="Action weights the joint run started from")
    parser.add_argument("--world-video", type=Path, help="Video-only or joint weights that imagine the frames")
    parser.add_argument("--config", type=Path, default=Path("configs/marvin/train_joint.yaml"))
    parser.add_argument("--wan", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--csv", type=Path, help="Defaults to val.csv under --prepared")
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--video-steps", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100008)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int, help="Shorten the rollouts, for smoke tests")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--world-gpu", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    args = parser.parse_args()
    if args.command == "report":
        return report(args)
    if args.command == "site":
        return site(args)
    if not all((args.checkpoint, args.video, args.action, args.wan, args.prepared, args.media_root)):
        parser.error(f"{args.command} needs --checkpoint --video --action --wan --prepared --media-root")
    if args.command == "imagine" and not (args.tag and args.world_video):
        parser.error("imagine needs --tag --world-video")
    if args.max_steps:
        global MAX_STEPS
        MAX_STEPS = args.max_steps
    args.csv = args.csv or args.prepared / "val.csv"
    return truth(args) if args.command == "truth" else imagine(args)


if __name__ == "__main__":
    main()
