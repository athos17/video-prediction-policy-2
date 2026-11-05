#!/usr/bin/env python3
"""Collect every offline test of the Marvin joint run into one site.

Writes data.json and the overview page into --site, and the single-step A-to-B page into
--site/next. gate_rollout.py site and gate1_web.py fill --site/rollout and --site/gate1.
"""

import argparse
import json
from pathlib import Path
import re
import shutil

import numpy as np

# Normalized MSE and L1 of repeating the first true action on the 1342 held-out windows.
HOLD_MSE = 0.116
HOLD_L1 = 0.157
FAR_GOALS = 0.08
ACTION_KEYS = ("closest", "object", "hand", "closest_phase_0.0", "closest_phase_0.5")
ACTION_VALUES = ("l1_demonstrated", "moves_other_hand", "gap_same_hand", "gap_other_hand", "gap_seed")
NEXT_ACTION = ("b_prompt_nearer_goal", "b_prompt_nearer_real_b", "a_prompt_nearer_real_a", "b_arm_moves_more")
NEXT_VIDEO = ("frame_b_prompt_nearer_b_grasp", "frame_a_prompt_nearer_a_grasp", "frame_b_prompt_nearer_real_b",
              "frame_a_prompt_nearer_real_a")
ROLLOUT_ORDER = ("s17500_v5000", "s30000_v5000", "s17500_j17500", "s30000_j30000")


def load(path):
    return json.loads(Path(path).read_text())


def step_of(tag):
    return 0 if tag == "init" else int(tag.lstrip("s"))


def evals(path):
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        match = re.search(r"\[eval\] step=(\d+) (.*)", line)
        if match:
            values = dict(pair.split("=") for pair in match.group(2).split())
            rows.append(dict(step=int(match.group(1)), mse=float(values["heldout/mse"]), l1=float(values["heldout/l1"]),
                             video=float(values["heldout/loss_video"])))
    return dict(rows=rows, hold_mse=HOLD_MSE, hold_l1=HOLD_L1)


def action_steps(directory):
    summary = load(directory / "action_summary.json")
    rows = []
    for name, item in summary.items():
        tag = name[len("action_") :]
        row = dict(tag=tag, step=step_of(tag), cases=item["cases"], rates={key: item[key] for key in ACTION_KEYS})
        row.update({key: item[key] for key in ACTION_VALUES})
        rows.append(row)
    return sorted(rows, key=lambda row: row["step"])


def gate1_rows(directory):
    summary = load(directory / "summary.json")
    tags = dict(init=0, s1000=1000, s2000=2000, s3000=3000, s4000=4000, s5000=5000)
    scores = []
    for key, item in summary.items():
        match = re.fullmatch(r"score/(.+)_val", key)
        if match:
            tag = match.group(1)
            step = tags.get(tag, int(tag[len("joint") :]) if tag.startswith("joint") else None)
            scores.append(dict(tag=tag, step=step, top1=item["top1"], object=item["object_top1"], hand=item["hand_top1"]))
    generated = []
    for key, item in summary.items():
        match = re.fullmatch(r"generate_(.+)_p50", key)
        if match:
            tag = match.group(1)
            step = tags.get(tag, int(tag[len("joint") :]) if tag.startswith("joint") else None)
            generated.append(dict(tag=tag, step=step, closest=item["closest_to_truth"], l1=item["l1_demonstrated"],
                                  prompt_gap=item["prompt_gap"], seed_gap=item["seed_gap"]))
    return dict(score=sorted(scores, key=lambda row: row["step"]), generate=sorted(generated, key=lambda row: row["step"]))


def same_hand_far(rows):
    """Outcomes on same-hand pairs whose goals are too far apart for one grasp to count as both."""
    kept = [row for row in rows if row["same_hand"] and row["goal_gap"] >= FAR_GOALS]
    outcome = dict(cases=len(kept), a_gets_a=0, a_gets_b=0, a_neither=0, b_gets_b=0, b_gets_a=0, b_neither=0)
    for row in kept:
        for prompt, own, other in (("a", "a_grasp_5cm", "a_prompt_grasps_b_5cm"), ("b", "b_grasp_5cm", "b_prompt_grasps_a_5cm")):
            mine, theirs = prompt, "b" if prompt == "a" else "a"
            if row[own] >= 0.5:
                outcome[f"{prompt}_gets_{mine}"] += 1
            elif row[other] >= 0.5:
                outcome[f"{prompt}_gets_{theirs}"] += 1
            else:
                outcome[f"{prompt}_neither"] += 1
    return outcome


def rollout(directory, old):
    path = directory / "rollout_summary.json"
    if not path.exists():
        return None
    summary = load(path)
    columns = []
    for name in ROLLOUT_ORDER:
        if f"rollout_{name}" in summary:
            cases = load(directory / f"rollout_{name}_cases.json")
            columns.append(dict(id=name, step=int(name.split("_")[0][1:]), world=name.split("_")[1], groups=summary[f"rollout_{name}"],
                                same_hand_far=same_hand_far(cases)))
    teachers = dict(s17500=load(old / "rollout_summary.json")["rollout_real"], s30000=summary.get("rollout_real"))
    return dict(columns=columns, teachers=teachers, real=summary["real"])


def next_summary(directory):
    summary = load(directory / "next_summary.json")
    rows = []
    for name, item in summary.items():
        if name.startswith("action_"):
            tag = name[len("action_") :]
            video = summary[f"video_{tag}"]
            rows.append(dict(
                tag=tag, step=step_of(tag),
                action={group: {key: values[key] for key in NEXT_ACTION} | dict(cases=values["cases"], prompt_gap=values["prompt_gap"], seed_gap=values["seed_gap"], progress_a=values["progress_a"], progress_b=values["progress_b"]) for group, values in item.items()},
                video={group: {key: values[key] for key in NEXT_VIDEO} | dict(prompt_gap=values["prompt_gap"], seed_gap=values["seed_gap"], l1=values["frame_l1_a_to_real_a"]) for group, values in video.items()},
            ))
    return sorted(rows, key=lambda row: row["step"])


def next_site(directory, site):
    site.mkdir(parents=True, exist_ok=True)
    tags = [path.name[len("action_") : -len("_cases.json")] for path in directory.glob("action_*_cases.json")]
    cases = {}
    for tag in tags:
        target = site / tag
        target.mkdir(exist_ok=True)
        for media in (directory / "media" / tag).iterdir():
            shutil.copy(media, target / media.name)
        videos = {row["case"]: row for row in load(directory / f"video_{tag}_cases.json")}
        for row in load(directory / f"action_{tag}_cases.json"):
            with np.load(directory / f"action_{tag}" / f"{row['case']}.npz") as data:
                pair = json.loads(str(data["case"]))
            item = cases.setdefault(row["case"], dict(
                id=row["case"], episode=row["episode"], sameHand=row["same_hand"],
                prompts=[dict(hand=candidate["hand"], object=candidate["object"]) for candidate in pair["candidates"]],
                runs={},
            ))
            video = videos[row["case"]]
            item["runs"][tag] = dict(
                progressB=round(row["progress_b"], 4), progressA=round(row["progress_a"], 4),
                goal=round(row["start_distance_to_goal"], 4), gap=round(row["prompt_gap"], 4), seedGap=round(row["seed_gap"], 4),
                actionB=row["b_prompt_nearer_goal"], videoB=video["frame_b_prompt_nearer_b_grasp"],
            )
    tags = sorted(tags, key=step_of)
    (site / "data.json").write_text(json.dumps(dict(tags=tags, cases=sorted(cases.values(), key=lambda item: item["id"])),
                                               ensure_ascii=False, separators=(",", ":")))
    shutil.copy(Path(__file__).with_name("gate_next_viewer.html"), site / "index.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--gate1", type=Path, required=True)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--next", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--old-rollout", type=Path, required=True, help="Rollouts of step 17500, for its real-frame bound")
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--notes", type=Path, help="JSON list of conclusions shown at the top of the page")
    args = parser.parse_args()
    args.site.mkdir(parents=True, exist_ok=True)
    data = dict(
        notes=load(args.notes) if args.notes else [],
        evals=evals(args.train_log), action=action_steps(args.action), gate1=gate1_rows(args.gate1),
        next=next_summary(args.next), rollout=rollout(args.rollout, args.old_rollout),
    )
    (args.site / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    shutil.copy(Path(__file__).with_name("gate_final_viewer.html"), args.site / "index.html")
    next_site(args.next, args.site / "next")


if __name__ == "__main__":
    main()
