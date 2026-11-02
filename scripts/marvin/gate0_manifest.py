#!/usr/bin/env python
"""Write counterfactual prompt files for zero-shot Stage-2 video prediction on Marvin.

Each case is the first frame of a held-out subtask paired with the demonstrated target
and with targets the same episode picks later, which should still be on the table. The
output suits `scripts/infer_video.sh --prompt-file`, one file per GPU shard.
"""

import argparse
import json
from pathlib import Path
import sys

import av
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vpp2.marvin_data import build_prompt  # noqa: E402


def first_frame(path):
    with av.open(str(path)) as container:
        frame = next(container.decode(video=0))
    # Matches the direct 416x240 resize applied to clips during training.
    return frame.to_image().resize((416, 240), Image.Resampling.BICUBIC)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end-effectors", nargs="+", default=["robot hand", "robot gripper"])
    parser.add_argument("--frames-per-episode", type=int, default=2)
    parser.add_argument("--max-targets", type=int, default=3)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    val = pd.read_csv(args.prepared / "val.csv", keep_default_na=False, na_values=[""])
    picks = val[val.segment_type == "PICK"].sort_values(["episode_index", "source_start"])
    (args.output / "images").mkdir(parents=True)
    cases = []
    for episode, group in picks.groupby("episode_index"):
        rows = list(group.itertuples(index=False))
        # Earliest subtasks leave the most candidates on the table.
        for position, row in enumerate(rows[: args.frames_per_episode]):
            targets, seen = [], set()
            for candidate in rows[position:]:
                key = (candidate.hand, candidate.object)
                if key not in seen:
                    seen.add(key)
                    targets.append(candidate)
            if len(targets) < 2:
                continue
            image = args.output / "images" / f"ep{episode:03d}_seg{row.seg_id:03d}.png"
            first_frame(args.media_root / row.video_path).save(image)
            for rank, target in enumerate(targets[: args.max_targets]):
                for end_effector in args.end_effectors:
                    cases.append(
                        dict(
                            episode=int(episode),
                            seg_id=int(row.seg_id),
                            demonstrated=rank == 0,
                            hand=target.hand,
                            object=target.object,
                            end_effector=end_effector,
                            image=str(image.resolve()),
                            prompt=build_prompt("PICK", target.hand, target.object, end_effector),
                        )
                    )
    if not cases:
        raise ValueError("No held-out frame offers more than one target")
    for shard in range(args.shards):
        selected = cases[shard :: args.shards]
        lines = [f"{case['prompt']}@@{case['image']}" for case in selected]
        (args.output / f"prompts_{shard}.txt").write_text("\n".join(lines) + "\n")
        (args.output / f"cases_{shard}.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps(dict(cases=len(cases), images=len({c["image"] for c in cases})), indent=2))


if __name__ == "__main__":
    main()
