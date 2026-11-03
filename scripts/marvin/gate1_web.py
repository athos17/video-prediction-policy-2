#!/usr/bin/env python3
"""Build the static site that compares Gate 1 generations across checkpoints."""

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd

HEIGHT, WIDTH = 240, 416
FPS = 7.5
SEEDS = 2
# The current frame; the eight frames before it are history at a one second stride.
FIRST_FRAME = 8
FUTURE_FRAME = 9
# Generated tiles fill rows 0-2 of an atlas; the ground truth sits in row 3, column 0.
ATLAS_ROWS = 4
PHASES = (
    dict(id="p50", label="phase 0.5", suffix="_p50", models=("init", "s1000", "s3000", "s4000")),
    dict(id="p0", label="phase 0", suffix="", models=("init", "s3000")),
)
SCORES = ("init", "s1000", "s2000", "s3000", "s4000")
LABELS = dict(init="预训练", s1000="1000 步", s2000="2000 步", s3000="3000 步", s4000="4000 步")
SHORT = dict(init="预", s1000="1k", s3000="3k", s4000="4k")


def encode(frames, path):
    process = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{frames.shape[2]}x{frames.shape[1]}",
            "-r", str(FPS), "-i", "-",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", "slow", "-crf", "25", "-g", str(frames.shape[0]),
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-movflags", "+faststart", "-an", str(path),
        ],
        input=frames.tobytes(), capture_output=True,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.decode())


def build_case(run, phase, name, metadata, videos):
    loaded = {}
    for model in phase["models"]:
        with np.load(run / f"generate_{model}{phase['suffix']}" / f"{name}.npz") as data:
            loaded[model] = dict(truth=data["truth"], videos=data["videos"], case=json.loads(str(data["case"])))
    first = loaded[phase["models"][0]]
    case, truth = first["case"], first["truth"]
    for model, item in loaded.items():
        if not np.array_equal(item["truth"], truth) or item["case"]["candidates"] != case["candidates"]:
            raise RuntimeError(f"{name}: {model} was evaluated on another window")

    real = truth[FUTURE_FRAME:].astype(np.float32)
    distance = {}
    for seed in range(SEEDS):
        atlas = np.zeros(
            (truth.shape[0] - FIRST_FRAME, HEIGHT * ATLAS_ROWS, WIDTH * len(phase["models"]), 3), np.uint8,
        )
        atlas[:, HEIGHT * (ATLAS_ROWS - 1) :, :WIDTH] = truth[FIRST_FRAME:]
        for column, model in enumerate(phase["models"]):
            generated = loaded[model]["videos"]
            for candidate in range(generated.shape[0]):
                atlas[
                    :, candidate * HEIGHT : (candidate + 1) * HEIGHT, column * WIDTH : (column + 1) * WIDTH,
                ] = generated[candidate, seed, FIRST_FRAME:]
        encode(atlas, videos / f"{name}_s{seed}.mp4")
    for model, item in loaded.items():
        future = item["videos"][:, :, FUTURE_FRAME:].astype(np.float32)
        distance[model] = np.abs(future - real).mean(axis=(2, 3, 4, 5))

    candidates = []
    for index, candidate in enumerate(case["candidates"]):
        row = metadata.loc[candidate["row"]]
        if row.hand != candidate["hand"] or row.object != candidate["object"]:
            raise RuntimeError(f"{name}: candidate {index} does not match its metadata row")
        candidates.append(dict(
            hand=candidate["hand"], object=candidate["object"], instruction=row.instruction,
            prompt=row.training_prompt,
            l1={model: [round(float(value), 2) for value in distance[model][index]] for model in phase["models"]},
        ))
    return dict(
        id=name, episode=case["episode"], seg=case["seg_id"], step=case["step"], grasp=case["grasp"],
        candidates=candidates,
        closest={model: int(distance[model].mean(axis=1).argmin()) for model in phase["models"]},
    ), distance


def summarize(case, distance):
    mean = distance.mean(axis=1)
    hands = [candidate["hand"] for candidate in case["candidates"]]
    same = [index for index, hand in enumerate(hands) if hand == hands[0]]
    row = dict(closest=float(mean.argmin() == 0), closest_chance=1 / len(hands), l1=float(mean[0]))
    if len(same) > 1:
        row["object"] = float(same[int(mean[same].argmin())] == 0)
        row["object_chance"] = 1 / len(same)
    if len(same) < len(hands):
        row["hand"] = float(hands[int(mean.argmin())] == hands[0])
        row["hand_chance"] = len(same) / len(hands)
    return row


def aggregate(rows):
    result = dict(cases=len(rows), closest=np.mean([row["closest"] for row in rows]),
                  l1=np.mean([row["l1"] for row in rows]))
    for key in ("closest_chance", "object", "object_chance", "hand", "hand_chance"):
        result[key] = np.mean([row[key] for row in rows if key in row])
    return {key: round(float(value), 4) for key, value in result.items()}


def build_phase(args, phase, metadata, videos):
    names = sorted(path.stem for path in (args.run / f"generate_{phase['models'][0]}{phase['suffix']}").glob("*.npz"))
    with ThreadPoolExecutor(args.workers) as pool:
        built = list(pool.map(lambda name: build_case(args.run, phase, name, metadata, videos), names))
    cases = [case for case, _ in built]
    metrics = {}
    for model in phase["models"]:
        report = {row["case"]: row for row in json.loads(
            (args.run / f"generate_{model}{phase['suffix']}_cases.json").read_text())}
        for case in cases:
            if float(case["closest"][model] == 0) != report[case["id"]]["top1"]:
                raise RuntimeError(f"{case['id']}: {model} disagrees with the Gate 1 report")
        metrics[model] = aggregate([summarize(case, distance[model]) for case, distance in built])
    return dict(id=phase["id"], label=phase["label"], models=phase["models"], metrics=metrics, cases=cases)


def build_scores(run):
    summary = json.loads((run / "summary.json").read_text())
    scores = {}
    for model in SCORES:
        item = summary[f"score/{model}_val"]
        scores[model] = {key: round(item[key]["rate"], 4) for key in ("top1", "object_top1", "hand_top1")}
    reference = summary[f"score/{SCORES[0]}_val"]
    chance = {key: round(reference[key]["chance"], 4) for key in ("top1", "object_top1", "hand_top1")}
    return dict(cases=reference["cases"], models=SCORES, values=scores, chance=chance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Gate 1 output directory")
    parser.add_argument("--csv", type=Path, required=True, help="metadata the cases were built from")
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    videos = args.site / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.csv, keep_default_na=False, na_values=[""])
    site = dict(
        labels=LABELS, short=SHORT, scores=build_scores(args.run),
        phases=[build_phase(args, phase, metadata, videos) for phase in PHASES],
    )
    (args.site / "data.json").write_text(json.dumps(site, ensure_ascii=False, separators=(",", ":")))
    shutil.copy(Path(__file__).with_name("gate1_viewer.html"), args.site / "index.html")
    print(json.dumps({phase["id"]: phase["metrics"] for phase in site["phases"]}, indent=2))


if __name__ == "__main__":
    main()
