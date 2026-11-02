"""Convert Marvin Pro + Sharpa Wave LeRobot v3 data into per-subtask VPP2 clips.

Each retained subtask becomes one Parquet and one RGB MP4, because training decodes
from the first frame of a file and full episodes reach almost 10k frames. Values keep
native 30 Hz timing and absolute units; only their order changes. Prompts, the split,
statistics and sampling weights live in the prepared directory, so changing them
never requires encoding video again.
"""

from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import json
from pathlib import Path
import shutil

import av
import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from .utils.conditioning import crop_box

FPS = 30
SOURCE_DIM = 76
VECTOR_DIM = 62
# Physical order shared with the OpenWAM Marvin reader:
# [left EEF xyz+rot6d (9), left hand (22), right EEF xyz+rot6d (9), right hand (22)].
VECTOR_SOURCE_COLUMNS = np.r_[58:67, 14:36, 67:76, 36:58]
ROT6D_COLUMNS = (3, 34)
SOURCE_ROT6D_LAYOUT = "[R00,R10,R20,R01,R11,R21]"
CAMERAS = (
    "observation.images.left_head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
# Twice the 440x240 Stage-2 canvas, keeping its geometry at training's direct 416x240 resize.
CANVAS_SIZE_WH = (880, 480)
HEAD_SIZE_WH = (640, 480)
WRIST_SIZE_WH = (240, 240)
SEGMENT_TYPES = ("PICK", "STILL")
PROMPT_SUFFIX = "This is a composed robot video with two wrist-view in the right side."
CRITICAL_PREFIX_ACTIONS = 8
ACTION_HORIZON = 32
# Half width of the critical interval around an annotated grasp or release: 0.5 s.
EVENT_HALF_WINDOW = 15
# Every dimension measured well above this; a smaller value indicates broken inputs.
MIN_STD = 1e-3


def physical_vector(source):
    """Select absolute EEF poses and hand joints from the 76-D source vector."""
    source = np.asarray(source, dtype=np.float32)
    if source.shape[-1] != SOURCE_DIM:
        raise ValueError(f"Expected {SOURCE_DIM}-D source vectors, got {source.shape}")
    return source[..., VECTOR_SOURCE_COLUMNS]


def rot6d_error(vectors):
    """Largest deviation from unit, orthogonal first-two-column rotations."""
    worst = 0.0
    for offset in ROT6D_COLUMNS:
        first = vectors[..., offset : offset + 3]
        second = vectors[..., offset + 3 : offset + 6]
        worst = max(
            worst,
            float(np.abs(np.linalg.norm(first, axis=-1) - 1).max()),
            float(np.abs(np.linalg.norm(second, axis=-1) - 1).max()),
            float(np.abs(np.sum(first * second, axis=-1)).max()),
        )
    return worst


def build_prompt(segment_type, hand, target, end_effector="robot hand"):
    """Stage-2 caption: left side first, an explicit idle side, then the camera layout."""
    if segment_type == "STILL":
        left, right = "keeps static", "keeps static"
    elif segment_type == "PICK":
        action = f"picks up the {target} and places it into the bin"
        if hand not in ("left", "right") or not target:
            raise ValueError(f"PICK needs a hand and a target, got {hand!r}/{target!r}")
        left, right = (action, "keeps static") if hand == "left" else ("keeps static", action)
    else:
        raise ValueError(f"Unsupported segment type: {segment_type}")
    sides = f"The left {end_effector} {left}, while the right {end_effector} {right}."
    return f"{sides} {PROMPT_SUFFIX}"


def compose_frame(head, left_wrist, right_wrist):
    """Crop the head view to 4:3, keep whole wrist views, and stack wrists on the right."""
    canvas = Image.new("RGB", CANVAS_SIZE_WH, "black")
    head = Image.fromarray(head)
    resampling = Image.Resampling.BICUBIC
    canvas.paste(head.crop(crop_box(head.size, 4 / 3)).resize(HEAD_SIZE_WH, resampling), (0, 0))
    canvas.paste(Image.fromarray(left_wrist).resize(WRIST_SIZE_WH, resampling), (640, 0))
    canvas.paste(Image.fromarray(right_wrist).resize(WRIST_SIZE_WH, resampling), (640, 240))
    return np.asarray(canvas)


def load_episodes(source):
    info = json.loads((source / "meta/info.json").read_text())
    if (info["codebase_version"], info["fps"]) != ("v3.0", FPS):
        raise ValueError(f"Expected a {FPS} Hz LeRobot v3.0 export")
    conversion = json.loads((source / "meta/openwam_conversion.json").read_text())
    if conversion["target_rotation_6d_layout"] != SOURCE_ROT6D_LAYOUT:
        raise ValueError("Expected first-two-column rot6d in the source export")
    for name, eef in (("action", "tj.control.eef"), ("observation.state", "tj.info.eef")):
        names = info["features"][name]["names"]
        if (
            info["features"][name]["shape"] != [SOURCE_DIM]
            or names[58] != f"{eef}_left.x"
            or names[67] != f"{eef}_right.x"
            or "left_thumb_CMC_FE" not in names[14]
            or "right_thumb_CMC_FE" not in names[36]
        ):
            raise ValueError(f"{name}: unexpected source vector layout")
    paths = sorted((source / "meta/episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(source / "meta/episodes")
    frame = pd.concat([pd.read_parquet(p) for p in paths]).sort_values("episode_index")
    if int(frame.length.sum()) != info["total_frames"] or len(frame) != info["total_episodes"]:
        raise ValueError("Episode metadata disagrees with meta/info.json")
    episodes = []
    for row in frame.to_dict("records"):
        videos = [
            source
            / info["video_path"].format(
                video_key=camera,
                chunk_index=int(row[f"videos/{camera}/chunk_index"]),
                file_index=int(row[f"videos/{camera}/file_index"]),
            )
            for camera in CAMERAS
        ]
        data = source / info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
        )
        episodes.append(
            dict(
                index=int(row["episode_index"]),
                length=int(row["length"]),
                first_row=int(row["dataset_from_index"]),
                data=data,
                videos=videos,
            )
        )
    return info, episodes


def load_segments(annotation, lengths):
    """Return retained segments by episode, rejecting labels that do not fit the data."""
    table = pd.read_parquet(annotation)
    required = {"episode_index", "seg_id", "type", "start", "end", "hand", "object", "exclude"}
    if required - set(table):
        raise ValueError(f"Missing annotation columns: {sorted(required - set(table))}")
    segments = {}
    for row in table[~table.exclude].sort_values(["episode_index", "start"]).to_dict("records"):
        episode, start, end = int(row["episode_index"]), int(row["start"]), int(row["end"])
        if episode not in lengths or not 0 <= start < end <= lengths[episode]:
            raise ValueError(f"Episode {episode}: invalid segment [{start},{end})")
        if row["type"] not in SEGMENT_TYPES:
            raise ValueError(f"Episode {episode}: retained segment of type {row['type']}")
        events = json.loads(row["events"]) if row.get("events") else {}
        if row["type"] == "PICK":
            # Validates the hand and target before any video work.
            build_prompt("PICK", row["hand"], row["object"])
            if not {"grasp", "release"} <= set(events):
                raise ValueError(f"Episode {episode}: PICK at {start} lacks grasp/release")
        segments.setdefault(episode, []).append(
            dict(
                seg_id=int(row["seg_id"]),
                type=row["type"],
                start=start,
                end=end,
                hand=row["hand"] if row["type"] == "PICK" else "",
                object=row["object"] if row["type"] == "PICK" else "",
                instruction=row["prompt"],
                n_candidates_same_side=row.get("n_candidates_same_side"),
                grasp=int(events["grasp"]) - start if "grasp" in events else "",
                release=int(events["release"]) - start if "release" in events else "",
            )
        )
    return segments


def _open_writer(path):
    container = av.open(str(path), "w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height = CANVAS_SIZE_WH
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "veryfast"}
    stream.codec_context.thread_count = 1
    stream.codec_context.time_base = Fraction(1, FPS)
    return container, stream


def _encode(container, stream, image, pts):
    frame = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(format="yuv420p")
    frame.pts = pts
    frame.time_base = Fraction(1, FPS)
    for packet in stream.encode(frame):
        container.mux(packet)


def _finish_clip(container, stream, temporary, destination, frames):
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    with av.open(str(temporary)) as video:
        output = video.streams.video[0]
        if (output.width, output.height, output.average_rate, output.frames) != (
            *CANVAS_SIZE_WH,
            FPS,
            frames,
        ):
            raise ValueError(f"Invalid output frame count/size/rate: {temporary}")
    temporary.replace(destination)


def _segment_table(action, state, segment):
    start, end = segment["start"], segment["end"]
    length = end - start
    frames = pa.array(np.arange(length))
    return pa.table(
        {
            "frame_index": frames,
            "raw_frame_index": frames,
            "source_frame_index": pa.array(np.arange(start, end)),
            "action": pa.array(action[start:end].tolist(), type=pa.list_(pa.float32())),
            "observation.state": pa.array(state[start:end].tolist(), type=pa.list_(pa.float32())),
        }
    )


def convert_episode(episode, segments, output):
    """Write the clips of one episode; the camera streams must match the table row for row."""
    columns = ["index", "episode_index", "frame_index", "action", "observation.state"]
    table = pd.read_parquet(episode["data"], columns=columns)
    index, length = episode["index"], episode["length"]
    for column, expected in (
        ("index", episode["first_row"] + np.arange(length)),
        ("episode_index", np.full(length, index)),
        ("frame_index", np.arange(length)),
    ):
        if not np.array_equal(table[column].to_numpy(), expected):
            raise ValueError(f"Episode {index}: invalid {column} alignment")
    action = physical_vector(np.stack(table["action"].to_numpy()))
    state = physical_vector(np.stack(table["observation.state"].to_numpy()))
    if not (np.isfinite(action).all() and np.isfinite(state).all()):
        raise ValueError(f"Episode {index}: non-finite action/state")
    if max(rot6d_error(action), rot6d_error(state)) > 1e-3:
        raise ValueError(f"Episode {index}: rotation columns are not orthonormal")

    rows, by_start = [], {segment["start"]: segment for segment in segments}
    sources = [av.open(str(path)) for path in episode["videos"]]
    writer = None
    decoded = 0
    try:
        for stream in (source.streams.video[0] for source in sources):
            stream.codec_context.thread_count = 1
        decoders = [source.decode(source.streams.video[0]) for source in sources]
        # strict=True rejects cameras whose frame counts differ.
        for decoded, frames in enumerate(zip(*decoders, strict=True)):
            if decoded in by_start:
                current = by_start[decoded]
                stem = f"episode_{index:06d}_seg_{current['seg_id']:03d}"
                destination = output / "videos_original" / f"{stem}.mp4"
                temporary = destination.with_suffix(".partial.mp4")
                writer = (*_open_writer(temporary), temporary, destination, current, stem)
            if writer is not None:
                container, stream, temporary, destination, current, stem = writer
                images = [frame.to_ndarray(format="rgb24") for frame in frames]
                _encode(container, stream, compose_frame(*images), decoded - current["start"])
                if decoded + 1 == current["end"]:
                    clip_length = current["end"] - current["start"]
                    _finish_clip(container, stream, temporary, destination, clip_length)
                    writer = None
                    parquet = f"data/{stem}.parquet"
                    pq.write_table(
                        _segment_table(action, state, current),
                        output / parquet,
                        compression="zstd",
                    )
                    rows.append(
                        dict(
                            episode_index=index,
                            seg_id=current["seg_id"],
                            segment_type=current["type"],
                            hand=current["hand"],
                            object=current["object"],
                            instruction=current["instruction"],
                            n_candidates_same_side=current["n_candidates_same_side"],
                            video_path=f"videos_original/{stem}.mp4",
                            parquet_path=parquet,
                            fps=FPS,
                            source_start=current["start"],
                            source_end=current["end"],
                            trim_start=0,
                            trim_end=clip_length,
                            num_frames=clip_length,
                            grasp=current["grasp"],
                            release=current["release"],
                        )
                    )
    finally:
        for source in sources:
            source.close()
    if decoded + 1 != length:
        raise ValueError(f"Episode {index}: decoded {decoded + 1} frames, expected {length}")
    if len(rows) != len(segments):
        raise ValueError(f"Episode {index}: wrote {len(rows)}/{len(segments)} clips")
    return rows


def convert(source, annotation, output, workers=16, limit=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("workers and limit must be positive")
    info, episodes = load_episodes(source)
    segments = load_segments(annotation, {e["index"]: e["length"] for e in episodes})
    selected = [e for e in episodes if e["index"] in segments]
    complete = limit is None
    if limit is not None:
        selected = selected[:limit]
    output.mkdir(parents=True)
    (output / "data").mkdir()
    (output / "videos_original").mkdir()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(convert_episode, e, segments[e["index"]], output) for e in selected]
        for completed, job in enumerate(jobs, 1):
            rows.extend(job.result())
            if completed % 10 == 0 or completed == len(jobs):
                print(f"Converted {completed}/{len(jobs)} episodes", flush=True)
    # Publish metadata only after every clip has passed conversion.
    pd.DataFrame(rows).to_csv(output / "segments.csv", index=False)
    names = {
        name: [info["features"][name]["names"][i] for i in VECTOR_SOURCE_COLUMNS]
        for name in ("action", "observation.state")
    }
    summary = dict(
        complete=complete,
        episodes=len(selected),
        segments=len(rows),
        frames=sum(r["num_frames"] for r in rows),
        fps=FPS,
        video_size=[CANVAS_SIZE_WH[1], CANVAS_SIZE_WH[0]],
        source=str(source),
        annotation=str(Path(annotation).resolve()),
        rot6d_layout=SOURCE_ROT6D_LAYOUT,
        vector_names=names,
    )
    (output / "conversion.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {k: v for k, v in summary.items() if k != "vector_names"}


def _load_clip(media_root, row):
    table = pd.read_parquet(media_root / row.parquet_path)
    length = int(row.num_frames)
    for column in ("frame_index", "raw_frame_index"):
        if not np.array_equal(table[column], np.arange(length)):
            raise ValueError(f"Expected identity frame mapping: {row.parquet_path}/{column}")
    action, state = np.stack(table.action), np.stack(table["observation.state"])
    if (
        action.shape != (length, VECTOR_DIM)
        or state.shape != action.shape
        or not (np.isfinite(action).all() and np.isfinite(state).all())
    ):
        raise ValueError(f"Invalid action/state: {row.parquet_path}")
    return action, state


def _statistics(values):
    stacked = np.concatenate(values).astype(np.float64)
    std = stacked.std(axis=0)
    if float(std.min()) < MIN_STD:
        raise ValueError(f"Near-constant dimensions: {np.flatnonzero(std < MIN_STD).tolist()}")
    return dict(
        default=dict(
            global_min=stacked.min(axis=0).tolist(),
            global_max=stacked.max(axis=0).tolist(),
            global_mean=stacked.mean(axis=0).tolist(),
            global_std=std.tolist(),
            global_q01=np.quantile(stacked, 0.01, axis=0).tolist(),
            global_q99=np.quantile(stacked, 0.99, axis=0).tolist(),
        )
    )


def _annotated_events(row, hand):
    """Annotated grasp/release as sampling events; the hands have no single gripper channel."""
    events = []
    for name in ("grasp", "release"):
        value = getattr(row, name)
        if not pd.isna(value):
            events.append(
                dict(
                    arm=hand,
                    start=max(0, int(value) - EVENT_HALF_WINDOW),
                    end=min(int(row.num_frames), int(value) + EVENT_HALF_WINDOW),
                    event_type=name,
                    accepted=True,
                )
            )
    return events


def prepare(media_root, holdout, output, end_effector="robot hand"):
    """Write prompts, the episode-level split, training statistics and the sampling index."""
    from .datasets.lerobot.critical_phase_sampling import (
        SCHEMA,
        SCORE_RECIPE,
        EventRules,
        row_key,
        row_masks,
    )

    media_root, output = Path(media_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    conversion = json.loads((media_root / "conversion.json").read_text())
    df = pd.read_csv(media_root / "segments.csv", keep_default_na=False, na_values=[""])
    df["task_name"] = df.segment_type.str.lower()
    df["dimension"] = "marvin"
    df["hand"] = df.hand.fillna("")
    df["object"] = df.object.fillna("")
    df["segment_terminal"] = True
    df["high_level_instruction"] = df.instruction
    df["training_prompt"] = [
        build_prompt(row.segment_type, row.hand, row.object, end_effector)
        for row in df.itertuples(index=False)
    ]
    prompts = list(dict.fromkeys(df.training_prompt))
    prompt_files = {p: f"text_embeds/prompt_{i:05d}.pt" for i, p in enumerate(prompts)}
    df["text_embedding_path"] = df.training_prompt.map(prompt_files)

    heldout = {int(e) for e in json.loads(Path(holdout).read_text())["held_out_episodes"]}
    missing = heldout - set(df.episode_index)
    if missing and conversion["complete"]:
        raise ValueError(f"Held-out episodes without segments: {sorted(missing)}")
    is_val = df.episode_index.isin(heldout)
    train, val = df[~is_val].copy(), df[is_val].copy()
    if train.empty or val.empty:
        raise ValueError("Both the training and held-out splits need segments")

    rules = EventRules(before=4, after=2, fps=FPS)
    actions, states, keys, masks, cores, offsets, report = [], [], [], [], [], [0], []
    for row in train.itertuples(index=False):
        action, state = _load_clip(media_root, row)
        actions.append(action)
        states.append(state)
        length = int(row.num_frames)
        mask, core, _, accepted, truncated = row_masks(
            _annotated_events(row, row.hand),
            0,
            length,
            rules,
            prefix=CRITICAL_PREFIX_ACTIONS,
        )
        if truncated:
            raise ValueError(f"Clipped events cannot be truncated: {row.parquet_path}")
        keys.append(
            row_key(row.parquet_path, row.video_path, 0, length, 0, length, row.training_prompt)
        )
        masks.append(mask)
        cores.append(core)
        offsets.append(offsets[-1] + len(mask))
        report.append(
            dict(
                episode=int(row.episode_index),
                seg=int(row.seg_id),
                events=len(accepted),
                frames=length,
            )
        )
    for row in val.itertuples(index=False):
        _load_clip(media_root, row)
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate sampling row identities")
    # Statistics use training segments only, so held-out data cannot leak into normalization.
    statistics = dict(
        action=_statistics(actions),
        state=_statistics(states),
        num_episodes=int(train.episode_index.nunique()),
        num_transition=int(train.num_frames.sum()),
    )

    output.mkdir(parents=True)
    df.to_csv(output / "all.csv", index=False)
    train.to_csv(output / "train.csv", index=False)
    val.to_csv(output / "val.csv", index=False)
    (output / "dataset_stats.json").write_text(json.dumps(statistics, indent=2) + "\n")
    (output / "prompts.json").write_text(
        json.dumps(prompt_files, ensure_ascii=False, indent=2) + "\n"
    )
    index = output / "critical_index"
    index.mkdir()
    np.savez_compressed(
        index / "start_weights.npz",
        keys=np.asarray(keys),
        offsets=np.asarray(offsets, dtype=np.int64),
        critical=np.concatenate(masks),
        core=np.concatenate(cores),
    )
    manifest = dict(
        schema=SCHEMA,
        rows=len(keys),
        rules=dict(before=rules.before, after=rules.after, fps=FPS, half_window=EVENT_HALF_WINDOW),
        sampling=SCORE_RECIPE,
        review_status="validated",
        validation="human_reviewed_grasp_release_annotations",
        contract=dict(
            action_horizon=ACTION_HORIZON,
            stride=1,
            action_offset=0,
            prefix=CRITICAL_PREFIX_ACTIONS,
            fps=FPS,
        ),
    )
    (index / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (index / "events_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = dict(
        train_segments=len(train),
        val_segments=len(val),
        train_frames=int(train.num_frames.sum()),
        val_frames=int(val.num_frames.sum()),
        heldout_episodes=sorted(heldout),
        prompts=len(prompts),
        end_effector=end_effector,
        media_root=str(media_root),
        complete_conversion=bool(conversion["complete"]),
        statistics_source="training segments only",
        smallest_action_std=float(np.min(statistics["action"]["default"]["global_std"])),
        smallest_state_std=float(np.min(statistics["state"]["default"]["global_std"])),
    )
    (output / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
