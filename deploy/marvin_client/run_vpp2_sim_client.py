#!/usr/bin/env python3
"""Simulate the Marvin client on a recorded segment. Nothing here can reach a robot.

The real policy client, sync loop and QP preprocessing run unchanged. Only MarvinRosIO is replaced:
ReplayIO reads observations from a dataset clip and keeps prepared commands instead of publishing
them. The replay follows the recording, so the actions never change what is observed next.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd

from marvin_gr00t_adapter.contracts import LANGUAGE_KEY
from marvin_gr00t_adapter.observation import ObservationPacket
from sharpa_marvin_client.config import load_config, parse_openwam_args
from sharpa_marvin_client.executor import MarvinQpExecutor
from sharpa_marvin_client.image_ops import initialize_image_ops
from sharpa_marvin_client.logging import RunLogger
from sharpa_marvin_client.native_worker import build_native_worker
from sharpa_marvin_client.openwam_policy import (
    OpenWamChunkPolicy,
    OpenWamPolicyClient,
    _batch_eef_columns_to_rows,
    parse_openwam_actions,
)
from sharpa_marvin_client.runtime import run_sync, warmup_openwam_policy

from run_vpp2_client import client_args

FPS = 30
WIRE_SIZE_WH = (320, 240)
# The clip's canvas: a 640x480 head view, then left over right 240x240 wrist views.
HEAD_WIDTH, WRIST_SIZE = 640, 240


def split_canvas(canvas):
    views = (canvas[:, :HEAD_WIDTH], canvas[:WRIST_SIZE, HEAD_WIDTH:], canvas[WRIST_SIZE:, HEAD_WIDTH:])
    return [cv2.resize(view, WIRE_SIZE_WH, interpolation=cv2.INTER_AREA) for view in views]


def read_clip(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise FileNotFoundError(path)
    return frames


def rows(vectors):
    """Column ROT6D dataset vectors to the row ROT6D the client works in."""
    converted = np.array(vectors, dtype=np.float32)
    converted[:, :9] = _batch_eef_columns_to_rows(converted[:, :9])
    converted[:, 31:40] = _batch_eef_columns_to_rows(converted[:, 31:40])
    return converted


class ReplayIO:
    """The part of MarvinRosIO the sync loop uses, backed by one clip. It has no publisher."""

    publisher_hz = 200.0

    def __init__(self, frames, states):
        self.frames, self.states = frames, rows(states)
        self.step = 0
        self.image_count = 0
        self.commands = []
        self.last_observation_metrics = {}

    def feedback(self):
        state = self.states[self.step]
        eef = dict(left=state[:9].astype(np.float64), right=state[31:40].astype(np.float64))
        hands = dict(left=state[9:31].copy(), right=state[40:62].copy())
        return None, eef, hands

    def prime_command(self, target):
        pass

    def wait_for_fresh_observation(self, previous_count, instruction, timeout_sec):
        head, left, right = split_canvas(self.frames[self.step])
        state = self.states[self.step]
        self.image_count += 1
        observation = dict(
            video=dict(ego_view=head[None, None], left_wrist_view=left[None, None],
                       right_wrist_view=right[None, None]),
            state=dict(left_wrist_eef=state[:9][None, None], left_hand_joints=state[9:31][None, None],
                       right_wrist_eef=state[31:40][None, None], right_hand_joints=state[40:62][None, None]),
            language={LANGUAGE_KEY: [[instruction]]},
        )
        stamp = self.step / FPS
        return ObservationPacket(sequence=self.step, stamp=stamp, observation=observation,
                                 source_stamps=dict(image_current=stamp))

    def enqueue_chunk(self, targets):
        self.commands.extend(targets)
        self.step += len(targets)
        return len(targets)

    def wait_until_commands_consumed(self, stop_requested, timeout_sec):
        return True


class Recorder:
    """Keeps each requested chunk and its server latency for the summary."""

    def __init__(self, policy):
        self.policy = policy
        self.chunks = []

    def request(self, observation):
        steps, info, latency = self.policy.request(observation)
        self.chunks.append(dict(steps=steps, info=info, latency=latency))
        return steps, info, latency

    def reset(self):
        self.chunks.clear()
        self.policy.reset()


def flat(steps):
    return np.stack([
        np.concatenate((s.left_wrist_eef, s.left_hand_joints, s.right_wrist_eef, s.right_hand_joints))
        for s in steps
    ])


def summarize(recorder, logged, truth, states, horizon):
    rows_out, latencies, servers = [], [], []
    for index, chunk in enumerate(recorder.chunks):
        start = index * horizon
        predicted = flat(chunk["steps"])
        real = truth[start : start + horizon]
        current = states[start]
        arm = {}
        for side, slot in (("left", 0), ("right", 31)):
            error = np.linalg.norm(predicted[:, slot : slot + 3] - real[:, slot : slot + 3], axis=1)
            arm[side] = dict(
                mean_position_error_m=float(error.mean()),
                end_position_error_m=float(error[-1]),
                first_step_jump_m=float(np.linalg.norm(predicted[0, slot : slot + 3] - current[slot : slot + 3])),
                real_travel_m=float(np.linalg.norm(real[-1, slot : slot + 3] - real[0, slot : slot + 3])),
                predicted_travel_m=float(np.linalg.norm(predicted[-1, slot : slot + 3] - predicted[0, slot : slot + 3])),
            )
        hands = np.abs(np.concatenate((predicted[:, 9:31] - real[:, 9:31], predicted[:, 40:62] - real[:, 40:62])))
        rows_out.append(dict(chunk=index, step=start, hand_mean_error_rad=float(hands.mean()), **arm))
        latencies.append(chunk["latency"])
        servers.append(chunk["info"].get("server_latency_ms") or 0.0)
    positions = [row[side]["mean_position_error_m"] for row in rows_out for side in ("left", "right")]
    jumps = [row[side]["first_step_jump_m"] for row in rows_out for side in ("left", "right")]
    return dict(
        chunks=len(rows_out),
        published_commands=0,
        prepared_commands=logged["prepared"],
        eef_limited_steps=logged["eef_limited"],
        hand_limited_steps=logged["hand_limited"],
        request_latency_mean_s=float(np.mean(latencies)),
        request_latency_max_s=float(np.max(latencies)),
        server_latency_mean_ms=float(np.mean(servers)),
        mean_position_error_m=float(np.mean(positions)),
        max_first_step_jump_m=float(np.max(jumps)),
        per_chunk=rows_out,
    )


def limit_counts(log_dir):
    prepared = eef = hand = 0
    with open(Path(log_dir) / "command.jsonl") as stream:
        for line in stream:
            record = json.loads(line)
            record = record.get("payload", record)
            prepared += 1
            eef += min(record["eef_limit_scale"].values()) < 1.0
            hand += min(record["hand_limit_scale"].values()) < 1.0
    return dict(prepared=prepared, eef_limited=eef, hand_limited=hand)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--segment", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    own, forwarded = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if "--execute" in forwarded:
        parser.error("The simulator never publishes; use run_vpp2_client.py on the robot")
    args = parse_openwam_args(client_args(forwarded))

    table = pd.read_csv(own.csv, keep_default_na=False, na_values=[""])
    selected = table[(table.episode_index == own.episode) & (table.seg_id == own.segment)]
    if len(selected) != 1:
        raise ValueError(f"Episode {own.episode} segment {own.segment} is not in {own.csv}")
    row = selected.iloc[0]
    instruction = (args.task or row.instruction).strip()
    data = pd.read_parquet(own.media_root / row.parquet_path)
    states = np.stack(data["observation.state"].to_numpy()).astype(np.float32)
    actions = np.stack(data["action"].to_numpy()).astype(np.float32)
    frames = read_clip(own.media_root / row.video_path)
    if len(frames) != len(states):
        raise ValueError(f"{len(frames)} frames but {len(states)} states")

    config, _ = load_config(args.config)
    policy_config = config["openwam_policy"]
    build_native_worker()
    print(f"image backend: {initialize_image_ops()}")
    logger = RunLogger(own.output)
    io = ReplayIO(frames, states)
    client = OpenWamPolicyClient(
        args.host or policy_config["host"], args.port or int(policy_config["port"]),
        connect_timeout_sec=args.connect_timeout_sec or float(policy_config["connect_timeout_sec"]),
        request_timeout_sec=args.request_timeout_sec or float(policy_config["request_timeout_sec"]),
        expected_representation=policy_config["expected_representation"],
        expected_action_dim=policy_config["expected_action_dim"],
        expected_rot6d_convention=policy_config["expected_rot6d_convention"],
    )
    executor = None
    try:
        contract = client.ping_and_validate()
        if contract.get("model_family") != "vpp2":
            raise ValueError(f"Expected model_family=vpp2, got {contract.get('model_family')!r}")
        if float(contract["action_hz"]) != args.control_hz:
            raise ValueError(f"Client {args.control_hz} Hz differs from server {contract['action_hz']} Hz")
        print(json.dumps(contract, indent=2))
        recorder = Recorder(OpenWamChunkPolicy(client, args.execute_horizon))
        warmup_openwam_policy(io=io, policy=recorder, instruction=instruction,
                              timeout_sec=args.fresh_observation_timeout_sec, logger=logger)
        recorder.chunks.clear()
        # Every requested chunk has its whole execution inside the recording.
        available = (len(frames) - 1) // args.execute_horizon
        args.max_loops = min(args.max_loops or available, available)
        executor = MarvinQpExecutor(io=io, config=config, dry_run=True)
        logger.write_config(dict(
            argv=list(sys.argv), task=instruction, demonstration=row.instruction,
            episode=own.episode, segment=own.segment, contract=contract, dry_run=True,
            execute_horizon=args.execute_horizon, video_crop=args.video_crop,
        ))
        started = time.monotonic()
        run_sync(args=args, instruction=instruction, control_hz=args.control_hz, io=io,
                 policy=recorder, executor=executor, logger=logger)
        elapsed = time.monotonic() - started
    finally:
        if executor is not None:
            executor.close()
        try:
            client.reset()
        except Exception as error:
            print(f"final reset failed: {error}", file=sys.stderr)
        client.close()
        logger.close()

    summary = summarize(recorder, limit_counts(logger.log_dir), rows(actions), io.states, args.execute_horizon)
    summary.update(task=instruction, demonstration=row.instruction, episode=own.episode,
                   segment=own.segment, wall_time_s=elapsed, log_dir=str(logger.log_dir))
    (logger.log_dir / "sim_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    printed = {key: value for key, value in summary.items() if key != "per_chunk"}
    print(json.dumps(printed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
