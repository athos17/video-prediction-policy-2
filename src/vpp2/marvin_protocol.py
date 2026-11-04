"""Marvin synchronous WebSocket protocol for VPP2, independent of torch and ROS.

Speaks the JSON protocol of sharpa-marvin-client's OpenWamPolicyClient: ping, reset, obs_chunk.
The robot's clock owns the history. A session counts executed actions, not seconds, so time spent
waiting for inference never enters the one second history stride VPP2 was trained with.
"""

import base64
from io import BytesIO
import json
import logging
import time

import numpy as np
from PIL import Image

ACTION_LAYOUT = "left_eef9,left_hand22,right_eef9,right_hand22"
ACTION_HZ = 30.0
ACTION_HORIZON = 32
RAW_ACTION_DIM = 62
ROT6D_STARTS = (3, 34)
WIRE_CAMERAS = ("head_camera", "left_wrist_camera", "right_wrist_camera")
WIRE_SIZE_WH = (320, 240)
MAX_MESSAGE_BYTES = 4 * 1024 * 1024
MAX_PNG_BASE64 = 1024 * 1024
HISTORY_FRAMES = 8
HISTORY_STRIDE = 30
# Divisors of the history stride, so that every history frame was really observed.
CHUNK_SIZES = (1, 2, 3, 5, 6, 10, 15, 30)


def condition_steps(step):
    """Steps of the condition: the first frame, then 8 frames ending now, clamped at the start."""
    history = np.arange(1 - HISTORY_FRAMES, 1) * HISTORY_STRIDE + step
    return [0, *np.maximum(history, 0).tolist()]


class History:
    """Frames of one episode, indexed by executed action step."""

    def __init__(self):
        self.frames = {}

    def add(self, step, frame):
        self.frames[step] = frame
        oldest = step - (HISTORY_FRAMES - 1) * HISTORY_STRIDE
        self.frames = {key: value for key, value in self.frames.items() if key == 0 or key >= oldest}

    def condition(self, step):
        steps = condition_steps(step)
        missing = sorted(set(steps) - set(self.frames))
        if missing:
            raise ValueError(f"History frames at steps {missing} were not observed")
        return [self.frames[key] for key in steps]


def validate_vectors(value, shape):
    """Finite vectors with non-degenerate first-two-column rotations."""
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Expected finite array {shape}, got {array.shape}")
    for start in ROT6D_STARTS:
        first, second = array[..., start : start + 3], array[..., start + 3 : start + 6]
        if (np.linalg.norm(first, axis=-1) < 1e-8).any() or (
            np.linalg.norm(np.cross(first, second), axis=-1) < 1e-8
        ).any():
            raise ValueError("Degenerate column ROT6D")
    return array


def decode_observation(request):
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
        raise ValueError("prompt must be a nonempty string of at most 4096 characters")
    images = request.get("images")
    if not isinstance(images, dict) or set(images) != set(WIRE_CAMERAS):
        raise ValueError(f"images must contain exactly {list(WIRE_CAMERAS)}")
    decoded = {}
    for key in WIRE_CAMERAS:
        encoded = images[key]
        if not isinstance(encoded, str) or len(encoded) > MAX_PNG_BASE64:
            raise ValueError(f"Invalid PNG payload: {key}")
        with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != WIRE_SIZE_WH:
                raise ValueError(f"{key} must be an RGB PNG of size {WIRE_SIZE_WH}")
            decoded[key] = np.asarray(image).copy()
    return dict(images=decoded, state=validate_vectors(request.get("state"), (RAW_ACTION_DIM,))), prompt


class HoldPolicy:
    """Commands the current state, for testing the link without weights or a GPU."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = History()

    def predict(self, observation, prompt, step):
        self.history.add(step, observation["images"])
        self.history.condition(step)
        return np.tile(observation["state"], (ACTION_HORIZON, 1))


class SyncSession:
    """One connection owns one episode. A failure requires a reset before continuing."""

    def __init__(self, policy, metadata=None):
        self.policy = policy
        self.metadata = metadata or {}
        self.reset()

    def reset(self):
        self.policy.reset()
        self.step = 0
        self.count = None
        self.prompt = None
        self.failed = False

    def handle(self, request):
        if not isinstance(request, dict):
            raise ValueError("Message must be a JSON object")
        kind = request.get("type")
        if kind == "ping":
            return dict(
                self.metadata, type="pong", model_family="vpp2", representation="eef",
                raw_action_dim=RAW_ACTION_DIM, rot6d_convention="column", action_layout=ACTION_LAYOUT,
                inference_mode="sync", supported_requests=["ping", "reset", "obs_chunk"],
                action_horizon=ACTION_HORIZON, max_action_chunk=max(CHUNK_SIZES), action_hz=ACTION_HZ,
                supported_chunk_sizes=list(CHUNK_SIZES),
            )
        if kind == "reset":
            self.reset()
            return dict(type="reset_ack")
        if kind != "obs_chunk":
            raise ValueError(f"Unsupported request {kind!r}; only synchronous obs_chunk is implemented")
        if self.failed:
            raise ValueError("Previous inference failed; reset the session before retrying")
        count = request.get("count")
        if type(count) is not int or count not in CHUNK_SIZES:
            raise ValueError(f"count must be one of {CHUNK_SIZES}")
        if self.count is not None and count != self.count:
            raise ValueError("Changing the chunk count requires a reset, it breaks history alignment")
        observation, prompt = decode_observation(request)
        if self.prompt is not None and prompt != self.prompt:
            raise ValueError("Changing the prompt starts a new episode and requires a reset")
        started = time.perf_counter()
        try:
            action = validate_vectors(
                self.policy.predict(observation, prompt, self.step), (ACTION_HORIZON, RAW_ACTION_DIM)
            )
        except Exception:
            self.failed = True
            raise
        response = dict(
            type="action_chunk", actions=action[:count].tolist(), count=count, step=self.step,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        self.step += count
        self.count, self.prompt = count, prompt
        return response

    def dispatch(self, raw):
        try:
            response = self.handle(json.loads(raw))
        except (ValueError, TypeError, OSError) as error:
            response = dict(type="error", code="validation_error", message=str(error))
        except Exception as error:
            logging.exception("Marvin inference failed")
            response = dict(type="error", code="inference_error", message=str(error))
        return json.dumps(response, allow_nan=False, separators=(",", ":"))
