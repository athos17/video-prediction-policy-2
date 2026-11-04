"""Protocol and history tests without torch: PYTHONPATH=src pytest scripts/marvin/test_marvin_protocol.py"""

import base64
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest

from vpp2.marvin_protocol import (
    CHUNK_SIZES,
    HISTORY_STRIDE,
    HoldPolicy,
    SyncSession,
    condition_steps,
    validate_vectors,
)


def png(value=10, size=(320, 240), mode="RGB"):
    buffer = BytesIO()
    Image.new(mode, size, value if mode == "L" else (value,) * 3).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def state():
    vector = np.zeros(62, dtype=np.float32)
    vector[3:9] = vector[34:40] = [1, 0, 0, 0, 1, 0]
    return vector.tolist()


def request(count=30, prompt="task", **changes):
    images = dict(head_camera=png(), left_wrist_camera=png(), right_wrist_camera=png())
    return dict(dict(type="obs_chunk", images=images, state=state(), prompt=prompt, count=count), **changes)


def send(session, message):
    return json.loads(session.dispatch(json.dumps(message)))


def test_condition_matches_the_dataset_grid():
    assert condition_steps(0) == [0] * 9
    assert condition_steps(70) == [0, 0, 0, 0, 0, 0, 10, 40, 70]
    assert condition_steps(300) == [0, 90, 120, 150, 180, 210, 240, 270, 300]


def test_ping_satisfies_the_openwam_client_contract():
    pong = send(SyncSession(HoldPolicy()), dict(type="ping"))
    assert pong["type"] == "pong" and pong["model_family"] == "vpp2"
    assert (pong["representation"], pong["raw_action_dim"], pong["rot6d_convention"]) == ("eef", 62, "column")
    assert pong["action_layout"] == "left_eef9,left_hand22,right_eef9,right_hand22"
    assert pong["inference_mode"] == "sync" and "obs_chunk" in pong["supported_requests"]
    assert pong["action_hz"] == 30.0 and 30 in pong["supported_chunk_sizes"]


@pytest.mark.parametrize("count", CHUNK_SIZES)
def test_every_chunk_size_observes_its_whole_history(count):
    session = SyncSession(HoldPolicy())
    for index in range(600 // count):
        reply = send(session, request(count))
        assert reply["type"] == "action_chunk", reply
        assert reply["step"] == index * count and len(reply["actions"]) == count
        assert np.allclose(reply["actions"][0], state())
    # The first frame and a window of seven strides.
    assert len(session.policy.history.frames) == 2 + 7 * HISTORY_STRIDE // count


def test_chunk_sizes_divide_the_stride():
    assert all(HISTORY_STRIDE % count == 0 and count <= 32 for count in CHUNK_SIZES)
    assert send(SyncSession(HoldPolicy()), request(32))["type"] == "error"


def test_history_gap_fails_and_requires_a_reset():
    session = SyncSession(HoldPolicy())
    for _ in range(2):
        send(session, request())
    session.step = 45
    assert "not observed" in send(session, request())["message"]
    assert "reset" in send(session, request())["message"]
    assert send(session, dict(type="reset"))["type"] == "reset_ack"
    assert send(session, request())["step"] == 0


def test_episode_rules():
    session = SyncSession(HoldPolicy())
    send(session, request())
    assert "chunk count" in send(session, request(15))["message"]
    assert "prompt" in send(session, request(prompt="other"))["message"]
    send(session, dict(type="reset"))
    assert send(session, request(15, prompt="other"))["step"] == 0


@pytest.mark.parametrize("changes", [
    dict(state=[0.0] * 62),
    dict(state=state()[:61]),
    dict(state=[float("nan")] + state()[1:]),
    dict(prompt=" "),
    dict(images=dict(head_camera=png())),
    dict(images=dict(head_camera=png(size=(640, 480)), left_wrist_camera=png(), right_wrist_camera=png())),
    dict(images=dict(head_camera=png(mode="L"), left_wrist_camera=png(), right_wrist_camera=png())),
    dict(images=dict(head_camera="not base64!", left_wrist_camera=png(), right_wrist_camera=png())),
])
def test_bad_observations_are_rejected_without_advancing(changes):
    session = SyncSession(HoldPolicy())
    assert send(session, request(**changes))["code"] == "validation_error"
    assert session.step == 0 and not session.failed


def test_policy_errors_are_reported_and_sticky():
    class Broken(HoldPolicy):
        def predict(self, observation, prompt, step):
            raise RuntimeError("out of memory")

    session = SyncSession(Broken())
    assert send(session, request())["code"] == "inference_error"
    assert "reset" in send(session, request())["message"]


def test_invalid_model_output_never_reaches_the_robot():
    class Nan(HoldPolicy):
        def predict(self, observation, prompt, step):
            return np.full((32, 62), np.nan)

    assert send(SyncSession(Nan()), request())["type"] == "error"
    with pytest.raises(ValueError):
        validate_vectors(np.zeros((32, 62)), (32, 62))
