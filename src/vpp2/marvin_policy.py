"""VPP2 joint checkpoint behind the Marvin protocol, with the inputs training saw."""

import json
from pathlib import Path
import re
from types import SimpleNamespace

from hydra.utils import instantiate
import pandas as pd
import torch
import torchvision.transforms.functional as transforms_F

from .cli import load_config
from .datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from .evaluation import _ActionMetrics
from .marvin_data import build_prompt, compose_frame
from .marvin_protocol import ACTION_HORIZON, History

VIDEO_SIZE_HW = (240, 416)
# Nine condition frames and eight future frames at the training stride.
NUM_VIDEO_FRAMES = 17
CONTEXT_LEN = 128
PICK_INSTRUCTION = re.compile(r"Pick up the (.+) with the (left|right) hand and put it in the bin\.?")


def load_processor(cfg, prepared):
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(
        load_dataset_stats_from_json(str(Path(prepared) / "dataset_stats.json"))
    )
    return processor


def load_runtime(checkpoint, *, config, wan, prepared, device="cuda:0", text_encoder=False):
    """The joint model from one checkpoint; no data or initial weights are needed."""
    cfg = load_config(config, [
        f"paths.wan={wan}",
        f"paths.video_init={checkpoint}",
        "paths.action_init=null",
        f"paths.data={prepared}",
        "model.skip_dit_load_from_pretrain=true",
        f"model.load_text_encoder={'true' if text_encoder else 'false'}",
        "model.video_dit_config.use_gradient_checkpointing=false",
        "model.action_dit_config.use_gradient_checkpointing=false",
    ])
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
    step = model.load_checkpoint(str(checkpoint))
    if step is None:
        raise ValueError(f"{checkpoint} has no training step")
    return model.eval().requires_grad_(False), load_processor(cfg, prepared), step


class PromptTable:
    """Task text to training context, from the cached embeddings of the prepared directory."""

    def __init__(self, prepared, encode=None):
        self.prepared = Path(prepared)
        self.encode = encode
        self.files = json.loads((self.prepared / "prompts.json").read_text())
        self.prompts = {}
        for name in ("train.csv", "val.csv"):
            table = pd.read_csv(self.prepared / name, usecols=["instruction", "training_prompt"], keep_default_na=False)
            self.prompts.update(zip(table.instruction, table.training_prompt))
        self.contexts = {}

    def training_prompt(self, task):
        task = task.strip()
        if task in self.files:
            return task
        if task in self.prompts:
            return self.prompts[task]
        match = PICK_INSTRUCTION.fullmatch(task)
        if match is None:
            raise ValueError(
                f"Unknown task {task!r}: use a training prompt or 'Pick up the X with the left or "
                "right hand and put it in the bin.'"
            )
        return build_prompt("PICK", match[2], match[1])

    def _load(self, prompt):
        if prompt in self.files:
            payload = torch.load(self.prepared / self.files[prompt], map_location="cpu")
            context, mask = payload["context"], payload.get("mask", payload.get("context_mask")).bool()
            padded = context.new_zeros((CONTEXT_LEN, context.shape[1]))
            length = min(CONTEXT_LEN, context.shape[0])
            padded[:length] = context[:length] * mask[:length, None]
            return padded
        if self.encode is None:
            raise ValueError(f"No cached embedding for {prompt!r}; start the server with --text-encoder")
        return self.encode(prompt)

    def context(self, task):
        prompt = self.training_prompt(task)
        if prompt not in self.contexts:
            self.contexts[prompt] = self._load(prompt)
        return self.contexts[prompt]


def prepare_frame(images):
    """Three wire views to the normalized [3,H,W] frame of a training clip."""
    canvas = compose_frame(images["head_camera"], images["left_wrist_camera"], images["right_wrist_camera"])
    frame = torch.from_numpy(canvas.copy()).permute(2, 0, 1).float().div(255.0)
    frame = transforms_F.resize(
        frame, size=list(VIDEO_SIZE_HW),
        interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True,
    )
    return frame.mul(2.0).sub(1.0)


def normalize_state(processor, state):
    key = processor.shape_meta["state"][0]["key"]
    batch = dict(state={key: torch.tensor(state, dtype=torch.float32)[None]})
    return processor.normalizer.forward(batch)["state"][key]


def denormalize_action(processor, action, proprio):
    owner = SimpleNamespace(dataset=SimpleNamespace(processor=processor))
    return _ActionMetrics._denormalize_action(owner, action, proprio).numpy()


class MarvinPolicy:
    def __init__(self, model, processor, prompts, *, steps=10, seed=None):
        self.model, self.processor, self.prompts = model, processor, prompts
        self.steps, self.seed = steps, seed
        self.reset()

    def reset(self):
        self.history = History()

    def sample(self, condition, proprio, context, seed):
        """One normalized [32,62] chunk from a normalized [3,9,H,W] condition."""
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            chunk = self.model.infer_action(
                prompt=None,
                input_image=condition[:, 0].unsqueeze(0),
                action_horizon=ACTION_HORIZON,
                condition_video=condition,
                proprio=proprio[0],
                context=context,
                context_mask=torch.ones(context.shape[0], dtype=torch.bool),
                num_video_frames=NUM_VIDEO_FRAMES,
                num_inference_steps=self.steps,
                seed=seed,
            )["action"]
        return chunk.detach().to("cpu", torch.float32).reshape(ACTION_HORIZON, -1)

    def predict(self, observation, prompt, step):
        context = self.prompts.context(prompt)
        self.history.add(step, prepare_frame(observation["images"]))
        condition = torch.stack(self.history.condition(step), dim=1)
        proprio = normalize_state(self.processor, observation["state"])
        seed = None if self.seed is None else self.seed + step
        return denormalize_action(self.processor, self.sample(condition, proprio, context, seed), proprio)
