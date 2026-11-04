"""Serve a VPP2 joint checkpoint to sharpa-marvin-client over WebSocket.

--mock answers with the current state and needs neither torch nor weights, to test the link.
"""

import argparse
import json
from pathlib import Path
import threading

from .marvin_protocol import MAX_MESSAGE_BYTES, HoldPolicy, SyncSession


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, help="Joint weights, checkpoints/weights/step_*.pt")
    p.add_argument("--wan", type=Path, help="Wan2.1-I2V-14B-480P directory with the VAE and CLIP")
    p.add_argument("--prepared", type=Path, help="Prepared data directory: statistics and text embeddings")
    p.add_argument("--config", type=Path, default=Path("configs/marvin/train_joint.yaml"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8851)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=10, help="Action denoising steps, as in evaluation")
    p.add_argument("--seed", type=int, help="Fixed noise; the default samples new noise every request")
    p.add_argument("--text-encoder", action="store_true", help="Also encode prompts absent from the cache")
    p.add_argument("--mock", action="store_true")
    return p


def make_handler(policy, metadata=None):
    """One model serves one robot: another connection is refused rather than queued behind it."""
    owner = threading.Lock()

    def handler(socket):
        if not owner.acquire(blocking=False):
            socket.send(json.dumps(dict(type="error", code="server_busy", message="Another client owns the model")))
            socket.close(code=1013, reason="One robot connection at a time")
            return
        print(f"client connected: {socket.remote_address}", flush=True)
        try:
            session = SyncSession(policy, metadata)
            for raw in socket:
                reply = session.dispatch(raw)
                print(reply[:160] if '"action_chunk"' not in reply[:40] else
                      f"step {session.step - session.count} count {session.count} "
                      f"{json.loads(reply)['latency_ms']:.0f} ms", flush=True)
                socket.send(reply)
        finally:
            policy.reset()
            owner.release()
            print("client disconnected", flush=True)

    return handler


def build_policy(args):
    if args.mock:
        return HoldPolicy(), dict(mock=True)
    if not all((args.checkpoint, args.wan, args.prepared)):
        raise SystemExit("A real model needs --checkpoint --wan --prepared, or use --mock")
    import torch

    from .marvin_policy import MarvinPolicy, PromptTable, load_runtime

    model, processor, step = load_runtime(
        args.checkpoint, config=args.config, wan=args.wan, prepared=args.prepared,
        device=args.device, text_encoder=args.text_encoder,
    )

    def encode(prompt):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model.encode_prompt(prompt)[0][0].to("cpu", torch.bfloat16)

    prompts = PromptTable(args.prepared, encode if args.text_encoder else None)
    policy = MarvinPolicy(model, processor, prompts, steps=args.steps, seed=args.seed)
    return policy, dict(checkpoint=str(args.checkpoint), checkpoint_step=step)


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 < args.port < 65536 or args.steps < 1:
        raise ValueError("Invalid port or denoising step count")
    policy, metadata = build_policy(args)
    from websockets.sync.server import serve

    with serve(make_handler(policy, metadata), args.host, args.port, max_size=MAX_MESSAGE_BYTES,
               compression=None, ping_interval=None) as server:
        print(f"VPP2 Marvin server listening on ws://{args.host}:{server.socket.getsockname()[1]}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
