#!/usr/bin/env python3
"""Run the Marvin control stack against a VPP2 server. Commands are published only with --execute."""
import argparse
from pathlib import Path
import sys

from sharpa_marvin_client.config import parse_openwam_args

ROOT = Path(__file__).resolve().parent
# Divisors of the one second history stride, so the server saw every history frame.
CHUNK_SIZES = (1, 2, 3, 5, 6, 10, 15, 30)
# The dataset's 600x480 head view (crop 184 120 600 480) center-cropped to 4:3, as training did.
VIDEO_CROP = ("184", "135", "600", "450")


def client_args(argv):
    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument("--execution-mode", default="sync")
    if selection.parse_known_args(argv)[0].execution_mode != "sync":
        raise ValueError("VPP2 supports only sync: it was not trained for RTC prefixes")
    forwarded = ["--config", str(ROOT / "config" / "vpp2_sync.yaml"), "--execution-mode", "sync",
                 "--execute-horizon", "30", "--control-hz", "30", "--video-crop", *VIDEO_CROP, *argv]
    args = parse_openwam_args(forwarded)
    if args.control_hz != 30.0:
        raise ValueError("VPP2 was trained at 30 Hz; --control-hz must be 30")
    if args.execute_horizon not in CHUNK_SIZES:
        raise ValueError(f"--execute-horizon must be one of {CHUNK_SIZES} for an exact history")
    return forwarded


def main(argv=None):
    forwarded = client_args(sys.argv[1:] if argv is None else argv)
    from run_openwam_client import main as run
    run(forwarded, expected_model_family="vpp2")


if __name__ == "__main__":
    main()
