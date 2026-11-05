import unittest

from run_vpp2_client import client_args
from sharpa_marvin_client.config import load_config, parse_openwam_args
from sharpa_marvin_client.video import validate_video_crop


class VPP2ClientTests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        args = parse_openwam_args(client_args([]))
        config, _ = load_config(args.config)
        self.assertEqual((args.execution_mode, args.execute_horizon, args.control_hz), ("sync", 30, 30))
        self.assertFalse(args.execute)
        self.assertEqual(args.video_crop, [184, 135, 600, 450])
        self.assertEqual(config["openwam_policy"]["host"], "127.0.0.1")
        self.assertEqual(config["openwam_policy"]["port"], 8851)
        self.assertEqual(config["execution_horizon"], 30)
        self.assertEqual(config["rates"]["action_hz"], 30.0)
        args = parse_openwam_args(client_args(["--host", "127.0.0.1", "--execute-horizon", "15"]))
        self.assertEqual((args.host, args.execute_horizon), ("127.0.0.1", 15))

    def test_crop_is_the_training_view(self):
        camera = load_config(parse_openwam_args(client_args([])).config)[0]["camera"]
        x, y, width, height = validate_video_crop(
            parse_openwam_args(client_args([])).video_crop,
            raw_width=camera["raw_width"], raw_height=camera["raw_height"],
        )
        self.assertEqual(width * 3, height * 4)
        # The center of the 600x480 recording crop at (184, 120).
        self.assertEqual((x, y + height / 2), (184, 120 + 240))

    def test_incompatible_modes_rejected(self):
        for args in (["--execution-mode", "single"], ["--execution-mode", "training_rtc"],
                     ["--control-hz", "40"], ["--execute-horizon", "32"], ["--execute-horizon", "16"]):
            with self.assertRaises(ValueError):
                client_args(args)


if __name__ == "__main__":
    unittest.main()
