import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from active_adaptation.envs.mdp.randomizations.core import window_cap_hand_load
from active_adaptation.utils.window_load_capacity import WindowLoadCapacityLookup


class WindowLoadCapacityTest(unittest.TestCase):
    def write_labels(self, path: Path, starts, ends):
        np.savez(
            path,
            motion_files=np.asarray(["/data/a.npz"]),
            bin_motion_idx=np.zeros(len(starts), dtype=np.int32),
            bin_idx=np.arange(len(starts), dtype=np.int32),
            start_frame=np.asarray(starts, dtype=np.int32),
            end_frame=np.asarray(ends, dtype=np.int32),
            window_cap_kg=np.ones(len(starts), dtype=np.float32),
        )

    def load(self, path: Path):
        return WindowLoadCapacityLookup.from_label_file(
            path,
            motion_source_paths=["/data/a.npz"],
            motion_labels=[{"source_path": "/data/a.npz"}],
            motion_lengths=[10],
            motion_fps=50.0,
            device="cpu",
            missing_motion_policy="error",
        )

    def test_legacy_final_endpoint_is_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.npz"
            self.write_labels(path, [0, 5], [5, 9])

            result = self.load(path).lookup(torch.tensor([0]), torch.tensor([9]))

            self.assertTrue(result["valid"].item())

    def test_gap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.npz"
            self.write_labels(path, [0, 6], [5, 10])

            with self.assertRaisesRegex(ValueError, "continuously cover"):
                self.load(path)

    def test_single_side_load_preserves_total_weight(self):
        load = object.__new__(window_cap_hand_load)
        load.env = SimpleNamespace(device=torch.device("cpu"))
        load.split_ratio_low = 0.35
        load.split_ratio_high = 0.65
        load.single_side_load_ratio = 1.0
        load.body_load_weights = torch.zeros((32, 2))
        env_ids = torch.arange(32)

        torch.manual_seed(0)
        load._sample_body_load_weights(env_ids)

        torch.testing.assert_close(load.body_load_weights.sum(dim=1), torch.ones(32))
        self.assertTrue(((load.body_load_weights == 0.0) | (load.body_load_weights == 1.0)).all())


if __name__ == "__main__":
    unittest.main()
