import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from active_adaptation.utils.window_load_capacity import WindowLoadCapacityLookup
from scripts.build_l7_window_cap_labels import LABEL_FIELDS, read_rollout_csvs
from scripts.finalize_l7_window_caps import split_feasible_rows
from scripts.render_l7_expert_videos import parse_clip_spec, select_clips
from scripts.utils.l7_load_eval import window_bounds


class L7LoadPipelineTest(unittest.TestCase):
    def test_video_clip_load_parsing(self):
        self.assertEqual(parse_clip_spec("3:10", None, 0.0), (3, 10, None, 0.0))
        self.assertEqual(parse_clip_spec("3:10:12", None, 0.0), (3, 10, None, 12.0))
        self.assertEqual(parse_clip_spec("3:10:20:5", None, 0.0), (3, 10, 20.0, 5.0))
        with self.assertRaises(ValueError):
            parse_clip_spec("3:10:20:5:1", None, 0.0)

        dataset = SimpleNamespace(
            motion_to_dataset_id=torch.tensor([0, 0, 0]),
            global_lengths=torch.tensor([10, 20, 20]),
        )
        self.assertEqual(len(select_clips(dataset, clip_steps=10, seed=0, group_quotas=(3,))), 3)

    def test_rollout_csv_rejects_missing_shards_and_duplicate_windows(self):
        row = {
            "motion_id": "0",
            "source_path": "a.npy",
            "window_idx": "0",
            "window_start_frame": "0",
            "window_end_frame": "250",
            "max_success_load_kg": "30",
            "status": "success",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for shard in (0, 2):
                with (root / f"shard_{shard}.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "Expected rollout shards"):
                read_rollout_csvs(str(root / "shard_*.csv"), LABEL_FIELDS, expected_shards=2)
            with self.assertRaisesRegex(ValueError, "Duplicate rollout window"):
                read_rollout_csvs(str(root / "shard_*.csv"), LABEL_FIELDS)

            tail = root / "tail.csv"
            tail_row = {**row, "expected_window_count": "2"}
            with tail.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(tail_row))
                writer.writeheader()
                writer.writerow(tail_row)
            with self.assertRaisesRegex(ValueError, "Missing rollout windows"):
                read_rollout_csvs(str(tail), LABEL_FIELDS)

    def test_windows_are_half_open(self):
        self.assertEqual(window_bounds(10, 5), [(0, 5), (5, 10)])
        self.assertEqual(window_bounds(251, 250), [(0, 250), (250, 251)])
        self.assertEqual(window_bounds(687, 250), [(0, 250), (250, 500), (500, 687)])
        self.assertEqual(window_bounds(100, 250), [(0, 100)])

    def test_any_failed_window_removes_the_whole_motion(self):
        rows = [
            {"teacher_mem_path": "l7/a", "local_motion_id": "0", "status": "success"},
            {"teacher_mem_path": "l7/a", "local_motion_id": "0", "status": "failed_at_zero"},
            {"teacher_mem_path": "l7/a", "local_motion_id": "1", "status": "success"},
        ]
        feasible, invalid = split_feasible_rows(rows)
        self.assertEqual(invalid, {("l7/a", 0)})
        self.assertEqual([row["local_motion_id"] for row in feasible], ["1"])

    def test_lookup_covers_final_frame_and_remaps_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.npz"
            np.savez(
                path,
                motion_files=np.asarray(["/data/b.npz", "/data/a.npz"]),
                bin_motion_idx=np.asarray([0, 1, 1]),
                bin_idx=np.asarray([0, 0, 1]),
                start_frame=np.asarray([0, 0, 5]),
                end_frame=np.asarray([5, 5, 10]),
                window_cap_kg=np.asarray([22.0, 11.0, 12.0], dtype=np.float32),
            )
            lookup = WindowLoadCapacityLookup.from_label_file(
                path,
                motion_source_paths=["/data/a.npz", "/data/b.npz"],
                motion_labels=[{"source_path": "/data/a.npz"}, {"source_path": "/data/b.npz"}],
                motion_lengths=[10, 5],
                motion_fps=50.0,
                device="cpu",
                missing_motion_policy="error",
            )

            result = lookup.lookup(torch.tensor([0, 0, 1]), torch.tensor([0, 9, 0]))
            self.assertTrue(result["valid"].all())
            self.assertEqual(result["cap_kg"].tolist(), [11.0, 12.0, 22.0])


if __name__ == "__main__":
    unittest.main()
