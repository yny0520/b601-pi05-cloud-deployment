import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from b601_lerobot_export import (
    assign_session_splits,
    build_plan,
    export_plan,
    summarize_plan,
)


class FakeDataset:
    instances = []
    lerobot_version = "test-double"

    @classmethod
    def create(cls, **kwargs):
        instance = cls(kwargs)
        cls.instances.append(instance)
        return instance

    def __init__(self, options):
        self.options = options
        self.frames = []
        self.saved_episodes = 0
        self.finalized = False

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self, parallel_encoding=False):
        self.saved_episodes += 1

    def finalize(self):
        self.finalized = True


class LeRobotExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="b601-lerobot-export-")
        self.root = Path(self.temp.name)
        FakeDataset.instances.clear()

    def tearDown(self):
        self.temp.cleanup()

    def make_episode(self, name, session, frame_count=32, outcome="success"):
        folder = self.root / "raw" / name
        (folder / "top").mkdir(parents=True)
        (folder / "hand").mkdir()
        for camera, color in (("top", (255, 0, 0)), ("hand", (0, 255, 0))):
            Image.new("RGB", (16, 12), color).save(folder / camera / "0000000.jpg")
        metadata = {
            "schema": "b601-raw-episode-v1", "status": "saved", "outcome": outcome,
            "joint_order": [f"J{i}" for i in range(1, 8)],
            "counts": {"teleop": frame_count},
        }
        quality = {
            "telemetry_records": frame_count,
            "write_errors": [],
            "queue_dropped": {},
            "telemetry_sequence_gaps": 0,
        }
        (folder / "metadata.json").write_text(json.dumps(metadata))
        (folder / "quality.json").write_text(json.dumps(quality))
        rows = []
        for i in range(frame_count):
            rows.append({
                "teleop_seq": i + 1,
                "teleop_session": session,
                "read_window_ns": [i * 71_428_571, i * 71_428_571 + 1],
                "state_deg": [float(i)] * 7,
                "action_position_reference_deg": [float(i + 1)] * 7,
                "demonstration_segment": True,
                "paused": False,
                "fault_hold": False,
                "images": {
                    camera: {"file": f"{camera}/0000000.jpg", "fresh": True,
                             "age_ms": 10.0, "reused": i > 0}
                    for camera in ("top", "hand")
                },
            })
        (folder / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        return folder

    def make_plan(self):
        raw = self.root / "raw"
        raw.mkdir()
        self.make_episode("episode-a", "session-a")
        self.make_episode("episode-b", "session-b")
        self.make_episode("episode-c", "session-c")
        return build_plan(raw)

    def test_audit_uses_successful_clean_episodes_and_reports_contract(self):
        self.make_plan()
        self.make_episode("episode-d", "session-d", outcome="failure")
        plan = build_plan(self.root / "raw")
        summary = summarize_plan(plan)
        self.assertEqual(summary["eligible_source_episode_count"], 3)
        self.assertEqual(summary["teleoperation_sessions"], 3)
        self.assertEqual(summary["camera_map"], {"top": "camera1", "hand": "camera2"})
        self.assertEqual(plan["rejected"][0]["reason"], "outcome_failure")

    def test_splits_keep_all_segments_from_one_session_together(self):
        episodes = []
        for session in "abcdefghi":
            episodes.append({"segments": [{"session": session}, {"session": session}]})
        assignment = assign_session_splits(episodes, seed=42)
        self.assertEqual(assignment, assign_session_splits(episodes, seed=42))
        self.assertEqual(set(assignment.values()), {"train", "validation", "test"})

    def test_export_writes_radians_standard_camera_keys_and_source_map(self):
        plan = self.make_plan()
        output = self.root / "lerobot"
        report = export_plan(plan, output, "pick and place", min_free_gib=0,
                             dataset_cls=FakeDataset)
        writer = FakeDataset.instances[-1]
        self.assertTrue(writer.finalized)
        self.assertEqual(writer.options["fps"], 14)
        self.assertEqual(writer.options["robot_type"], "b601")
        self.assertEqual(len(writer.frames), report["converted_frame_count"])
        frame = writer.frames[0]
        np.testing.assert_allclose(frame["observation.state"], np.deg2rad(np.zeros(7)), atol=1e-6)
        np.testing.assert_allclose(frame["action"], np.deg2rad(np.ones(7)), atol=1e-6)
        self.assertEqual(frame["observation.images.camera1"].shape, (12, 16, 3))
        self.assertEqual(frame["observation.images.camera2"].shape, (12, 16, 3))
        self.assertEqual(frame["task"], "pick and place")
        self.assertTrue((output / "source_frame_map.csv").is_file())
        self.assertTrue((output / "episode_split.json").is_file())
        self.assertTrue((output / "train_only_stats.json").is_file())

    def test_stale_image_splits_and_excludes_that_frame(self):
        self.make_plan()
        sample_path = self.root / "raw" / "episode-a" / "samples.jsonl"
        rows = [json.loads(line) for line in sample_path.read_text().splitlines()]
        rows[10]["images"]["top"]["age_ms"] = 250
        sample_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        plan = build_plan(self.root / "raw")
        self.assertGreater(plan["filter_counts"]["stale_top"], 0)
        self.assertTrue(plan["episodes"])

    def test_path_traversal_and_queue_write_errors_are_rejected(self):
        self.make_plan()
        sample_path = self.root / "raw" / "episode-a" / "samples.jsonl"
        rows = [json.loads(line) for line in sample_path.read_text().splitlines()]
        rows[0]["images"]["hand"]["file"] = "hand/../../outside.jpg"
        sample_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        quality_path = self.root / "raw" / "episode-b" / "quality.json"
        quality = json.loads(quality_path.read_text())
        quality["write_errors"] = ["simulated"]
        quality_path.write_text(json.dumps(quality))
        plan = build_plan(self.root / "raw")
        reasons = {item["source_episode"]: item["reason"] for item in plan["rejected"]}
        self.assertNotIn("episode-a", reasons)
        self.assertEqual(plan["filter_counts"]["missing_hand"], 1)
        self.assertEqual(reasons["episode-b"], "write_errors")

    def test_export_refuses_existing_or_nested_output(self):
        plan = self.make_plan()
        with self.assertRaises(FileExistsError):
            export_plan(plan, self.root / "raw", "pick", min_free_gib=0,
                        dataset_cls=FakeDataset)
        with self.assertRaises(ValueError):
            export_plan(plan, self.root / "raw" / "nested", "pick", min_free_gib=0,
                        dataset_cls=FakeDataset)


if __name__ == "__main__":
    unittest.main()
