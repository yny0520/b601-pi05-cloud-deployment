#!/usr/bin/env python3
"""Export reviewed B601 raw episodes to the LeRobot v3 dataset contract."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from importlib.metadata import version as package_version
import inspect
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image

CAMERAS = {"top": "camera1", "hand": "camera2"}
JOINTS = [f"J{i}" for i in range(1, 8)]
DEFAULT_TASK = "Pick up the screwdriver, place it inside the box, release it, and return the arm to its home pose."
DEFAULT_FPS = 14
MAX_CONTROL_GAP_S = 0.150
MAX_IMAGE_AGE_MS = 200.0
MIN_SEGMENT_FRAMES = DEFAULT_FPS


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _valid_seven(values):
    return (isinstance(values, list) and len(values) == 7
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in values))


def _image_path(episode: Path, name: str, value):
    if not isinstance(value, dict) or not value.get("file"):
        return None
    relative = Path(value["file"])
    if (relative.is_absolute() or not relative.parts or ".." in relative.parts
            or relative.parts[0] != name):
        return None
    path = (episode / relative).resolve()
    if not path.is_relative_to(episode.resolve()) or not path.is_file():
        return None
    return path


def _usable_row(row, episode: Path, max_image_age_ms=MAX_IMAGE_AGE_MS):
    if row.get("demonstration_segment") is not True:
        return False, "outside_demonstration"
    if row.get("paused") is not False:
        return False, "paused_or_unknown"
    if row.get("fault_hold") is True:
        return False, "fault_hold"
    if not _valid_seven(row.get("state_deg")):
        return False, "invalid_state"
    if not _valid_seven(row.get("action_position_reference_deg")):
        return False, "invalid_action"
    timing = row.get("read_window_ns")
    if (not isinstance(timing, list) or len(timing) != 2
            or not all(isinstance(x, int) for x in timing) or timing[0] > timing[1]):
        return False, "invalid_control_timestamp"
    images = row.get("images", {})
    for name in CAMERAS:
        image = images.get(name, {})
        age = image.get("age_ms")
        if image.get("fresh") is not True or not isinstance(age, (int, float)):
            return False, f"stale_{name}"
        if not math.isfinite(age) or age < 0 or age > max_image_age_ms:
            return False, f"stale_{name}"
        if _image_path(episode, name, image) is None:
            return False, f"missing_{name}"
    if not isinstance(row.get("teleop_session"), str) or not row["teleop_session"]:
        return False, "missing_teleop_session"
    if not isinstance(row.get("teleop_seq"), int):
        return False, "missing_teleop_sequence"
    return True, "kept"


def _segments(rows, episode, fps, max_image_age_ms):
    candidates, run = [], []
    rejected = Counter()
    for index, row in enumerate(rows):
        usable, reason = _usable_row(row, episode, max_image_age_ms)
        if not usable:
            rejected[reason] += 1
            if run:
                candidates.append(run)
                run = []
            continue
        if run:
            previous = rows[run[-1]]
            time_gap = (row["read_window_ns"][1] - previous["read_window_ns"][1]) / 1e9
            seq_gap = row["teleop_session"] != previous["teleop_session"] or row["teleop_seq"] != previous["teleop_seq"] + 1
            if time_gap <= 0 or time_gap > MAX_CONTROL_GAP_S or seq_gap:
                candidates.append(run)
                run = []
        run.append(index)
    if run:
        candidates.append(run)

    selected = []
    segment_no = 0
    for indices in candidates:
        if len(indices) < max(fps, MIN_SEGMENT_FRAMES):
            rejected["segment_too_short"] += len(indices)
            continue
        stamps = np.asarray([rows[i]["read_window_ns"][1] for i in indices], dtype=np.int64)
        times = (stamps - stamps[0]).astype(np.float64) / 1e9
        if np.any(np.diff(times) <= 0):
            rejected["non_monotonic_control_timestamp"] += len(indices)
            continue
        grid = np.arange(int(np.floor(times[-1] * fps)) + 1, dtype=np.float64) / fps
        source_positions = np.searchsorted(times, grid, side="right") - 1
        if np.any(source_positions < 0):
            rejected["no_causal_observation"] += len(indices)
            continue
        lag = grid - times[source_positions]
        windows, window = [], []
        for grid_index, value in enumerate(lag):
            if value > 1.5 / fps:
                rejected["resample_lag_too_large"] += 1
                if window:
                    windows.append(window)
                    window = []
                continue
            window.append(grid_index)
        if window:
            windows.append(window)
        for window in windows:
            if len(window) < max(fps, MIN_SEGMENT_FRAMES):
                rejected["segment_too_short"] += len(window)
                continue
            session = rows[indices[0]]["teleop_session"]
            selected.append({
                "segment_no": segment_no,
                "session": session,
                "source_indices": [indices[int(source_positions[j])] for j in window],
                "target_times": (grid[window] - grid[window[0]]).tolist(),
                "resample_lags": lag[window].tolist(),
            })
            segment_no += 1
    return selected, rejected


def _episode_rejection(episode, metadata, quality):
    if metadata.get("schema") != "b601-raw-episode-v1":
        return "unsupported_raw_schema"
    if metadata.get("status") != "saved":
        return "episode_not_saved_cleanly"
    if metadata.get("outcome") != "success":
        return f"outcome_{metadata.get('outcome', 'missing')}"
    if metadata.get("joint_order") != JOINTS:
        return "joint_order_mismatch"
    if not isinstance(quality, dict):
        return "missing_quality_report"
    if quality.get("write_errors"):
        return "write_errors"
    samples = episode / "samples.jsonl"
    if not samples.is_file():
        return "missing_samples"
    return None


def build_plan(raw_root: Path, fps=DEFAULT_FPS, max_image_age_ms=MAX_IMAGE_AGE_MS,
               limit_episodes=0):
    raw_root = Path(raw_root).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw dataset root does not exist: {raw_root}")
    if fps < 1 or fps > 30:
        raise ValueError("LeRobot FPS must be in 1..30")

    plan = {"raw_root": raw_root, "fps": fps, "episodes": [], "rejected": [],
            "filter_counts": Counter(), "source_episode_count": 0}
    folders = sorted(p for p in raw_root.iterdir() if p.is_dir() and (p / "metadata.json").is_file())
    if limit_episodes:
        folders = folders[:limit_episodes]
    for episode in folders:
        plan["source_episode_count"] += 1
        try:
            metadata = read_json(episode / "metadata.json")
            quality = read_json(episode / "quality.json") if (episode / "quality.json").is_file() else None
            rejection = _episode_rejection(episode, metadata, quality)
            if rejection:
                plan["rejected"].append({"source_episode": episode.name, "reason": rejection})
                plan["filter_counts"][rejection] += 1
                continue
            rows = read_jsonl(episode / "samples.jsonl")
            counts = metadata.get("counts", {})
            if (counts.get("teleop") != len(rows)
                    or quality.get("telemetry_records") != len(rows)):
                plan["rejected"].append({"source_episode": episode.name, "reason": "telemetry_count_mismatch"})
                plan["filter_counts"]["telemetry_count_mismatch"] += 1
                continue
            segments, rejected = _segments(rows, episode, fps, max_image_age_ms)
            plan["filter_counts"].update(rejected)
            if not segments:
                plan["rejected"].append({"source_episode": episode.name, "reason": "no_trainable_segments"})
                plan["filter_counts"]["no_trainable_segments"] += 1
                continue
            plan["episodes"].append({"folder": episode, "metadata": metadata, "rows": rows,
                                     "segments": segments})
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            reason = f"invalid_episode:{type(exc).__name__}:{exc}"
            plan["rejected"].append({"source_episode": episode.name, "reason": reason})
            plan["filter_counts"]["invalid_episode"] += 1
    return plan


def assign_session_splits(episodes, seed=2026):
    grouped = defaultdict(list)
    for episode in episodes:
        for segment in episode["segments"]:
            grouped[segment["session"]].append((episode, segment))
    sessions = sorted(grouped)
    if len(sessions) < 3:
        raise ValueError("At least three independent teleoperation sessions are required for leakage-free splits")
    best = None
    rng = random.Random(seed)
    total = sum(len(items) for items in grouped.values())
    for _ in range(2000):
        shuffled = sessions[:]
        rng.shuffle(shuffled)
        n = max(1, round(len(shuffled) * 0.15))
        assignment = {name: "validation" for name in shuffled[:n]}
        assignment.update({name: "test" for name in shuffled[n:2 * n]})
        assignment.update({name: "train" for name in shuffled[2 * n:]})
        counts = Counter()
        for session, items in grouped.items():
            counts[assignment[session]] += len(items)
        cost = sum(abs(counts[part] / total - target)
                   for part, target in (("train", .70), ("validation", .15), ("test", .15)))
        if best is None or cost < best[0]:
            best = (cost, assignment)
    return best[1]


def summarize_plan(plan):
    segments = [segment for episode in plan["episodes"] for segment in episode["segments"]]
    return {
        "source_root": str(plan["raw_root"]),
        "source_episode_count": plan["source_episode_count"],
        "eligible_source_episode_count": len(plan["episodes"]),
        "rejected_source_episodes": plan["rejected"],
        "segments": len(segments),
        "planned_frames": sum(len(segment["source_indices"]) for segment in segments),
        "teleoperation_sessions": len({segment["session"] for segment in segments}),
        "fps": plan["fps"],
        "filter_counts": dict(plan["filter_counts"]),
        "camera_map": CAMERAS,
        "action_semantics": "7D absolute joint position references in radians; J7 is adapted to MIT torque+damping by the existing controller",
        "state_semantics": "7D measured follower joint positions in radians",
        "resampling": "causal previous-observation sampling; invalid rows, session changes, sequence gaps, control gaps over 150ms, and excessive resampling lag split segments",
        "output_format": "LeRobot dataset v3.0",
    }


def _image_shape(plan):
    for episode in plan["episodes"]:
        for segment in episode["segments"]:
            row = episode["rows"][segment["source_indices"][0]]
            shapes = {}
            for camera, alias in CAMERAS.items():
                path = _image_path(episode["folder"], camera, row["images"][camera])
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    shapes[alias] = (rgb.height, rgb.width, 3)
            return shapes
    raise ValueError("No valid image frame found")


def _le_robot_features(shapes):
    features = {
        key: {"dtype": "float32", "shape": (7,), "names": JOINTS}
        for key in ("observation.state", "action")
    }
    for alias, shape in shapes.items():
        features[f"observation.images.{alias}"] = {
            "dtype": "video", "shape": shape, "names": ["height", "width", "channels"]
        }
    return features


def _make_writer(output_root, fps, features, dataset_cls=None):
    if dataset_cls is None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset as dataset_cls
    options = {
        "repo_id": "local/b601_demonstrations",
        "fps": fps,
        "root": output_root,
        "robot_type": "b601",
        "features": features,
        "use_videos": True,
        "video_backend": "pyav",
        "vcodec": "h264",
    }
    parameters = inspect.signature(dataset_cls.create).parameters
    if "image_writer_threads" in parameters:
        options["image_writer_threads"] = 8
    elif "encoder_threads" in parameters:
        options["encoder_threads"] = 4
    return dataset_cls.create(**options)


def export_plan(plan, output_root: Path, task: str, seed=2026, min_free_gib=2,
                dataset_cls=None):
    output_root = Path(output_root).expanduser().resolve()
    raw_root = plan["raw_root"]
    if output_root.exists():
        raise FileExistsError(f"Dataset destination must not exist: {output_root}")
    if output_root == raw_root or output_root.is_relative_to(raw_root) or raw_root.is_relative_to(output_root):
        raise ValueError("Output path must be separate from the raw dataset root")
    if not plan["episodes"]:
        raise ValueError("No reviewed, usable success episodes are available")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("A consistent task instruction is required")
    session_split = assign_session_splits(plan["episodes"], seed)
    counts = Counter(session_split.values())
    if any(counts[part] == 0 for part in ("train", "validation", "test")):
        raise ValueError("Session split produced an empty partition")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_root.parent).free
    if free < min_free_gib * 1024**3:
        raise OSError(f"Insufficient free space: need at least {min_free_gib} GiB")

    shapes = _image_shape(plan)
    features = _le_robot_features(shapes)
    dataset = _make_writer(output_root, plan["fps"], features, dataset_cls=dataset_cls)
    split_indices = {f"{part}_episode_indices": [] for part in ("train", "validation", "test")}
    frame_map = []
    stats_values = {"observation.state": [], "action": []}
    output_episode_index = 0
    try:
        for episode in plan["episodes"]:
            for segment in episode["segments"]:
                part = session_split[segment["session"]]
                split_indices[f"{part}_episode_indices"].append(output_episode_index)
                for frame_index, (source_index, target_time, lag) in enumerate(zip(
                        segment["source_indices"], segment["target_times"], segment["resample_lags"])):
                    row = episode["rows"][source_index]
                    state = np.deg2rad(np.asarray(row["state_deg"], dtype=np.float32))
                    action = np.deg2rad(np.asarray(row["action_position_reference_deg"], dtype=np.float32))
                    if state.shape != (7,) or action.shape != (7,) or not np.isfinite(state).all() or not np.isfinite(action).all():
                        raise ValueError(f"Invalid state/action in {episode['folder'].name}:{source_index}")
                    images = {}
                    for camera, alias in CAMERAS.items():
                        path = _image_path(episode["folder"], camera, row["images"][camera])
                        with Image.open(path) as image:
                            rgb = image.convert("RGB")
                            if (rgb.height, rgb.width, 3) != shapes[alias]:
                                raise ValueError(f"Image dimensions changed: {episode['folder'].name}/{camera}")
                            images[f"observation.images.{alias}"] = np.asarray(rgb, dtype=np.uint8).copy()
                    dataset.add_frame({"observation.state": state, "action": action,
                                       "task": task.strip(), **images})
                    frame_map.append({
                        "episode_index": output_episode_index,
                        "frame_index": frame_index,
                        "source_episode": episode["folder"].name,
                        "session": segment["session"],
                        "source_frame": source_index,
                        "source_mono_ns": row["read_window_ns"][1],
                        "segment_no": segment["segment_no"],
                        "target_time_s": target_time,
                        "resample_lag_s": lag,
                        "split": part,
                    })
                    if part == "train":
                        stats_values["observation.state"].append(state)
                        stats_values["action"].append(action)
                dataset.save_episode(parallel_encoding=False)
                output_episode_index += 1
    finally:
        dataset.finalize()
        stop_writer = getattr(dataset, "stop_image_writer", None)
        if stop_writer:
            stop_writer()

    if not frame_map or any(not split_indices[f"{part}_episode_indices"] for part in ("train", "validation", "test")):
        raise ValueError("Export did not produce all train/validation/test partitions")
    stats = {}
    for key, values in stats_values.items():
        array = np.stack(values).astype(np.float64)
        stats[key] = {"mean": array.mean(0).tolist(),
                      "std": np.maximum(array.std(0), 1e-6).tolist(),
                      "min": array.min(0).tolist(), "max": array.max(0).tolist(),
                      "count": [len(array)]}
    split_manifest = {
        **split_indices, "seed": seed,
        "split_unit": "teleoperation session before segmentation",
        "session_assignments": session_split,
    }
    lerobot_version = getattr(dataset_cls, "lerobot_version", None) or package_version("lerobot")
    report = summarize_plan(plan)
    report.update({
        "task": task.strip(), "robot_type": "b601", "lerobot_fps": plan["fps"],
        "lerobot_version": lerobot_version,
        "action_units": "radians", "state_units": "radians",
        "normalization": "train_only_stats.json contains training partition statistics",
        "converted_episode_count": output_episode_index,
        "converted_frame_count": len(frame_map),
        "train_episode_count": len(split_indices["train_episode_indices"]),
        "validation_episode_count": len(split_indices["validation_episode_indices"]),
        "test_episode_count": len(split_indices["test_episode_indices"]),
        "filter_counts": dict(plan["filter_counts"]),
    })
    write_json(output_root / "b601_export_report.json", report)
    write_json(output_root / "episode_split.json", split_manifest)
    write_json(output_root / "train_only_stats.json", stats)
    with (output_root / "source_frame_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_map[0]))
        writer.writeheader()
        writer.writerows(frame_map)
    write_json(output_root / "rejected_episodes.json", plan["rejected"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--max-image-age-ms", type=float, default=MAX_IMAGE_AGE_MS)
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-free-gib", type=int, default=2)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    if args.limit_episodes < 0:
        parser.error("--limit-episodes must be non-negative")
    plan = build_plan(args.raw_root, args.fps, args.max_image_age_ms, args.limit_episodes)
    if args.audit_only:
        print(json.dumps(summarize_plan(plan), ensure_ascii=False, indent=2))
        return 0
    if args.output_root is None:
        parser.error("--output-root is required unless --audit-only is used")
    report = export_plan(plan, args.output_root, args.task, args.seed, args.min_free_gib)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
