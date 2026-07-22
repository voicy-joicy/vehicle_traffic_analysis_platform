"""Utility functions for the Vehicle Traffic Analysis Platform."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Set, Tuple, List, Optional

import cv2
import numpy as np


# COCO class IDs used by pretrained YOLOv8 models.
# 1 = bicycle, 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
DEFAULT_VEHICLE_CLASS_IDS = [2, 3, 5, 7]

VEHICLE_CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# A track must appear in at least this number of processed frames before it is counted.
# This reduces false IDs from one-frame mistakes.
MIN_TRACK_FRAMES_FOR_COUNT = 3

ProgressCallback = Optional[Callable[[float, str, Optional[np.ndarray]], None]]


def ensure_project_folders(base_dir: Path) -> None:
    """Create required project folders if they do not already exist."""
    folders = [
        base_dir / "uploads",
        base_dir / "outputs",
        base_dir / "models",
        base_dir / "utils",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def _open_video(input_path: Path) -> cv2.VideoCapture:
    """Open a video file and raise a clear error if it fails."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    return cap


def _create_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create an MP4 video writer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    return writer


def _get_congestion_level(avg_vehicles_per_frame: float) -> str:
    """Return a simple congestion level based on the average vehicles per frame."""
    if avg_vehicles_per_frame < 3:
        return "Low"
    if avg_vehicles_per_frame < 8:
        return "Moderate"
    return "High"


def _draw_text_box(frame: np.ndarray, text: str, x: int, y: int) -> None:
    """Draw readable label text above a bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size

    y = max(y, text_h + 8)
    cv2.rectangle(frame, (x, y - text_h - 8), (x + text_w + 8, y), (15, 23, 42), -1)
    cv2.putText(frame, text, (x + 4, y - 5), font, scale, (255, 255, 255), thickness)


def _draw_tracking_line(frame: np.ndarray, line_y: int) -> None:
    """Draw the counting line used for up/down crossing estimation."""
    height, width = frame.shape[:2]
    cv2.line(frame, (0, line_y), (width, line_y), (255, 255, 0), 2)
    cv2.putText(
        frame,
        "Counting Line",
        (20, max(30, line_y - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
    )


def _update_direction_counts(
    track_id: int,
    center_y: int,
    line_y: int,
    previous_centers: Dict[int, int],
    counted_crossings: Set[Tuple[int, str]],
    direction_counts: Dict[str, int],
) -> None:
    """Update up/down counts when a tracked vehicle crosses the middle line."""
    previous_y = previous_centers.get(track_id)

    if previous_y is not None:
        if previous_y < line_y <= center_y and (track_id, "down") not in counted_crossings:
            direction_counts["down"] += 1
            counted_crossings.add((track_id, "down"))
        elif previous_y > line_y >= center_y and (track_id, "up") not in counted_crossings:
            direction_counts["up"] += 1
            counted_crossings.add((track_id, "up"))

    previous_centers[track_id] = center_y

def _majority_class(class_votes: Dict[str, int], default_class: str = "Vehicle") -> str:
    """Return the most frequent class name for a tracked object."""
    if not class_votes:
        return default_class
    return max(class_votes.items(), key=lambda item: item[1])[0]

def process_video(
    input_path: Path,
    output_path: Path,
    model: Any,
    vehicle_class_ids: Iterable[int] = DEFAULT_VEHICLE_CLASS_IDS,
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    frame_skip: int = 1,
    use_tracker: bool = True,
    tracker_config: str = "bytetrack.yaml",
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    """
     Process a video using YOLOv8 detection and optional ByteTrack tracking.

    Important tracking improvements:
    - Tracking automatically uses every frame, because skipping frames makes ID switching worse.
    - A vehicle is counted only after it appears in several frames.
    - Vehicle type is decided using majority voting per track ID, so one mistaken frame
      does not count a car as a motorcycle or truck.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    vehicle_class_ids = [int(class_id) for class_id in vehicle_class_ids]

    # ByteTrack should receive consecutive frames. If frames are skipped, the tracker may lose vehicles.
    frame_skip = 1 if use_tracker else max(1, int(frame_skip))
    
    cap = _open_video(input_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if fps <= 0:
        fps = 25.0

    writer = _create_video_writer(output_path, fps, width, height)

    line_y = height // 2
    frames_processed = 0
    total_vehicle_detections = 0
    tracked_vehicle_ids: Set[int] = set()
    previous_centers: Dict[int, int] = {}
    counted_crossings: Set[Tuple[int, str]] = set()
    direction_counts = {"down": 0, "up": 0}

    # Tracking memory
    tracked_vehicle_ids: Set[int] = set()
    track_frame_counts: Dict[int, int] = defaultdict(int)
    track_class_votes: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Used only when tracking is disabled.
    fallback_detection_id = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frames_processed += 1

            annotated_frame = frame.copy()
            _draw_tracking_line(annotated_frame, line_y)

            should_process = (frames_processed - 1) % frame_skip == 0

            if should_process:
                if use_tracker:
                    results = model.track(
                        frame,
                        persist=True,
                        tracker=tracker_config,
                        classes=vehicle_class_ids,
                        conf=confidence_threshold,
                        iou=iou_threshold,
                        verbose=False,
                    )
                else:
                    results = model.predict(
                        frame,
                        classes=vehicle_class_ids,
                        conf=confidence_threshold,
                        iou=iou_threshold,
                        verbose=False,
                    )

                if results and results[0].boxes is not None:
                    boxes = results[0].boxes

                    xyxy = boxes.xyxy.cpu().numpy().astype(int) if boxes.xyxy is not None else []
                    class_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
                    confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []

                    if use_tracker and boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)
                    else:
                        track_ids = []
                        for _ in range(len(xyxy)):
                            fallback_detection_id += 1
                            track_ids.append(fallback_detection_id)

                    for box, class_id, confidence, track_id in zip(xyxy, class_ids, confidences, track_ids):
                        class_id = int(class_id)

                        # Extra safety: ignore anything outside the selected vehicle classes.
                        if class_id not in vehicle_class_ids:
                            continue

                        class_name = VEHICLE_CLASS_NAMES.get(class_id)
                        if class_name is None:
                            continue

                        x1, y1, x2, y2 = box
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        track_id = int(track_id)

                        total_vehicle_detections += 1
                        tracked_vehicle_ids.add(track_id)
                        track_frame_counts[track_id] += 1
                        track_class_votes[track_id][class_name] += 1

                        stable_class_name = _majority_class(track_class_votes[track_id], class_name)

                        if use_tracker:
                            _update_direction_counts(
                                track_id,
                                center_y,
                                line_y,
                                previous_centers,
                                counted_crossings,
                                direction_counts,
                            )

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (34, 211, 238), 2)
                        cv2.circle(annotated_frame, (center_x, center_y), 4, (251, 146, 60), -1)

                        if use_tracker:
                            label = f"{stable_class_name} ID:{track_id} {float(confidence):.2f}"
                        else:
                            label = f"{stable_class_name} {float(confidence):.2f}"

                        _draw_text_box(annotated_frame, label, x1, y1 - 8)

            writer.write(annotated_frame)

            if progress_callback is not None:
                progress = frames_processed / total_frames if total_frames > 0 else 0.0
                mode = "tracking" if use_tracker else "detection"
                message = f"Processing frame {frames_processed} of {total_frames} using {mode}..."

                preview_frame = annotated_frame if frames_processed % max(1, frame_skip) == 0 else None
                progress_callback(progress, message, preview_frame)

    finally:
        cap.release()
        writer.release()

    # Count only stable tracks that appeared in enough frames.
    if use_tracker:
        stable_track_ids = {
            track_id
            for track_id, count in track_frame_counts.items()
            if count >= MIN_TRACK_FRAMES_FOR_COUNT
        }
    else:
        stable_track_ids = tracked_vehicle_ids

    class_counts: Dict[str, int] = {}
    for track_id in stable_track_ids:
        class_name = _majority_class(track_class_votes.get(track_id, {}), "Vehicle")
        if class_name != "Vehicle":
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    unique_vehicle_detections = len(stable_track_ids)
    avg_vehicles_per_frame = total_vehicle_detections / frames_processed if frames_processed else 0.0

    return {
        "frames_processed": frames_processed,
        "total_vehicle_detections": total_vehicle_detections,
        "unique_vehicle_detections": unique_vehicle_detections,
        "tracked_vehicles": unique_vehicle_detections if use_tracker else 0,
        "direction_counts": direction_counts,
        "avg_vehicles_per_frame": avg_vehicles_per_frame,
        "congestion_level": _get_congestion_level(avg_vehicles_per_frame),
        "class_counts": class_counts,
        "tracker_used": tracker_config if use_tracker else "None",
    }
