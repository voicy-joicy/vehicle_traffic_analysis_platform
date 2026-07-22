"""Utility functions for the Vehicle Traffic Analysis Platform."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2


# COCO class IDs used by pretrained YOLOv8 models.
# 1 = bicycle, 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
DEFAULT_VEHICLE_CLASS_IDS: Dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    class_id: int
    class_name: str
    score: float
    last_seen: int
    time_since_update: int = 0
    history: list[tuple[int, int, int]] = field(default_factory=list)
    line_crossed: bool = False
    crossing_direction: Optional[str] = None


class ByteTrack:
    """Simplified ByteTrack-style tracker for vehicle tracking."""

    def __init__(self, max_time_lost: int = 30, iou_threshold: float = 0.3) -> None:
        self.max_time_lost = max_time_lost
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.tracks: List[TrackState] = []

    def update(self, detections: List[Dict[str, Any]], frame_idx: int) -> List[TrackState]:
        """Update tracks using the latest detections and return active tracks."""
        for track in self.tracks:
            track.time_since_update += 1

        candidate_tracks = [track for track in self.tracks if track.time_since_update <= self.max_time_lost]

        for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
            best_track: Optional[TrackState] = None
            best_iou = 0.0

            for track in candidate_tracks:
                if track.class_id != detection["class_id"]:
                    continue
                iou = _box_iou(track.bbox, detection["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track is not None and best_iou >= self.iou_threshold:
                best_track.bbox = detection["bbox"]
                best_track.score = detection["score"]
                best_track.last_seen = frame_idx
                best_track.time_since_update = 0
                cx, cy = detection["center"]
                best_track.history.append((frame_idx, cx, cy))
                detection["track_id"] = best_track.track_id
                candidate_tracks.remove(best_track)
            else:
                self.next_id += 1
                cx, cy = detection["center"]
                new_track = TrackState(
                    track_id=self.next_id,
                    bbox=detection["bbox"],
                    class_id=detection["class_id"],
                    class_name=detection["class_name"],
                    score=detection["score"],
                    last_seen=frame_idx,
                )
                new_track.history.append((frame_idx, cx, cy))
                detection["track_id"] = new_track.track_id
                self.tracks.append(new_track)

        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_time_lost]
        return self.tracks


def ensure_project_folders(base_dir: Path) -> None:
    """Create required project folders if they do not already exist."""
    for folder in ["uploads", "outputs", "models", "sample_videos", "utils"]:
        (base_dir / folder).mkdir(parents=True, exist_ok=True)


def get_video_properties(cap: cv2.VideoCapture) -> Dict[str, int | float]:
    """Read useful video properties from an OpenCV VideoCapture object."""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fallback values for videos whose metadata cannot be read correctly.
    if width <= 0:
        width = 640
    if height <= 0:
        height = 480
    if fps <= 0:
        fps = 25.0
    if total_frames < 0:
        total_frames = 0

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
    }


def _box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Compute IoU between two bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def draw_detection(
    frame,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
    confidence: float,
    vehicle_id: Optional[int] = None,
) -> None:
    """Draw a bounding box and label on a video frame."""
    colors = {
        "car": (0, 255, 0),
        "bus": (0, 0, 255),
        "truck": (0, 165, 255),
        "motorcycle": (191, 0, 255),
    }
    box_color = colors.get(label.lower(), (0, 255, 0))
    text_color = (255, 255, 255)
    label_bg_color = tuple(min(c + 30, 255) for c in box_color)

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    label_text = label
    if vehicle_id is not None:
        label_text = f"{label}#{vehicle_id}"
    text = f"{label_text} {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    y_text = max(y1 - 10, text_height + 10)

    cv2.rectangle(
        frame,
        (x1, y_text - text_height - baseline),
        (x1 + text_width + 6, y_text + baseline),
        label_bg_color,
        -1,
    )
    cv2.putText(frame, text, (x1 + 3, y_text), font, font_scale, text_color, thickness)


def draw_summary_overlay(frame, frame_number: int, total_detections: int) -> None:
    """Draw a small summary overlay on the frame."""
    text = f"Frame: {frame_number} | Vehicle detections: {total_detections}"
    cv2.rectangle(frame, (10, 10), (520, 48), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )


def process_video(
    input_path: Path,
    output_path: Path,
    model,
    vehicle_class_ids: Dict[int, str],
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    frame_skip: int = 1,
    use_tracker: bool = False,
    tracking_line_position: float = 0.5,
    progress_callback: Optional[Callable[[float, str, object], None]] = None,
) -> Dict[str, object]:
    """
    Process a traffic video and save an annotated output video.

    Parameters
    ----------
    input_path:
        Path to the uploaded input video.
    output_path:
        Path where the processed video will be saved.
    model:
        Loaded Ultralytics YOLO model.
    vehicle_class_ids:
        Dictionary of YOLO class IDs to keep.
    confidence_threshold:
        Minimum confidence score for detections.
    iou_threshold:
        IoU threshold for non-maximum suppression.
    frame_skip:
        Process every Nth frame. Use 1 for best result.
    use_tracker:
        Whether to use ByteTrack-style tracking for stable IDs.
    tracking_line_position:
        Position of the crossing line as a fraction of frame height.
    progress_callback:
        Optional callback used by Streamlit to display progress.
    """
    if frame_skip < 1:
        frame_skip = 1

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {input_path}")

    props = get_video_properties(cap)
    width = int(props["width"])
    height = int(props["height"])
    fps = float(props["fps"])
    total_frames = int(props["total_frames"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    tracker = ByteTrack(max_time_lost=max(30, frame_skip * 5), iou_threshold=0.3) if use_tracker else None
    line_y = int(height * tracking_line_position)

    frames_read = 0
    frames_processed = 0
    total_vehicle_detections = 0
    unique_vehicle_detections = 0
    class_counts: Dict[str, int] = {}
    active_tracks: List[Dict[str, object]] = []
    vehicle_tracks: Dict[int, Dict[str, object]] = {}
    cars_per_frame: List[int] = []
    tracked_ids: set[int] = set()
    direction_counts = {"up": 0, "down": 0}
    line_crossing_counts = {"up": 0, "down": 0}
    track_id_seq = 0
    track_max_age = frame_skip + 5
    start_time = time.time()
    last_annotated_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_read += 1
        should_process = (frames_read - 1) % frame_skip == 0

        if should_process:
            frames_processed += 1

            results = model.predict(
                source=frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False,
            )

            detections: List[Dict[str, Any]] = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    if class_id not in vehicle_class_ids:
                        continue

                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_name = vehicle_class_ids[class_id]
                    bbox = (x1, y1, x2, y2)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append(
                        {
                            "bbox": bbox,
                            "score": confidence,
                            "class_id": class_id,
                            "class_name": class_name,
                            "center": (cx, cy),
                        }
                    )

            total_vehicle_detections += len(detections)

            if tracker is not None:
                tracker.update(detections, frames_read)
                current_tracks = [track for track in tracker.tracks if track.last_seen == frames_read]
                frame_vehicle_count = len(current_tracks)

                for track in current_tracks:
                    x1, y1, x2, y2 = track.bbox
                    draw_detection(frame, x1, y1, x2, y2, track.class_name, track.score, vehicle_id=track.track_id)

                    if track.track_id not in tracked_ids:
                        tracked_ids.add(track.track_id)
                        unique_vehicle_detections += 1
                        class_counts[track.class_name] = class_counts.get(track.class_name, 0) + 1

                    if not track.line_crossed and len(track.history) >= 2:
                        _, _, previous_y = track.history[-2]
                        _, _, current_y = track.history[-1]
                        if previous_y < line_y <= current_y:
                            track.line_crossed = True
                            track.crossing_direction = "down"
                            direction_counts["down"] += 1
                            line_crossing_counts["down"] += 1
                        elif previous_y > line_y >= current_y:
                            track.line_crossed = True
                            track.crossing_direction = "up"
                            direction_counts["up"] += 1
                            line_crossing_counts["up"] += 1
            else:
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes

                    frame_vehicle_count = 0
                    for box in boxes:
                        class_id = int(box.cls[0].item())
                        if class_id not in vehicle_class_ids:
                            continue

                        confidence = float(box.conf[0].item())
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        class_name = vehicle_class_ids[class_id]
                        bbox = (x1, y1, x2, y2)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        best_track = None
                        best_iou = 0.0
                        for track in active_tracks:
                            if track["class_name"] != class_name:
                                continue
                            if frames_read - track["last_seen"] > track_max_age:
                                continue

                            current_iou = _box_iou(track["bbox"], bbox)
                            if current_iou > best_iou:
                                best_iou = current_iou
                                best_track = track

                        if best_track is not None and best_iou >= 0.30:
                            matched_id = best_track["id"]
                            best_track["bbox"] = bbox
                            best_track["last_seen"] = frames_read
                            best_track["history"].append((frames_read, cx, cy))
                        else:
                            track_id_seq += 1
                            matched_id = track_id_seq
                            new_track = {
                                "id": matched_id,
                                "bbox": bbox,
                                "class_name": class_name,
                                "last_seen": frames_read,
                                "history": [(frames_read, cx, cy)],
                            }
                            active_tracks.append(new_track)
                            vehicle_tracks[matched_id] = new_track
                            unique_vehicle_detections += 1
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1

                        draw_detection(frame, x1, y1, x2, y2, class_name, confidence, vehicle_id=matched_id)
                        frame_vehicle_count += 1

                    active_tracks = [
                        track
                        for track in active_tracks
                        if frames_read - track["last_seen"] <= track_max_age
                    ]
                else:
                    frame_vehicle_count = 0

            cars_per_frame.append(frame_vehicle_count)
            draw_summary_overlay(frame, frames_read, total_vehicle_detections)
            last_annotated_frame = frame.copy()
        else:
            draw_summary_overlay(frame, frames_read, total_vehicle_detections)

        writer.write(frame)

        if progress_callback and (frames_read % 10 == 0 or frames_read == 1):
            progress = frames_read / total_frames if total_frames > 0 else 0.0
            progress_callback(
                progress,
                f"Processed {frames_read} of {total_frames if total_frames else 'unknown'} frames...",
                last_annotated_frame if last_annotated_frame is not None else frame,
            )

    cap.release()
    writer.release()

    elapsed = max(time.time() - start_time, 0.001)
    processing_fps = frames_read / elapsed

    avg_car_speed = 0.0
    if use_tracker:
        car_tracks = [
            track for track in tracker.tracks if track.class_name == "car" and len(track.history) > 1
        ]
    else:
        car_tracks = [
            track for track in vehicle_tracks.values() if track["class_name"] == "car" and len(track["history"]) > 1
        ]

    if car_tracks:
        speed_values = []
        for track in car_tracks:
            total_dist = 0.0
            total_time = 0.0
            history = track.history if use_tracker else track["history"]
            for (frame_idx, x, y), (next_frame, nx, ny) in zip(history, history[1:]):
                dist = math.hypot(nx - x, ny - y)
                time_delta = (next_frame - frame_idx) / fps
                if time_delta > 0:
                    total_dist += dist
                    total_time += time_delta
            if total_time > 0:
                speed_values.append(total_dist / total_time)
        if speed_values:
            avg_car_speed = sum(speed_values) / len(speed_values)

    avg_vehicles_per_frame = sum(cars_per_frame) / len(cars_per_frame) if cars_per_frame else 0.0
    if avg_vehicles_per_frame > 8:
        congestion_level = "High"
    elif avg_vehicles_per_frame > 4:
        congestion_level = "Medium"
    else:
        congestion_level = "Low"

    if progress_callback:
        progress_callback(1.0, "Video processing completed.", last_annotated_frame)

    return {
        "frames_read": frames_read,
        "frames_processed": frames_processed,
        "total_vehicle_detections": total_vehicle_detections,
        "unique_vehicle_detections": unique_vehicle_detections,
        "tracked_vehicles": len(tracked_ids) if use_tracker else unique_vehicle_detections,
        "avg_car_speed": avg_car_speed,
        "congestion_level": congestion_level,
        "avg_vehicles_per_frame": avg_vehicles_per_frame,
        "class_counts": class_counts,
        "direction_counts": direction_counts,
        "line_crossing_counts": line_crossing_counts,
        "processing_fps": processing_fps,
        "output_video": str(output_path),
        "input_video": str(input_path),
        "frame_skip": frame_skip,
    }
