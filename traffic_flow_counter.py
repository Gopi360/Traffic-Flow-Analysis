#!/usr/bin/env python3
"""
traffic_flow_counter.py
Counts vehicles per lane with YOLO (COCO) detection + ByteTrack tracking.
- Downloads the YouTube video if a local copy isn't present (requires yt-dlp)
- Supports 3+ polygon lanes from lanes.json (created by lanes_annotator.py)
- Writes:
    - overlayed output video
    - CSV with rows: VehicleID, Lane, Frame, Timestamp
    - Summary at end of run

Usage:
python traffic_flow_counter.py --video traffic.mp4 --lanes lanes.json --out_video out.mp4 --out_csv counts.csv

Note: Uses Ultralytics YOLO and supervision ByteTrack. Install requirements from requirements.txt.
"""

import argparse, os, sys, time, csv, json, math, datetime
from collections import defaultdict, deque
from typing import List, Tuple, Dict

import cv2
import numpy as np

def download_video_ytdlp(video_url: str, out_path: str) -> str:
    try:
        import yt_dlp
    except Exception as e:
        print("[WARN] yt-dlp not installed; cannot download video automatically.")
        return ""
    ydl_opts = {
        'format': 'mp4/best',
        'outtmpl': out_path,
        'quiet': True,
        'noprogress': True,
    }
    print(f"[INFO] Downloading video via yt-dlp: {video_url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return out_path

def load_lanes(lanes_json_path: str) -> List[np.ndarray]:
    with open(lanes_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lanes = [np.array(poly, dtype=np.int32) for poly in data["lanes"]]
    return lanes

def format_ts(frame_idx: int, fps: float) -> str:
    secs = frame_idx / fps if fps > 0 else 0.0
    return str(datetime.timedelta(seconds=secs))

def center_of_bbox(xyxy):
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def point_in_polygon(point, polygon: np.ndarray) -> bool:
    # polygon: (N,2) int32, closed polygon assumed
    res = cv2.pointPolygonTest(polygon, point, False)
    return res >= 0  # inside or on edge

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="traffic.mp4")
    parser.add_argument("--video_url", type=str, default="https://www.youtube.com/watch?v=MNn9qKG2UFI")
    parser.add_argument("--lanes", type=str, default="lanes.json")
    parser.add_argument("--out_video", type=str, default="traffic_overlay.mp4")
    parser.add_argument("--out_csv", type=str, default="vehicle_counts.csv")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Ultralytics YOLO model weights (COCO). e.g., yolov8n.pt / yolov8s.pt")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Set to 'cuda:0' for GPU, else None for CPU")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--classes", type=str, default="car,truck,bus,motorbike",
                        help="Comma-separated class names to include")
    parser.add_argument("--dwell", type=int, default=3, help="Frames to remain in a lane before counting")
    args = parser.parse_args()

    # Lazy imports (so this file can be opened without env ready)
    try:
        from ultralytics import YOLO
        import supervision as sv
    except Exception as e:
        print("[ERROR] Missing dependencies. Install with: pip install -r requirements.txt")
        sys.exit(1)

    if not os.path.exists(args.video):
        print(f"[INFO] Local video '{args.video}' not found.")
        out = download_video_ytdlp(args.video_url, args.video)
        if not out or not os.path.exists(args.video):
            print("[ERROR] Could not obtain video. Provide --video or install yt-dlp.")
            sys.exit(1)

    if not os.path.exists(args.lanes):
        print("[ERROR] Lanes file not found. Run lanes_annotator.py to create lanes.json")
        sys.exit(1)

    # Load lanes
    lanes = load_lanes(args.lanes)
    if len(lanes) < 3:
        print("[WARN] You provided fewer than 3 lanes. The script will still run, but the assessment requires 3.")
    # Initialize YOLO
    print("[INFO] Loading YOLO model...")
    model = YOLO(args.model)
    if args.device is not None:
        model.to(args.device)
    names = model.names

    # Map desired classes by name to class IDs
    keep_names = [n.strip() for n in args.classes.split(",") if n.strip()]
    keep_ids = [cls_id for cls_id, name in names.items() if name in keep_names]

    # Tracker
    tracker = sv.ByteTrack()

    # Video IO
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("[ERROR] Failed to open video")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_video, fourcc, fps, (width, height))

    # Counting state
    lane_counts = [0 for _ in lanes]
    counted_pairs = set()     # (track_id, lane_idx) already counted
    inside_tally = defaultdict(lambda: defaultdict(int))  # inside_tally[track_id][lane_idx] -> frames inside
    last_seen_lane = {}       # track_id -> lane_idx or None

    # CSV
    csv_f = open(args.out_csv, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["VehicleID", "Lane", "Frame", "Timestamp"])

    frame_idx = 0
    vis_color_lane = (0, 255, 0)
    vis_color_box = (255, 255, 255)

    print("[INFO] Processing... Press 'q' to stop early.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Detection
            results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            # Convert to supervision Detections
            dets = sv.Detections.from_ultralytics(results)

            # Keep only selected classes
            if dets.class_id is not None and len(dets) > 0:
                mask = np.isin(dets.class_id, keep_ids)
                dets = dets[mask]

            # Track
            tracks = tracker.update_with_detections(dets)

            # Draw lanes
            for i, poly in enumerate(lanes):
                cv2.polylines(frame, [poly], True, vis_color_lane, 2)
                M = poly.mean(axis=0).astype(int).ravel()
                label = f"Lane {i+1}: {lane_counts[i]}"
                cv2.putText(frame, label, (M[0], M[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, vis_color_lane, 2, cv2.LINE_AA)

            # For each tracked object
            for i in range(len(tracks)):
                xyxy = tracks.xyxy[i]
                track_id = tracks.tracker_id[i]
                if track_id is None or track_id < 0:
                    continue

                x1,y1,x2,y2 = map(int, xyxy)
                cx, cy = center_of_bbox(xyxy)

                # Determine lane membership
                current_lane = None
                for lane_idx, poly in enumerate(lanes):
                    if point_in_polygon((cx, cy), poly):
                        current_lane = lane_idx
                        break

                # Update dwell/inside counts
                if current_lane is not None:
                    inside_tally[track_id][current_lane] += 1
                # Reset other lanes' tallies for this track to avoid accidental carry-over
                for lane_idx in list(inside_tally[track_id].keys()):
                    if lane_idx != current_lane:
                        inside_tally[track_id][lane_idx] = max(0, inside_tally[track_id][lane_idx] - 1)

                # Count when dwell threshold reached and not counted before in that lane
                if current_lane is not None and inside_tally[track_id][current_lane] >= args.dwell:
                    pair = (track_id, current_lane)
                    if pair not in counted_pairs:
                        counted_pairs.add(pair)
                        lane_counts[current_lane] += 1
                        csv_w.writerow([track_id, current_lane + 1, frame_idx, format_ts(frame_idx, fps)])

                # Draw track
                cv2.rectangle(frame, (x1,y1), (x2,y2), vis_color_box, 2)
                cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)
                label = f"ID {track_id}"
                cv2.putText(frame, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vis_color_box, 2, cv2.LINE_AA)

            # HUD
            total = sum(lane_counts)
            hud = f"Frame: {frame_idx} | Total vehicles: {total}"
            cv2.rectangle(frame, (0,0), (width, 40), (0,0,0), -1)
            cv2.putText(frame, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

            writer.write(frame)
            cv2.imshow("Traffic Flow Counter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        csv_f.close()
        cv2.destroyAllWindows()

    # Summary
    print("\n=== SUMMARY ===")
    for i, c in enumerate(lane_counts):
        print(f"Lane {i+1}: {c}")
    print(f"TOTAL: {sum(lane_counts)}")

    # Also save summary JSON next to CSV
    summary_path = os.path.splitext(args.out_csv)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"per_lane": {f"lane_{i+1}": c for i, c in enumerate(lane_counts)},
                   "total": int(sum(lane_counts))}, f, indent=2)
    print(f"[OK] Wrote summary JSON -> {summary_path}")

if __name__ == "__main__":
    main()
