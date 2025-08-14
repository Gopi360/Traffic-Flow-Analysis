# Traffic Flow Analysis — Implementation Guide

This guide walks you through a complete, **assessment-ready** setup that satisfies the rubric:
- Pre-trained **COCO** detector (Ultralytics YOLO)
- Tracking across frames (ByteTrack)
- **Three lanes** drawn as polygons with per-lane counts
- CSV export with **VehicleID, Lane, Frame, Timestamp**
- Video overlay with lane boundaries & live counts
- End-of-run **summary** per lane

> Video source in brief: `https://www.youtube.com/watch?v=MNn9qKG2UFI` (downloaded automatically if missing)

---

## 1) Environment

```bash
# (Recommended) create a fresh venv
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install deps
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch (choose CUDA/CPU build that fits your system)
# See: https://pytorch.org/get-started/locally/
# Example CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 2) Annotate the three lanes

Create polygons once and reuse them.

```bash
python lanes_annotator.py --video traffic.mp4 --lanes_out lanes.json
```
Controls:
- Left-click: add points for the current lane
- **n**: finish the lane and start a new one
- **u**: undo last point
- **r**: reset current unfinished lane
- **s**: save & exit

Tips:
- Draw each lane as a tight polygon around the road area for that lane.
- You can save more than 3 lanes if needed; the counter will display all.

---

## 3) Run the counter

```bash
python traffic_flow_counter.py   --video traffic.mp4   --lanes lanes.json   --out_video traffic_overlay.mp4   --out_csv vehicle_counts.csv   --model yolov8n.pt   --conf 0.35   --imgsz 640
```

What you get:
- `traffic_overlay.mp4` — overlayed video with polygons, boxes, IDs, and live counts
- `vehicle_counts.csv` — one row **per unique vehicle per lane**
- `vehicle_counts_summary.json` — per-lane and total summary

CSV columns:
- **VehicleID**: Tracker ID (stable across frames)
- **Lane**: 1-based lane index
- **Frame**: frame index when the count was registered
- **Timestamp**: derived from FPS

---

## 4) Performance notes

- Use `--model yolov8s.pt` or `yolov8n.pt` for speed; set `--device cuda:0` if you have a GPU.
- Lower `--conf` and `--imgsz` for higher FPS (at the cost of accuracy).
- ByteTrack is robust for multi-object tracking and avoids duplicate counts well.

---

## 5) Demo video (1–2 minutes)

Record your screen (e.g., OBS) while running the script for ~20–30 seconds. Show:
1. Lanes visible + running counts increasing per lane
2. The `vehicle_counts.csv` content and the printed summary
3. Optional: A quick mention of the approach in voiceover/captions

---

## 6) Repository structure (suggested)

```
traffic-flow-analysis/
├─ lanes_annotator.py
├─ traffic_flow_counter.py
├─ requirements.txt
├─ lanes.json                 # produced by annotator
├─ traffic.mp4                # downloaded automatically if missing
├─ vehicle_counts.csv         # produced by main script
├─ vehicle_counts_summary.json
└─ README.md
```

---

## 7) Technical summary (for submission)

- **Detector**: Ultralytics YOLO (COCO weights) filtered to {car, truck, bus, motorbike}.
- **Tracker**: ByteTrack via `supervision` for robust ID persistence across frames.
- **Lane logic**: Three user-defined polygons. A track is counted in a lane **once** after
  a short dwell (`--dwell` frames) to avoid flicker. We log VehicleID/Lane/Frame/Timestamp.
- **Output**: Overlayed video, CSV per event, and JSON summary.

**Challenges & solutions**
- *Lane assignment precision*: Used polygon containment of the bbox center and a dwell window.
- *Duplicate counting*: Per-(ID,lane) set ensures each ID is counted once per lane.
- *Performance*: Lightweight YOLO model (`yolov8n.pt`) and filtered classes.

---

## 8) Run options (quick reference)

- `--classes car,truck,bus,motorbike` (default) — change if needed.
- `--dwell N` — increase to require more frames inside a lane before counting.
- `--device cuda:0` — enable GPU if available.
- `--conf 0.25..0.5` — confidence threshold tradeoff between recall and precision.
- `--imgsz 480..960` — image size (speed/accuracy tradeoff).
```

