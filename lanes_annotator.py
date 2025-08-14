#!/usr/bin/env python3
"""
lanes_annotator.py
Create and save 3 (or more) lane polygons for a given video.
- Left-click to add points for the current polygon
- 'n' to finish the current lane and start a new one
- 'u' to undo last point for the current lane
- 'r' to reset current unfinished lane
- 's' to save lanes JSON and exit
- 'q' to exit without saving

If --video is missing, the script will try to download from --video_url (YouTube) using yt-dlp.
"""

import argparse, os, json, sys, time
from typing import List, Tuple
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

def draw_ui(frame, lanes: List[np.ndarray], current_pts: List[Tuple[int,int]]):
    vis = frame.copy()
    # Existing lanes
    for idx, poly in enumerate(lanes):
        if len(poly) >= 3:
            cv2.polylines(vis, [poly], isClosed=True, color=(0,255,0), thickness=2)
            # Put index label
            M = poly.mean(axis=0).astype(int).ravel()
            cv2.putText(vis, f"Lane {idx+1}", (M[0], M[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
    # Current lane (unfinished)
    if len(current_pts) > 0:
        pts = np.array(current_pts, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=False, color=(0,165,255), thickness=2)
        for p in current_pts:
            cv2.circle(vis, p, 4, (0,165,255), -1)
    # Instructions overlay
    cv2.rectangle(vis, (0,0), (vis.shape[1], 70), (0,0,0), -1)
    cv2.putText(vis, "Left-click: add point | 'n': finish lane | 'u': undo | 'r': reset lane | 's': save & exit | 'q': quit",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return vis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="traffic.mp4",
                        help="Path to local video file (will be created if downloaded)")
    parser.add_argument("--video_url", type=str, default="https://www.youtube.com/watch?v=MNn9qKG2UFI",
                        help="YouTube video URL to download if local file missing")
    parser.add_argument("--lanes_out", type=str, default="lanes.json",
                        help="Where to save the lanes JSON (list of polygons)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[INFO] Local video '{args.video}' not found.")
        out = download_video_ytdlp(args.video_url, args.video)
        if not out or not os.path.exists(args.video):
            print("[ERROR] Could not obtain video. Provide --video or install yt-dlp.")
            sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("[ERROR] Failed to open video")
        sys.exit(1)

    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("[ERROR] Could not read first frame.")
        sys.exit(1)

    lanes = []  # list of np.ndarray (N,2), int32
    current_pts = []

    def on_mouse(event, x, y, flags, param):
        nonlocal current_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            current_pts.append((x,y))

    cv2.namedWindow("Annotate Lanes", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotate Lanes", 1280, 720)
    cv2.setMouseCallback("Annotate Lanes", on_mouse)

    print("[INFO] Start annotating lanes. Create 3 polygons (lanes).")
    while True:
        vis = draw_ui(frame, lanes, current_pts)
        cv2.imshow("Annotate Lanes", vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('n'):
            if len(current_pts) >= 3:
                lanes.append(np.array(current_pts, dtype=np.int32))
                current_pts = []
                print(f"[INFO] Lane {len(lanes)} saved (temporary).")
            else:
                print("[WARN] Need at least 3 points to form a polygon.")
        elif key == ord('u'):
            if current_pts:
                current_pts.pop()
        elif key == ord('r'):
            current_pts = []
        elif key == ord('s'):
            break
        elif key == ord('q'):
            print("[INFO] Quit without saving.")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()
    # Save lanes as list of lists
    serializable = [poly.reshape(-1,2).tolist() for poly in lanes]
    with open(args.lanes_out, "w", encoding="utf-8") as f:
        json.dump({"lanes": serializable}, f, indent=2)
    print(f"[OK] Saved lanes to {args.lanes_out}")

if __name__ == "__main__":
    main()
