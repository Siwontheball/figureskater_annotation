import yaml
import cv2
import numpy as np
import subprocess, os

# 1) bring in our video I/O and drawing helper
from utils.video_utils import open_video, init_writer, draw_feet_ellipse

# 2) bring in YOLO + motion‐filter classes
from yolo_inference import YoloDetector
from motion_filter import MotionFilter

def is_on_ice(frame, cx, cy, patch_h=10, thresh=0.6):
    # sample a small patch just above the feet
    y0 = max(cy - patch_h, 0)
    patch = frame[y0:cy, cx-patch_h:cx+patch_h]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = (hsv[:,:,2] > 200) & (hsv[:,:,1] < 40)
    return float(mask.sum()) / mask.size > thresh

def main():
    # ─── 1) load settings ─────────────────────────────────────────────────────
    cfg = yaml.safe_load(open("config.yaml"))
    detector = YoloDetector(cfg["yolo_model"])
    motfilt  = MotionFilter(cfg["min_velocity"])

    # ─── 2) open video & prepare output ──────────────────────────────────────
    cap, fps, size = open_video("input/segment.webm")
    writer = init_writer("output2/annotated_segment2.webm", fps, size)

    # ─── 3) grab first frame for optical flow init ──────────────────────────
    ret, prev_frame = cap.read()
    if not ret:
        print("Cannot read input video!")
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # ─── 4) process each frame ───────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # A) estimate camera shift via optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam_dx, cam_dy = motfilt.estimate_camera_shift(prev_gray, gray)

        # B) detect & track people
        detections = detector.detect_and_track(frame)

        # C) filter out anyone moving slower than your skater‐threshold
        skaters = motfilt.filter(detections, cam_dx, cam_dy)

        # ── D) additional filters ─────────────────────────────────
        # 1) Y-position: only keep those low enough on screen
        ice_line = frame.shape[0] * 0.6
        final_skaters = []

        # 2) White-ice check: sample under each feet
        for t in skaters:
            x1,y1,x2,y2 = map(int, t['xyxy'])
            cx, cy = (x1+x2)//2, y2
            if cy > ice_line and is_on_ice(frame, cx, cy):
                final_skaters.append(t)
        skaters = final_skaters
        # ──────────────────────────────────────────────────────────

        # E) draw your flat ellipse on the remaining skaters
        out_frame = draw_feet_ellipse(frame, skaters)
        writer.write(out_frame)


        # F) prepare for next loop iteration
        prev_gray = gray

    # ─── cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    print("Done—output saved to output2/annotated_segment3.webm")


if __name__ == "__main__":
    main()
