import numpy as np
import cv2
from ultralytics import YOLO

class MotionFilter:
    def __init__(self, min_vel):
        self.min_vel = min_vel
        self.history = {}   # track_id → (prev_cx, prev_cy)

    def estimate_camera_shift(self, prev_gray, gray):
        """
        Dense Farneback optical flow → median gives (dx,dy) camera motion.
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3,
            winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0
        )
        dx, dy = np.median(flow.reshape(-1,2), axis=0)
        return dx, dy

    def filter(self, tracks, cam_dx, cam_dy):
        """
        Given a list of {'xyxy','id'} and camera shift,
        return only those whose true speed ≥ min_vel.
        """
        kept = []
        for t in tracks:
            tid = t['id']
            x1,y1,x2,y2 = t['xyxy']
            cx, cy = (x1+x2)/2, (y1+y2)/2

            if tid in self.history:
                px, py = self.history[tid]
                raw_dx, raw_dy = cx - px, cy - py
                # remove camera motion
                rel_dx = raw_dx - cam_dx
                rel_dy = raw_dy - cam_dy
                speed = (rel_dx**2 + rel_dy**2)**0.5
                if speed >= self.min_vel:
                    kept.append(t)

            self.history[tid] = (cx, cy)

        return kept
    
    def compute_speeds(self, tracks, cam_dx, cam_dy):
        out = []
        for t in tracks:
            tid = t['id']
            x1, y1, x2, y2 = t['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if tid in self.history:
                px, py = self.history[tid]
                raw_dx, raw_dy = cx - px, cy - py
                rel_dx = raw_dx - cam_dx
                rel_dy = raw_dy - cam_dy
                speed = (rel_dx**2 + rel_dy**2) ** 0.5
            else:
                speed = 0.0

            self.history[tid] = (cx, cy)
            t_out = t.copy()
            t_out['speed'] = speed
            out.append(t_out)

        return out