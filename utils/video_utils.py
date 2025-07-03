# in utils/video_utils.py

import cv2

def open_video(path):
    """
    Opens a video and returns (cap, fps, (width, height)).
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, (width, height)

def init_writer(path, fps, size):
    """
    Creates a VideoWriter for the given path, fps, and frame size.
    """
    # If you’re writing .webm, change this to 'VP80' as discussed
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    return cv2.VideoWriter(path, fourcc, fps, size)

def draw_feet_ellipse(frame, tracks, label="skater"):
    for t in tracks:
        x1,y1,x2,y2 = map(int, t['xyxy'])
        cx, cy = (x1+x2)//2, y2         # bottom-center of bbox
        w, h = x2 - x1, y2 - y1

        # pick axes: make the ellipse wide and very flat
        axis_major = max(int(w * 0.4), 16)   # ~20% of person width
        axis_minor = max(int(h * 0.05), 8) #  2% of person height

        cv2.ellipse(
            frame,
            center=(cx, cy),
            axes=(axis_major, axis_minor),
            angle=0,               # no rotation
            startAngle=0,
            endAngle=360,
            color=(0,255,0),
            thickness=4
        )

        # label
        font_scale = 1.2    # was 0.6
        font_thick = 3      # was 2
        text_size, _ = cv2.getTextSize(label,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale,
                                       font_thick)
        text_x = cx - text_size[0] // 2
        text_y = cy - axis_minor - 10  # 10px above the ellipse

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            font_thick
        )
        
    return frame
