# Figure Skater Annotation

## What it does  
This repo combines a real-time YOLOv8 skater detector with a trained jump-classifier (LSTM/TCN) to annotate figure‐skate videos.
I annotated only the skaters by using the following method.
for each frame t:
  1. Track bbox → compute (u_t,v_t) centroid
  2. Estimate camera shift → (Δu_cam, Δv_cam)
  3. Compute object motion:
       Δu_obj = (u_t - u_{t-1}) - Δu_cam
       Δv_obj = (v_t - v_{t-1}) - Δv_cam
       v = sqrt(Δu_obj^2 + Δv_obj^2) / Δt
  4. Base‐patch brightness:
       x_c = (x1 + x2)/2, y_b = y2
       Ī = mean of I(x, y_b) for x in [x_c - w/2, x_c + w/2]
       if Ī ≥ T:
         ground_contact = True
       else:
         ground_contact = False
Each skater is boxed in green ellipse.

## Requirements  

# clone & enter
git clone https://github.com/<your-name>/figureskater_annotation_yolo.git
cd figureskater_annotation_yolo

# (a) set up Python
python -m venv .venv           # once
source .venv/bin/activate
pip install -r requirements.txt

# (b) drop your video in input/
mkdir -p input
cp /path/to/your_video.mp4 input/

# (c) run
python main.py --video input/my_video.mp4 --out output/ --device cpu