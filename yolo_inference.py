from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path):
        # loads yolov8x.pt (or whatever you specify in config.yaml)
        self.model = YOLO(model_path)

    def detect_and_track(self, frame):
        """
        Runs YOLOv8.track on a single frame.
        Returns a list of dicts: {'xyxy': [x1,y1,x2,y2], 'id': track_id}
        """
        # Only look for people (COCO class 0), use ByteTrack
        # in yolo_inference.py
        results = self.model.track(
            source=frame,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False
        )[0]

        dets = []
        for box in results.boxes:
            # box.xyxy is a tensor [[x1,y1,x2,y2]]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            track_id = int(box.id)           # the tracker ID
            dets.append({"xyxy":[x1,y1,x2,y2], "id":track_id})
        return dets
