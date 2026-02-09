from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path: str, conf: float, iou: float, device: str):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect(self, frame_bgr):
        """
        Returns list of detections:
          { 'label': str, 'conf': float, 'bbox': [x1,y1,x2,y2] }
        """
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            # classes=self.classes, 
            verbose=False
        )
        r = results[0]
        dets = []
        if r.boxes is None:
            return dets

        names = r.names
        for b in r.boxes:
            cls = int(b.cls.item())
            label = names.get(cls, str(cls))
            conf = float(b.conf.item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append({
                "label": label,
                "conf": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
        return dets