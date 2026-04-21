import json
import cv2

class QRReader:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def decode_bgr(self, frame_bgr):
        data, points, _ = self.detector.detectAndDecode(frame_bgr)
        if data and data.strip():
            s = data.strip()
            return s, self._try_parse(s)
        return None, None
    
    def decode_multi_bgr(self, frame_bgr):
        """
        Returns a list of QR results:
          [{"raw": str, "payload": dict|None, "bbox": [x1,y1,x2,y2], "center": (cx,cy)}]
        Requires OpenCV build that supports detectAndDecodeMulti.
        If not supported, safely falls back to single decode.
        """
        results = []

        if hasattr(self.detector, "detectAndDecodeMulti"):
            ok, decoded_info, points, _ = self.detector.detectAndDecodeMulti(frame_bgr)
            if ok and points is not None and len(decoded_info) == len(points):
                for s, pts in zip(decoded_info, points):
                    if not s or not s.strip():
                        continue
                    s = s.strip()
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1, x2 = int(min(xs)), int(max(xs))
                    y1, y2 = int(min(ys)), int(max(ys))
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    results.append({
                        "raw": s,
                        "payload": self._try_parse(s),
                        "bbox": [x1, y1, x2, y2],
                        "center": (cx, cy),
                    })
                return results

        # fallback: try single
        raw, payload = self.decode_bgr(frame_bgr)
        if raw:
            # no bbox available in fallback; return center unknown
            results.append({"raw": raw, "payload": payload, "bbox": None, "center": None})
        return results

    def decode_roi(self, frame_bgr, bbox, pad=12):
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(w - 1, int(x2) + pad)
        y2 = min(h - 1, int(y2) + pad)

        if x2 <= x1 or y2 <= y1:
            return None, None

        roi = frame_bgr[y1:y2, x1:x2]
        return self.decode_bgr(roi)

    def _try_parse(self, s: str):
        try:
            return json.loads(s)
        except Exception:
            return None