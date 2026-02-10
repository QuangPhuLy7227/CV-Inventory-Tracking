import json
import cv2

class QRReader:
    """
    Uses OpenCV QRCodeDetector to decode QR codes from:
      - full frame (optional)
      - cropped ROI (recommended for speed)
    Returns:
      decoded_str, parsed_payload (dict or None)
    """
    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def decode_bgr(self, frame_bgr):
        data, points, _ = self.detector.detectAndDecode(frame_bgr)
        if data and data.strip():
            return data.strip(), self._try_parse(data.strip())
        return None, None
    
    def decode_roi(self, frame_bgr, bbox, pad=12):
        """
        bbox: [x1,y1,x2,y2]
        pad: add margin to improve decode success
        """
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
        # Try JSON first (our recommended format)
        try:
            return json.loads(s)
        except Exception:
            return None

    
    