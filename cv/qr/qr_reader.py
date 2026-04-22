import json
import cv2
import numpy as np

try:
    from pyzbar.pyzbar import decode as _pyzbar_decode
    from pyzbar.pyzbar import ZBarSymbol
    _PYZBAR_AVAILABLE = True
except ImportError:
    _PYZBAR_AVAILABLE = False

_SHARPEN_KERNEL = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]], dtype=np.float32)


class QRReader:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def _preprocess(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.filter2D(gray, -1, _SHARPEN_KERNEL)

    def _upscale(self, img, min_side=200):
        h, w = img.shape[:2]
        if min(h, w) < min_side:
            scale = min_side / min(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        return img

    def decode_bgr(self, frame_bgr):
        for img in (frame_bgr, self._preprocess(frame_bgr)):
            data, points, _ = self.detector.detectAndDecode(img)
            if data and data.strip():
                s = data.strip()
                return s, self._try_parse(s)
        return None, None

    def decode_multi_bgr(self, frame_bgr):
        """
        Returns a list of QR results (deduped by content):
          [{"raw": str, "payload": dict|None, "bbox": [x1,y1,x2,y2], "center": (cx,cy)}]

        Pass 1 — OpenCV detectAndDecodeMulti on raw + sharpened frames.
        Pass 2 — pyzbar on raw + sharpened frames (catches small/rotated codes OpenCV misses).
        Results are merged; first detection of a given raw string wins for bbox.
        """
        results = {}  # raw -> result dict

        # --- Pass 1: OpenCV ---
        if hasattr(self.detector, "detectAndDecodeMulti"):
            for img in (frame_bgr, self._preprocess(frame_bgr)):
                ok, decoded_info, points, _ = self.detector.detectAndDecodeMulti(img)
                if not ok or points is None or len(decoded_info) != len(points):
                    continue
                for s, pts in zip(decoded_info, points):
                    if not s or not s.strip() or s.strip() in results:
                        continue
                    s = s.strip()
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1, x2 = int(min(xs)), int(max(xs))
                    y1, y2 = int(min(ys)), int(max(ys))
                    results[s] = {
                        "raw": s,
                        "payload": self._try_parse(s),
                        "bbox": [x1, y1, x2, y2],
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    }

        # --- Pass 2: pyzbar (runs even if OpenCV found some, to catch missed codes) ---
        if _PYZBAR_AVAILABLE:
            for img in (frame_bgr, self._preprocess(frame_bgr)):
                img = self._upscale(img)
                decoded = _pyzbar_decode(img, symbols=[ZBarSymbol.QRCODE])
                for r in decoded:
                    try:
                        s = r.data.decode("utf-8").strip()
                    except Exception:
                        continue
                    if not s or s in results:
                        continue
                    x1 = r.rect.left
                    y1 = r.rect.top
                    x2 = x1 + r.rect.width
                    y2 = y1 + r.rect.height
                    results[s] = {
                        "raw": s,
                        "payload": self._try_parse(s),
                        "bbox": [x1, y1, x2, y2],
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    }

        # Fallback: single OpenCV decode if multi not supported and pyzbar absent
        if not results and not hasattr(self.detector, "detectAndDecodeMulti") and not _PYZBAR_AVAILABLE:
            raw, payload = self.decode_bgr(frame_bgr)
            if raw:
                results[raw] = {"raw": raw, "payload": payload, "bbox": None, "center": None}

        return list(results.values())

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
