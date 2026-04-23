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
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _preprocess(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.filter2D(gray, -1, _SHARPEN_KERNEL)

    def _preprocess_clahe(self, frame_bgr):
        """CLAHE contrast enhancement + sharpen — helps small/low-contrast QR codes."""
        if len(frame_bgr.shape) == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr
        enhanced = self._clahe.apply(gray)
        return cv2.filter2D(enhanced, -1, _SHARPEN_KERNEL)

    def _preprocess_thresh(self, frame_bgr):
        """Otsu binarization — clean black/white image, best for QR cell contrast."""
        if len(frame_bgr.shape) == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _zoom(self, img, scale=2.0):
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

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

        Pass 1 — OpenCV detectAndDecodeMulti on raw + sharpened + CLAHE frames.
        Pass 2 — pyzbar on raw + sharpened + CLAHE frames, plus 2x zoomed versions
                  so small QR codes in large frames become readable.
        Results are merged; first detection of a given raw string wins for bbox.
        """
        results = {}  # raw -> result dict

        # --- Pass 1: OpenCV ---
        if hasattr(self.detector, "detectAndDecodeMulti"):
            for img in (frame_bgr, self._preprocess(frame_bgr),
                        self._preprocess_clahe(frame_bgr), self._preprocess_thresh(frame_bgr)):
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

        # --- Pass 2: pyzbar — normal + 2x + 3x zoomed variants with multiple preprocessings ---
        if _PYZBAR_AVAILABLE:
            z2 = self._zoom(frame_bgr, scale=2.0)
            z3 = self._zoom(frame_bgr, scale=3.0)

            variants = [
                # normal size
                (frame_bgr,                         1.0),
                (self._preprocess(frame_bgr),       1.0),
                (self._preprocess_clahe(frame_bgr), 1.0),
                (self._preprocess_thresh(frame_bgr),1.0),
                # 2x zoom
                (z2,                                2.0),
                (self._preprocess(z2),              2.0),
                (self._preprocess_clahe(z2),        2.0),
                (self._preprocess_thresh(z2),       2.0),
                # 3x zoom
                (z3,                                3.0),
                (self._preprocess_thresh(z3),       3.0),
            ]

            for img, scale in variants:
                if results:  # stop as soon as we have hits (avoids unnecessary work)
                    decoded = _pyzbar_decode(img, symbols=[ZBarSymbol.QRCODE])
                    new_hits = [r for r in decoded if r.data]
                    # only skip if ALL codes already found
                    already_have = all(
                        r.data.decode("utf-8", errors="ignore").strip() in results
                        for r in new_hits if r.data
                    )
                    if already_have and new_hits:
                        continue
                decoded = _pyzbar_decode(img, symbols=[ZBarSymbol.QRCODE])
                for r in decoded:
                    try:
                        s = r.data.decode("utf-8").strip()
                    except Exception:
                        continue
                    if not s or s in results:
                        continue
                    # scale bbox back to original frame coordinates
                    x1 = int(r.rect.left / scale)
                    y1 = int(r.rect.top / scale)
                    x2 = int((r.rect.left + r.rect.width) / scale)
                    y2 = int((r.rect.top + r.rect.height) / scale)
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

    def decode_roi_zoomed(self, frame_bgr, bbox, pad=20, zoom=4.0):
        """
        Crop the spool bbox, zoom it up, and try all preprocessing variants.
        Returns same format as decode_multi_bgr but bboxes are in original frame coords.
        """
        fh, fw = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(fw - 1, int(x2) + pad)
        y2 = min(fh - 1, int(y2) + pad)

        if x2 <= x1 or y2 <= y1:
            return []

        roi = frame_bgr[y1:y2, x1:x2]
        zoomed = self._zoom(roi, scale=zoom)
        results = {}

        variants = [
            zoomed,
            self._preprocess(zoomed),
            self._preprocess_clahe(zoomed),
            self._preprocess_thresh(zoomed),
        ]

        # pyzbar on each variant
        if _PYZBAR_AVAILABLE:
            for img in variants:
                decoded = _pyzbar_decode(img, symbols=[ZBarSymbol.QRCODE])
                for r in decoded:
                    try:
                        s = r.data.decode("utf-8").strip()
                    except Exception:
                        continue
                    if not s or s in results:
                        continue
                    bx1 = x1 + int(r.rect.left / zoom)
                    by1 = y1 + int(r.rect.top / zoom)
                    bx2 = x1 + int((r.rect.left + r.rect.width) / zoom)
                    by2 = y1 + int((r.rect.top + r.rect.height) / zoom)
                    results[s] = {
                        "raw": s,
                        "payload": self._try_parse(s),
                        "bbox": [bx1, by1, bx2, by2],
                        "center": ((bx1 + bx2) // 2, (by1 + by2) // 2),
                    }

        # OpenCV on each variant
        if hasattr(self.detector, "detectAndDecodeMulti"):
            for img in variants:
                ok, decoded_info, points, _ = self.detector.detectAndDecodeMulti(img)
                if not ok or points is None or len(decoded_info) != len(points):
                    continue
                for s, pts in zip(decoded_info, points):
                    if not s or not s.strip() or s.strip() in results:
                        continue
                    s = s.strip()
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    bx1 = x1 + int(min(xs) / zoom)
                    by1 = y1 + int(min(ys) / zoom)
                    bx2 = x1 + int(max(xs) / zoom)
                    by2 = y1 + int(max(ys) / zoom)
                    results[s] = {
                        "raw": s,
                        "payload": self._try_parse(s),
                        "bbox": [bx1, by1, bx2, by2],
                        "center": ((bx1 + bx2) // 2, (by1 + by2) // 2),
                    }

        return list(results.values())

    def _try_parse(self, s: str):
        try:
            return json.loads(s)
        except Exception:
            return None
