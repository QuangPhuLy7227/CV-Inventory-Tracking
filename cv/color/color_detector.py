import cv2
import numpy as np

class FilamentColorDetector:
    """
    Simple, fast color detector for filament spools.
    Works best when:
      - lighting is stable
      - filament is visible (not fully wrapped)
      - QR label isn't covering most of the spool
    """

    def __init__(self):
        # Hue ranges in OpenCV HSV (H: 0-179)
        self.hue_ranges = {
            "red":   [(0, 10), (170, 179)],
            "orange":[(11, 22)],
            "yellow":[(23, 35)],
            "green": [(36, 85)],
            "blue":  [(86, 130)],
            "purple":[(131, 160)],
        }

    def classify(self, bgr_crop):
        """
        Returns: (color_name, confidence_float)
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return "unknown", 0.0

        # 1) blur to reduce noise
        img = cv2.GaussianBlur(bgr_crop, (5, 5), 0)

        # 2) convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 3) mask out low-saturation pixels (grays/whites/blacks/background)
        # You can tune these thresholds for your lighting.
        h, s, v = cv2.split(hsv)
        mask = (s > 45) & (v > 40)  # keep "colorful enough" pixels

        # If too few pixels remain, it may be black/white/gray
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        if ratio < 0.05:
            # Decide black/white/gray based on brightness
            mean_v = float(np.mean(v))
            if mean_v < 60:
                return "black", 0.6
            elif mean_v > 170:
                return "white", 0.6
            else:
                return "gray", 0.5

        # 4) compute histogram of hue in masked pixels
        hue_vals = h[mask]
        if hue_vals.size < 20:
            return "unknown", 0.0

        hist = np.bincount(hue_vals, minlength=180).astype(np.float32)
        hist /= (hist.sum() + 1e-6)

        # 5) score each color by summing hist bins inside hue ranges
        scores = {}
        for cname, ranges in self.hue_ranges.items():
            score = 0.0
            for lo, hi in ranges:
                score += float(hist[lo:hi+1].sum())
            scores[cname] = score

        # pick best color
        best = max(scores, key=scores.get)
        best_score = scores[best]

        # handle "brown" as low-brightness orange-ish
        # (optional: you can remove if you don't care)
        mean_v = float(np.mean(v[mask]))
        if best in ["orange", "yellow"] and mean_v < 90:
            best = "brown"
            best_score = max(best_score, 0.4)

        # confidence: how dominant the top hue group is
        # If top score is low, it's uncertain
        conf = min(1.0, best_score * 1.4)
        if conf < 0.25:
            return "unknown", conf

        return best, conf