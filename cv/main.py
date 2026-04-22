import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from cv.pipeline import CVPipeline
from cv.utils.fps import FPS
import yaml

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_yaml("config/cv.yaml")
    cam_cfg = cfg["camera"]

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam_cfg["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam_cfg["height"]))
    cap.set(cv2.CAP_PROP_FPS, int(cam_cfg["fps"]))
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # disable auto-exposure flicker

    pipeline = CVPipeline("config/cv.yaml", "config/zones.json")
    fps = FPS()

    print("CV Service running. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from camera.")
            break

        annotated, debug = pipeline.step(frame)
        f = fps.tick()

        # overlay fps + counts
        cv2.putText(annotated, f"FPS: {f:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        y = 70
        if debug.get("counts"):
            for k, v in debug["counts"].items():
                cv2.putText(annotated, f"{k}: {v}", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                y += 28

        anomaly_count = debug.get("anomaly_count", 0)
        color = (0, 0, 255) if anomaly_count > 0 else (255, 255, 255)
        cv2.putText(annotated, f"Anomalies: {anomaly_count}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.imshow("Inventory CV (Phase 2)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()