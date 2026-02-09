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

    cap = cv2.VideoCapture(int(cam_cfg["index"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam_cfg["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam_cfg["height"]))
    cap.set(cv2.CAP_PROP_FPS, int(cam_cfg["fps"]))

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

        if debug.get("counts"):
            y = 70
            for k, v in debug["counts"].items():
                cv2.putText(annotated, f"{k}: {v}", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                y += 28

        cv2.imshow("Inventory CV (Phase 2)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()