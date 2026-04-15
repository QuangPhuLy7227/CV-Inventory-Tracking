import cv2
from ultralytics import YOLO

def main(
    coco_model="yolov8s.pt",
    spool_model="runs/inventory/spool_detector_v1/weights/best.pt",
    conf=0.25,
    cam_index=0
):
    models = {
        "COCO": YOLO(coco_model),
        "SPOOL": YOLO(spool_model),
    }
    active = "COCO"

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    print("Press '1' for COCO, '2' for SPOOL, 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = models[active].predict(frame, conf=conf, verbose=False)
        annotated = results[0].plot()
        cv2.putText(annotated, f"MODEL: {active}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        cv2.imshow("Compare Models", annotated)
        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break
        if k == ord("1"):
            active = "COCO"
        if k == ord("2"):
            active = "SPOOL"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()