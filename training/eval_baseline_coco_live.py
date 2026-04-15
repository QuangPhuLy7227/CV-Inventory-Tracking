import cv2
from ultralytics import YOLO

def main(model_name="yolov8s.pt", conf=0.25, cam_index=0):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    print("COCO baseline running. Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=conf, verbose=False)
        annotated = results[0].plot()  # draw all COCO detections
        cv2.imshow("COCO Baseline", annotated)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()