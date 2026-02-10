import cv2
from cv.qr.qr_reader import QRReader

def main(cam_index=0):
    qr = QRReader()
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    print("Show a QR to camera. Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        raw, payload = qr.decode_bgr(frame)
        if raw:
            print("QR:", raw)
            if payload:
                print("Parsed:", payload)

        cv2.imshow("QR Test", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()