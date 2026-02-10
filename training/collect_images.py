import os
import time
import cv2

def main(
    out_dir="datasets/inventory_v1/raw_images",
    cam_index=0,
    every_n_frames=5,
    max_images=500
):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Try cam_index=1 or check permissions.")
    
    print("Press 's' to start/stop saving, 'q' to quit.")
    saving = False
    saved = 0
    frame_i = 0
    last_save_t = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_i += 1

        # overlay
        cv2.putText(frame, f"saving={saving} saved={saved}/{max_images}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Collect Images", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break
        if k == ord("s"):
            saving = not saving
            print("saving =", saving)

        if saving and saved < max_images and (frame_i % every_n_frames == 0):
            # avoid saving too fast
            if time.time() - last_save_t < 0.1:
                continue
            last_save_t = time.time()

            fname = f"img_{int(time.time()*1000)}.jpg"
            path = os.path.join(out_dir, fname)
            cv2.imwrite(path, frame)
            saved += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()