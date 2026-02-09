import cv2
import json

ZONES = []
DRAWING = False
START = None
CURRENT = None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    print("Zone Editor:")
    print("- Drag mouse to draw a rectangle zone")
    print("- Press 's' to save zones.json")
    print("- Press 'c' to clear zones")
    print("- Press 'q' to quit")

    def on_mouse(event, x, y, flags, param):
        global DRAWING, START, CURRENT, ZONES
        if event == cv2.EVENT_LBUTTONDOWN:
            DRAWING = True
            START = (x, y)
            CURRENT = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and DRAWING:
            CURRENT = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            DRAWING = False
            x1, y1 = START
            x2, y2 = CURRENT
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            zone_id = f"Zone_{len(ZONES)+1}"
            ZONES.append({"zone_id": zone_id, "type": "rack_slot", "shape": "rect", "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            print(f"Added {zone_id}: ({x1},{y1})-({x2},{y2})")

    cv2.namedWindow("Zone Editor")
    cv2.setMouseCallback("Zone Editor", on_mouse)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # draw saved zones
        for z in ZONES:
            cv2.rectangle(frame, (z["x1"], z["y1"]), (z["x2"], z["y2"]), (255, 255, 255), 2)
            cv2.putText(frame, z["zone_id"], (z["x1"] + 6, z["y1"] + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # draw current zone
        if DRAWING and START and CURRENT:
            x1, y1 = START
            x2, y2 = CURRENT
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        cv2.imshow("Zone Editor", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            with open("config/zones.json", "w", encoding="utf-8") as f:
                json.dump({"zones": ZONES}, f, indent=2)
            print("Saved to config/zones.json")

        if key == ord("c"):
            ZONES.clear()
            print("Cleared zones")

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()