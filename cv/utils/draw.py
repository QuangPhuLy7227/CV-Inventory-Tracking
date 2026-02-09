import cv2

def draw_rect_zone(frame, zone, color=(255, 255, 255), thickness=2):
    x1, y1, x2, y2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, zone["zone_id"], (x1 + 6, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def draw_bbox(frame, bbox, label, conf, color=(255, 255, 255)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)