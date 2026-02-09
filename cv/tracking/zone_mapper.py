def bbox_center(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) // 2, (y1 + y2) // 2

def point_in_rect(px, py, rect):
    return rect["x1"] <= px <= rect["x2"] and rect["y1"] <= py <= rect["y2"]

def assign_to_zones(detections, zones):
    """
    For each detection, assign to at most one zone using bbox center point.
    Returns list of detections with 'zone_id' (or None).
    """
    out = []
    for d in detections:
        cx, cy = bbox_center(d["bbox"])
        zone_id = None
        for z in zones:
            if z.get("shape") == "rect" and point_in_rect(cx, cy, z):
                zone_id = z["zone_id"]
                break
        d2 = dict(d)
        d2["zone_id"] = zone_id
        out.append(d2)
    return out

def count_by_zone(detections, zones):
    counts = {z["zone_id"]: 0 for z in zones}
    for d in detections:
        zid = d.get("zone_id")
        if zid in counts:
            counts[zid] += 1
    return counts