from typing import Dict, List, Optional, Tuple
import math
import time

def bbox_center(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

class SimpleTracker:
    """
    Lightweight multi-object tracker using centroid matching (greedy).
    Emits per-object events:
      - transfers: Zone_A -> Zone_B (direct) OR Zone_A -> None -> Zone_B (gap)
      - enters:    Outside(None) -> Zone_X
      - exits:     Zone_X -> Outside(None) confirmed after max_zone_gap_frames
                  OR when a track expires while outside.

    Notes:
      - "Outside" is represented by zone_id == None.
      - We do NOT emit exit immediately when it becomes None because it might be a gap transfer.
        We emit exit only after the gap exceeds max_zone_gap_frames, or on track expiration.
    """
    def __init__(
        self,
        max_age_frames: int = 30,
        match_dist_px: float = 140.0,
        max_zone_gap_frames: int = 10,
        enforce_same_label: bool = True,
    ):
        self.next_id = 1
        self.tracks: Dict[int, dict] = {}
        self.max_age_frames = int(max_age_frames)
        self.match_dist_px = float(match_dist_px)
        self.max_zone_gap_frames = int(max_zone_gap_frames)
        self.enforce_same_label = bool(enforce_same_label)
        self.frame_i = 0

    def _start_gap(self, t: dict, from_zone: str):
        if t.get("zone_gap_start_frame") is None:
            t["zone_gap_from"] = from_zone
            t["zone_gap_start_frame"] = self.frame_i
            t["exit_emitted"] = False

    def _clear_gap(self, t: dict):
        t["zone_gap_from"] = None
        t["zone_gap_start_frame"] = None
        t["exit_emitted"] = False

    def update(self, detections: List[dict]) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
        """
        detections: list of dicts with keys: bbox, label, conf, zone_id
        Returns:
          tracks_out: list[{track_id,label,conf,bbox,zone_id,prev_zone_id}]
          transfers:  list[{track_id,label,from_zone,to_zone,reason}]
          enters:     list[{track_id,label,to_zone,reason}]
          exits:      list[{track_id,label,from_zone,reason}]
        """
        self.frame_i += 1
        now = time.time()

        # Prepare detection centers
        det_centers = []
        for d in detections:
            cx, cy = bbox_center(d["bbox"])
            det_centers.append((cx, cy))

        # Mark tracks as unmatched initially
        for tid in self.tracks:
            self.tracks[tid]["matched"] = False

        # Match detections to existing tracks (greedy nearest)
        assigned_track = [-1] * len(detections)
        used_tracks = set()

        # Greedy nearest-centroid matching
        for di, d in enumerate(detections):
            best_tid = None
            best_dist = 1e9
            cx, cy = det_centers[di]

            for tid, t in self.tracks.items():
                if tid in used_tracks:
                    continue
                # Option: enforce same label to reduce ID swaps
                if self.enforce_same_label and t["label"] != d["label"]:
                    continue
                tcx, tcy = t["center"]
                dd = dist((cx, cy), (tcx, tcy))
                if dd < best_dist and dd <= self.match_dist_px:
                    best_dist = dd
                    best_tid = tid

            if best_tid is not None:
                assigned_track[di] = best_tid
                used_tracks.add(best_tid)
                self.tracks[best_tid]["matched"] = True

        transfers = []
        enters = []
        exits = []
        tracks_out = []

        # Update matched tracks / create new tracks
        for di, d in enumerate(detections):
            cx, cy = det_centers[di]
            tid = assigned_track[di]
            new_zone = d.get("zone_id")
            label = d["label"]

            if tid == -1:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "label": label,
                    "center": (cx, cy),
                    "bbox": d["bbox"],
                    "conf": d["conf"],
                    "zone_id": new_zone,
                    "prev_zone_id": None,
                    "last_seen_frame": self.frame_i,
                    "last_seen_time": now,
                    "matched": True,

                    # gap tracking
                    "zone_gap_from": None,
                    "zone_gap_start_frame": None,
                    "exit_emitted": False,
                }
                
                if new_zone is not None:
                    enters.append({
                        "track_id": tid,
                        "label": label,
                        "to_zone": new_zone,
                        "reason": "new_track_in_zone",
                    })

            else:
                t = self.tracks[tid]
                prev_zone = t.get("zone_id")
                # new_zone = d.get("zone_id")

                # Store previous zone for output/debug
                t["prev_zone_id"] = prev_zone

                # Update geometry/conf
                t["center"] = (cx, cy)
                t["bbox"] = d["bbox"]
                t["conf"] = d["conf"]
                t["last_seen_frame"] = self.frame_i
                t["last_seen_time"] = now

                # ---- Events ----

                # A) direct transfer: Zone -> Zone
                if prev_zone is not None and new_zone is not None and prev_zone != new_zone:
                    transfers.append({
                        "track_id": tid,
                        "label": label,
                        "from_zone": prev_zone,
                        "to_zone": new_zone,
                        "reason": "direct_zone_change",
                    })
                    self._clear_gap(t)

                # B) leaving zone into outside: start gap (but don't exit yet)
                if prev_zone is not None and new_zone is None:
                    self._start_gap(t, from_zone=prev_zone)

                # C) entering a zone from outside:
                #    could be (i) a gap transfer completion OR (ii) a true enter from outside
                if prev_zone is None and new_zone is not None:
                    gap_from = t.get("zone_gap_from")
                    gap_start = t.get("zone_gap_start_frame")

                    if gap_from is not None and gap_start is not None:
                        gap_len = self.frame_i - gap_start
                        if gap_len <= self.max_zone_gap_frames and gap_from != new_zone:
                            # complete a gap transfer
                            transfers.append({
                                "track_id": tid,
                                "label": label,
                                "from_zone": gap_from,
                                "to_zone": new_zone,
                                "reason": "no_zone_gap",
                            })
                        else:
                            # gap too long => treat as enter (it disappeared then reappeared)
                            enters.append({
                                "track_id": tid,
                                "label": label,
                                "to_zone": new_zone,
                                "reason": "enter_after_long_gap",
                            })
                        self._clear_gap(t)
                    else:
                        # no gap recorded => true enter
                        enters.append({
                            "track_id": tid,
                            "label": label,
                            "to_zone": new_zone,
                            "reason": "enter_from_outside",
                        })
                # Update current zone at end
                t["zone_id"] = new_zone

            # Add to output list
            t2 = self.tracks[tid]
            tracks_out.append({
                "track_id": tid,
                "label": t2["label"],
                "conf": t2["conf"],
                "bbox": t2["bbox"],
                "zone_id": t2.get("zone_id"),
                "prev_zone_id": t2.get("prev_zone_id"),
            })

        # Handle unmatched tracks: confirm exits if gap is too long
        for tid, t in list(self.tracks.items()):
            if t.get("matched"):
                continue

            # still alive but not detected this frame
            age = self.frame_i - t["last_seen_frame"]
            if age > self.max_age_frames:
                # Track expires. If it was outside with an active gap, emit exit once.
                gap_from = t.get("zone_gap_from")
                gap_start = t.get("zone_gap_start_frame")
                if gap_from is not None and gap_start is not None and not t.get("exit_emitted", False):
                    exits.append({
                        "track_id": tid,
                        "label": t["label"],
                        "from_zone": gap_from,
                        "reason": "exit_on_expire",
                    })
                self.tracks.pop(tid, None)
                continue
            
            # If it has an active gap and it has lasted too long => confirm exit
            gap_from = t.get("zone_gap_from")
            gap_start = t.get("zone_gap_start_frame")
            if gap_from is not None and gap_start is not None and not t.get("exit_emitted", False):
                gap_len = self.frame_i - gap_start
                if gap_len > self.max_zone_gap_frames:
                    exits.append({
                        "track_id": tid,
                        "label": t["label"],
                        "from_zone": gap_from,
                        "reason": "exit_after_gap_timeout",
                    })
                    t["exit_emitted"] = True
                    # clear gap so we don't later convert this into a transfer
                    self._clear_gap(t)
                    # keep zone_id as None (outside)
                    t["zone_id"] = None

        return tracks_out, transfers, enters, exits