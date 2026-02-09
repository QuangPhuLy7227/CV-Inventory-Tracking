from collections import defaultdict, deque

class ZoneStateTracker:
    """
    Debounced zone-count change detector.
    - Tracks counts per zone over time
    - Emits changes only after min_stable_frames confirmations
    """

    def __init__(self, zones, min_stable_frames=3):
        self.zones = [z["zone_id"] for z in zones]
        self.min_stable_frames = max(1, int(min_stable_frames))

        self.prev_counts = {z: 0 for z in self.zones}
        self.candidate = {}  # zone_id -> (new_count, stable_frames)

    def update(self, counts_now):
        """
        Returns list of zone changes:
          { zone_id, old, new }
        """
        changes = []

        for zid in self.zones:
            old = self.prev_counts.get(zid, 0)
            new = counts_now.get(zid, 0)

            if new == old:
                # reset candidate if it matches old (no change)
                if zid in self.candidate:
                    self.candidate.pop(zid, None)
                continue

            # changed -> debounce
            cand = self.candidate.get(zid)
            if cand is None or cand[0] != new:
                self.candidate[zid] = (new, 1)
            else:
                self.candidate[zid] = (new, cand[1] + 1)

            if self.candidate[zid][1] >= self.min_stable_frames:
                changes.append({"zone_id": zid, "old": old, "new": new})
                self.prev_counts[zid] = new
                self.candidate.pop(zid, None)

        return changes

def infer_transfers(changes):
    """
    Infer a simple from->to transfer if one zone decremented and another incremented.
    Returns list of transfers:
      { from_zone, to_zone }
    Also returns residual changes that are not paired.
    """
    dec = [c for c in changes if c["new"] < c["old"]]
    inc = [c for c in changes if c["new"] > c["old"]]

    transfers = []
    used_inc = set()
    used_dec = set()

    for i, d in enumerate(dec):
        for j, u in enumerate(inc):
            if j in used_inc:
                continue
            # Pair one decrement with one increment (basic)
            transfers.append({"from_zone": d["zone_id"], "to_zone": u["zone_id"]})
            used_inc.add(j)
            used_dec.add(i)
            break

    residual = []
    for i, c in enumerate(changes):
        # if this change belongs to a used inc/dec, skip
        if c in [dec[k] for k in used_dec] or c in [inc[k] for k in used_inc]:
            continue
        residual.append(c)

    return transfers, residual