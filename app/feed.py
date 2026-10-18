"""Scenario feed: the only mock sensor. Replays scenarios/harbor_night.json contacts at 1 Hz
and answers acoustic ID pings the way each contact's truth block says it would.
`python -m app.feed` regenerates the scenario file."""
import json
import math
import random
import sys

from .models import Contact

PING_RANGE_M = 50  # how close a scenario contact must be to a track for the ping to reach it

# Harbor Night: five contacts over 240 s. Speed in m/s, heading in degrees from east (north = 90).
# turn = (seconds after t0, duration s, rad/s): one coordinated turn; noise is added at generation.
CONTACTS = {
    "A": dict(t0=0,  t1=240, x=-250, y=-350, speed=0.424, hdg=45.0,  depth=15),  # USC glider, inside box
    "B": dict(t0=10, t1=240, x=380,  y=400,  speed=0.224, hdg=153.4, depth=8),   # inspection ROV, ~80 m outside box
    "C": dict(t0=30, t1=150, x=-50,  y=0,    speed=0.632, hdg=18.4,  depth=25),  # research AUV, late; silent after t0+120
    "D": dict(t0=60, t1=120, x=300,  y=-500, speed=0.158, hdg=18.4,  depth=2),   # debris drifting with current
    "E": dict(t0=90, t1=240, x=173,  y=-453, speed=1.5,   hdg=13.7,  depth=12,   # unknown AUV, 30-degree turn onto the fuel pier
              turn=(6, 6, 0.0873)),
}
TRUTH = {
    "A": {"identity": "USC glider",          "ping_id": "USC-GL-07", "answers_ping": True,  "reply_delay": 3},
    "B": {"identity": "Inspection ROV",      "ping_id": "PIN-ROV-2", "answers_ping": True,  "reply_delay": 3},
    "C": {"identity": "Research AUV (late, then lost)", "ping_id": "SIO-AUV-3", "answers_ping": True, "reply_delay": 3},
    "D": {"identity": "Debris / biologic",   "ping_id": None,        "answers_ping": False, "reply_delay": None},
    "E": {"identity": "Unknown AUV",         "ping_id": None,        "answers_ping": False, "reply_delay": None},
}
BOOKINGS = [
    dict(operator="USC AUV Lab", vehicle_id="usc-glider-07", ping_id="USC-GL-07",
         x_min=-350, y_min=-450, x_max=-50, y_max=-150, window_start=-600, window_end=3000),
    dict(operator="Pacific Hull Inspection", vehicle_id="pin-rov-2", ping_id="PIN-ROV-2",
         x_min=100, y_min=300, x_max=300, y_max=500, window_start=-900, window_end=2400),
    dict(operator="Scripps Research", vehicle_id="sio-auv-3", ping_id="SIO-AUV-3",
         x_min=-200, y_min=-150, x_max=100, y_max=150, window_start=-4800, window_end=-1170),
]


def truth_path(c):
    """Integrate heading at 1 Hz. Yields (t, x, y) for t0..t1."""
    x, y, psi = c["x"], c["y"], math.radians(c["hdg"])
    turn = c.get("turn")
    for t in range(c["t0"], c["t1"] + 1):
        yield t, x, y
        rel = t - c["t0"]
        if turn and turn[0] <= rel < turn[0] + turn[1]:
            psi += turn[2]
        x += c["speed"] * math.cos(psi)
        y += c["speed"] * math.sin(psi)


def generate(sigma=2.0, seed=1):
    rng = random.Random(seed)
    rows = []
    for cid, c in CONTACTS.items():
        for t, x, y in truth_path(c):
            rows.append({"t": t, "contact_id": cid, "x": round(x + rng.gauss(0, sigma), 2),
                         "y": round(y + rng.gauss(0, sigma), 2), "depth": round(c["depth"] + rng.gauss(0, 0.3), 1)})
    rows.sort(key=lambda r: (r["t"], r["contact_id"]))
    return {"bookings": BOOKINGS, "rows": rows, "truth": TRUTH}


class Feed:
    def __init__(self, path):
        with open(path) as f:
            d = json.load(f)
        self.bookings = d["bookings"]
        self.truth = d["truth"]
        self.by_t = {}
        for r in d["rows"]:
            self.by_t.setdefault(r["t"], []).append(r)
        self.end = max(self.by_t)

    def contacts_at(self, t):
        return [Contact(id=r["contact_id"], t=r["t"], x=r["x"], y=r["y"], depth=r["depth"])
                for r in self.by_t.get(t, [])]

    def ping(self, x, y, t):
        """Mock acoustic ID. The nearest scenario contact within PING_RANGE_M answers per its
        truth block. Returns (ping_id, reply_delay_s) or None for silence."""
        rows = self.by_t.get(t, [])
        if not rows:
            return None
        r = min(rows, key=lambda r: (r["x"] - x) ** 2 + (r["y"] - y) ** 2)
        if (r["x"] - x) ** 2 + (r["y"] - y) ** 2 > PING_RANGE_M ** 2:
            return None
        tr = self.truth[r["contact_id"]]
        return (tr["ping_id"], tr["reply_delay"]) if tr["answers_ping"] else None


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "scenarios/harbor_night.json"
    with open(out, "w") as f:
        json.dump(generate(), f, indent=0)
    print("wrote", out)
