"""One process owns the pipeline: feed -> tracker -> scorer + ladder -> event log -> snapshot.
`python -m app.engine` replays Harbor Night headless, prints the event log and dumps it to JSON."""
import os
from dataclasses import asdict

from .feed import Feed
import math

import numpy as np

from .imm import MODELS
from .ladder import HIDDEN_OPTION, OPTIONS, Ladder, bearing
from .scoring import dist_to_polygon
from .models import Booking, Ledger
from .registry import Registry
from .scoring import load_config
from .tracker import Tracker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIO = os.path.join(ROOT, "scenarios", "harbor_night.json")
ASSETS = os.path.join(ROOT, "config", "assets.json")
SCORING = os.path.join(ROOT, "config", "scoring.json")
TAIL_S = 0           # the run stops with the last contact row
EDGE_MARGIN_M = 20   # a dive box is drawn once its vehicle is this close to the edge
GATE_M = 150         # triage correlation gate around an unknown contact
HOT_HOLD_S = 10      # a keep-out incursion stays lit this long after the last ETA hit
RANK = {"keepout_incursion": 1, "triage": 2, "registered_lost": 3, "out_of_area": 4, "unknown_lost": 5, "keepout_approach": 4}


class Engine:
    def __init__(self, scenario=SCENARIO, assets=ASSETS, scoring=SCORING):
        self.feed = Feed(scenario)
        self.assets_cfg = load_config(assets)
        self.scoring_cfg = load_config(scoring)
        self.speed = 1.0
        self.reset()

    def reset(self):
        self.t, self.running = 0, False
        self.ledger = Ledger()
        self.reg = Registry(self.ledger)
        for b in self.feed.bookings:
            self.reg.add(Booking(id=None, **b))
        self.tracker = Tracker()
        self.ledger.tracks = self.tracker.tracks
        self.ladder = Ladder(self.ledger, self.reg, self.feed, self.assets_cfg["assets"],
                             self.scoring_cfg, self.assets_cfg["keepout_radius_m"])

    def tick(self):
        """Advance one scenario second if running."""
        if self.running:
            self.step()

    def step(self):
        """Advance one scenario second unconditionally."""
        t = self.t
        for tr in self.tracker.step(self.feed.contacts_at(t), t):
            self.ladder.lose(tr, t)
        for tr in self.tracker.confirmed():
            self.ladder.step(tr, t)
            if tr.eta or self.inside_zone(tr.x, tr.y):
                tr.hot_until = t + HOT_HOLD_S  # hold the incursion through ETA jitter
        for tr in self.tracker.tracks.values():  # uncertainty halo of each lost contact grows under CV-quiet
            if tr.ended_at is not None and tr.cone:
                P = tr.last_fix[1] if tr.halo_P is None else tr.halo_P
                tr.halo_P = MODELS[0].predict(tr.last_fix[0], P)[1]
        self.t += 1
        if self.t > self.feed.end + TAIL_S:
            self.running = False

    # --- inputs from the web faces -------------------------------------------------------
    def control(self, action, speed=None):
        if speed:
            self.speed = float(speed)
        if action == "start":
            self.running = True
        elif action == "pause":
            self.running = False
        elif action == "reset":
            self.reset()

    def keepouts(self):
        r = self.assets_cfg["keepout_radius_m"]
        return [(tr.x, tr.y, r, tr.id) for tr in self.tracker.confirmed() if tr.rung >= 3]

    def book(self, **fields):
        return self.reg.create(Booking(id=None, **fields), self.t, self.keepouts())

    def decide(self, track_id, option, officer, note=""):
        tr = self.tracker.tracks.get(track_id)
        if tr is None or tr.ended_at is not None:
            raise ValueError(f"no live track {track_id}")
        self.ladder.decide(tr, option, officer, note, self.t)

    # --- triage actions (detail card) ---------------------------------------------------
    def live_track(self, track_id):
        tr = self.tracker.tracks.get(track_id)
        if tr is None or tr.ended_at is not None:
            raise ValueError(f"no live track {track_id}")
        return tr

    def match(self, track_id, booking_id, action, officer):
        tr, b = self.live_track(track_id), self.reg.get(booking_id)
        if b is None:
            raise ValueError(f"no booking {booking_id}")
        (self.ladder.confirm_match if action == "confirm" else self.ladder.reject_match)(tr, b, self.t, officer)

    def escalate(self, track_id, officer):
        self.ladder.escalate(self.live_track(track_id), self.t, officer)

    def ack(self, track_id, officer):
        self.ladder.ack(self.live_track(track_id), self.t, officer)

    def contact_operator(self, track_id, officer):
        tr = self.live_track(track_id)
        if tr.matched_booking_id is None:
            raise ValueError(f"{track_id} has no registered operator")
        self.ladder.contact_operator(tr, self.reg.get(tr.matched_booking_id), self.t, officer)

    # --- map symbology: one derived exception per contact ------------------------------
    def inside_zone(self, x, y):
        return next((a["name"] for a in self.assets_cfg["assets"] if dist_to_polygon(x, y, a["polygon"]) <= a["standoff_m"]), None)

    def halo_r(self, tr):
        P = tr.last_fix[1] if tr.halo_P is None else tr.halo_P
        return float(np.sqrt(np.linalg.eigvalsh(P[:2, :2]).max()))

    def candidates(self, tr):
        """Lost registered vehicles whose halo reaches the unknown's correlation gate, best first."""
        out = []
        for o in self.tracker.tracks.values():
            if o.ended_at is None or not o.cone or not o.matched_booking_id or o.matched_booking_id in tr.rejected:
                continue
            lx, ly = float(o.last_fix[0][0]), float(o.last_fix[0][1])
            d, r = math.hypot(tr.x - lx, tr.y - ly), self.halo_r(o)
            if d <= GATE_M + r:
                b = self.reg.get(o.matched_booking_id)
                out.append({"track_id": o.id, "booking_id": b.id, "vehicle_id": b.vehicle_id, "operator": b.operator,
                            "distance_m": round(d), "score": round(100 * max(0.0, 1 - d / (GATE_M + r)))})
        return sorted(out, key=lambda c: -c["score"])

    def exception(self, tr):
        """(name, rank) for a live contact; (None, 0) when nominal."""
        hot = self.t <= tr.hot_until
        if tr.matched_booking_id:
            if hot:
                return "keepout_approach", RANK["keepout_approach"]
            return ("out_of_area", RANK["out_of_area"]) if tr.out_of_area else (None, 0)
        if hot:
            return "keepout_incursion", RANK["keepout_incursion"]
        return "triage", RANK["triage"]

    # --- drone tasking: published after an officer releases it ---------------------------
    def latlon(self, x, y):
        o = self.assets_cfg["origin"]
        return {"lat": round(o["lat"] + y / 111320, 6),
                "lon": round(o["lon"] + x / (111320 * math.cos(math.radians(o["lat"]))), 6)}

    def task_json(self, task):
        tr = self.tracker.tracks.get(task.track_id)
        if tr is not None and tr.ended_at is not None and task.status not in ("complete", "cancelled", "track_lost"):
            task.status = "track_lost"
        state = tr.last_fix[0] if tr is not None and tr.ended_at is not None else tr.state
        x, y, vx, vy = (float(v) for v in state)
        path = tr.filter.predict_ahead(120)
        return {"id": task.id, "track_id": task.track_id, "status": task.status, "issued_t": task.t,
                "issued_clock": f"{2 + int(task.t) // 3600:02d}:{int(task.t) % 3600 // 60:02d}:{int(task.t) % 60:02d}",
                "authorized_by": task.officer, "objective": task.objective,
                "target": {"x_m": round(x, 1), "y_m": round(y, 1), **self.latlon(x, y),
                           "depth_m": round(tr.depth, 1), "course_deg": round(bearing(vx, vy)),
                           "speed_mps": round(math.hypot(vx, vy), 2), "updated_t": tr.last_seen,
                           "live": tr.ended_at is None, "rung": tr.rung, "posterior": round(tr.posterior, 2),
                           "headline": tr.card["headline"] if tr.card else None},
                "predicted": [{"s": s, **self.latlon(float(path[s][0]), float(path[s][1]))} for s in (30, 60, 120)],
                "asset_eta": tr.eta, "standoff_m": self.assets_cfg["keepout_radius_m"],
                "origin": self.assets_cfg["origin"], "reports": task.reports}

    def tasks_json(self, include_done=False):
        out = [self.task_json(t) for t in self.ledger.tasks]
        return out if include_done else [t for t in out if t["status"] not in ("complete", "cancelled", "track_lost")]

    def report(self, task_id, status, note="", media_url=""):
        task = next((t for t in self.ledger.tasks if t.id == task_id), None)
        if task is None:
            raise ValueError(f"no tasking {task_id}")
        task.status = status
        task.reports.append({"t": self.t, "status": status, "note": note, "media_url": media_url})
        self.ledger.log(self.t, task.track_id, "drone_report", f"{task.id} {status}" + (f": {note}" if note else "") + (f" [{media_url}]" if media_url else ""))
        return task

    # --- snapshot polled at 1 Hz ---------------------------------------------------------
    def track_json(self, tr):
        path = tr.filter.predict_ahead(60)[::5]
        b = self.reg.get(tr.matched_booking_id) if tr.matched_booking_id else None
        name, rank = self.exception(tr)
        d = {"id": tr.id, "x": tr.x, "y": tr.y, "vx": tr.vx, "vy": tr.vy, "speed": tr.speed,
             "heading_deg": round(bearing(tr.vx, tr.vy)), "depth": tr.depth, "rung": tr.rung,
             "posterior": tr.posterior, "features": tr.features, "mode": tr.filter.mode, "modes": tr.filter.modes,
             "booking_id": tr.matched_booking_id, "resolution": tr.resolution, "decision": tr.decision,
             "out_of_area": tr.out_of_area, "acked": tr.acked, "human": tr.human,
             "exception": name, "rank": rank, "zone": self.inside_zone(tr.x, tr.y),
             "time_in_state": self.t - (tr.out_of_area_since if name == "out_of_area" and tr.out_of_area_since is not None else tr.rung_since),
             "uncertainty_m": round(float(np.sqrt(np.linalg.eigvalsh(tr.filter.P[:2, :2]).max())), 1),
             "task": next(({"id": k.id, "status": k.status, "reports": len(k.reports)} for k in self.ledger.tasks if k.track_id == tr.id), None),
             "pings": tr.pings, "replied": tr.ping_reply is not None, "rung_since": tr.rung_since,
             "age": self.t - tr.first_seen, "since_contact": self.t - tr.last_seen,
             "predicted": [[float(x), float(y)] for x, y in path], "eta": tr.eta}
        if b:
            inside = min(tr.x - b.x_min, b.x_max - tr.x, tr.y - b.y_min, b.y_max - tr.y)
            tr.box_shown = tr.out_of_area or inside <= EDGE_MARGIN_M or (tr.box_shown and inside <= 2 * EDGE_MARGIN_M)
            d.update({"registry": {"booking_id": b.id, "operator": b.operator, "vehicle_id": b.vehicle_id, "ping_id": b.ping_id,
                                   "box": [b.x_min, b.y_min, b.x_max, b.y_max], "window": [b.window_start, b.window_end]},
                      "distance_outside": round(self.reg.distance_outside(b, tr.x, tr.y)),
                      "show_box": tr.box_shown})
        else:
            d["candidates"] = self.candidates(tr)
        return d

    def lost_json(self, tr):
        x, y, vx, vy = (float(v) for v in tr.last_fix[0])
        b = self.reg.get(tr.matched_booking_id) if tr.matched_booking_id else None
        name = "registered_lost" if b else "unknown_lost"
        return {"id": tr.id, "x": x, "y": y, "vx": vx, "vy": vy, "rung": tr.rung, "resolution": tr.resolution,
                "booking_id": tr.matched_booking_id, "vehicle_id": b.vehicle_id if b else None, "operator": b.operator if b else None,
                "lost_at": tr.ended_at, "last_seen": tr.last_seen, "since": self.t - tr.last_seen,
                "exception": name, "rank": RANK[name], "halo_r": round(self.halo_r(tr)), "depth": tr.depth,
                "heading_deg": round(bearing(vx, vy)), "speed": round(math.hypot(vx, vy), 2),
                "cone": tr.cone, "recovery": tr.recovery}

    def booking_json(self, b):
        d = asdict(b)
        d["display_status"] = ("closed" if self.t > b.window_end else "scheduled" if self.t < b.window_start
                               else "amended" if b.status == "amended" else "active")
        lost = next((tr for tr in self.tracker.tracks.values() if tr.matched_booking_id == b.id and tr.cone), None)
        d["recovery"] = lost.recovery if lost else None
        d["out_of_area"] = any(tr.matched_booking_id == b.id and tr.out_of_area and tr.ended_at is None
                               for tr in self.tracker.tracks.values())
        return d

    def counters(self):
        done = [tr for tr in self.tracker.tracks.values() if tr.resolution in ("known", "lost", "handed_off", "reacquired")]
        auto = sum(1 for tr in done if not tr.human and tr.rung < 4)
        return {"auto": auto, "total": len(done),
                "external_without_human": sum(1 for a in self.ladder.external if not a["officer"]),
                "external_total": len(self.ladder.external)}

    def snapshot(self):
        live = self.tracker.confirmed()
        return {"t": self.t, "running": self.running, "speed": self.speed, "end": self.feed.end + TAIL_S,
                "tracks": [self.track_json(tr) for tr in live],
                "lost": [self.lost_json(tr) for tr in self.tracker.tracks.values() if tr.cone],
                "queue": [self.ladder.card(tr) for tr in sorted(live, key=lambda tr: -tr.rung_since) if tr.rung == 4],
                "events": [asdict(e) for e in self.ledger.recent(20)],
                "counters": self.counters(),
                "counts": {"registered": sum(1 for tr in live if tr.matched_booking_id), "unknown": sum(1 for tr in live if not tr.matched_booking_id),
                           "nominal": sum(1 for tr in live if self.exception(tr)[0] is None), "lost": sum(1 for tr in self.tracker.tracks.values() if tr.cone)},
                "bookings": [self.booking_json(b) for b in self.reg.all()],
                "options": OPTIONS, "hidden_option": HIDDEN_OPTION}

    def run(self, officer=None, dump=None):
        """Headless: run to the end. officer(card) -> option is called once per Hold card."""
        self.running, decided = True, set()
        while self.running:
            self.tick()
            for card in self.snapshot()["queue"] if officer else []:
                if card["track_id"] not in decided:
                    decided.add(card["track_id"])
                    self.decide(card["track_id"], officer(card), "test officer")
        if dump:
            self.ledger.dump(dump)
        return self.snapshot()


if __name__ == "__main__":
    eng = Engine()
    out = os.path.join(ROOT, "scenarios", "harbor_night.events.json")
    snap = eng.run(officer=lambda card: OPTIONS[2], dump=out)
    for e in eng.ledger.events:
        print(f"t={e.t:>4.0f}  {e.track_id:<5} {e.kind:<16} {e.detail}")
    c = snap["counters"]
    print(f"\nResolved without a human: {c['auto']} of {c['total']}. "
          f"External actions without a human: {c['external_without_human']} (total {c['external_total']}).")
    print("event log written to", out)
