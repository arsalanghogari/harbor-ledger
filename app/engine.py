"""One process owns the pipeline: feed -> tracker -> scorer + ladder -> event log -> snapshot.
`python -m app.engine` replays Harbor Night headless, prints the event log and dumps it to JSON."""
import os
from dataclasses import asdict

from .feed import Feed
import math

from .ladder import HIDDEN_OPTION, OPTIONS, Ladder, bearing
from .models import Booking, Ledger
from .registry import Registry
from .scoring import load_config
from .tracker import Tracker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIO = os.path.join(ROOT, "scenarios", "harbor_night.json")
ASSETS = os.path.join(ROOT, "config", "assets.json")
SCORING = os.path.join(ROOT, "config", "scoring.json")
TAIL_S = 0  # the run stops with the last contact; C and D go lost inside the arc


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
        return {"id": tr.id, "x": tr.x, "y": tr.y, "vx": tr.vx, "vy": tr.vy, "speed": tr.speed,
                "depth": tr.depth, "rung": tr.rung, "posterior": tr.posterior, "features": tr.features,
                "mode": tr.filter.mode, "modes": tr.filter.modes,
                "booking_id": tr.matched_booking_id, "resolution": tr.resolution, "decision": tr.decision,
                "out_of_area": tr.out_of_area,
                "task": next(({"id": k.id, "status": k.status, "reports": len(k.reports)} for k in self.ledger.tasks if k.track_id == tr.id), None),
                "pings": tr.pings, "replied": tr.ping_reply is not None, "rung_since": tr.rung_since,
                "age": self.t - tr.first_seen, "since_contact": self.t - tr.last_seen,
                "predicted": [[float(x), float(y)] for x, y in path], "eta": tr.eta,
                "keepout": self.assets_cfg["keepout_radius_m"] if tr.rung >= 3 else None}

    def lost_json(self, tr):
        x, y, vx, vy = (float(v) for v in tr.last_fix[0])
        return {"id": tr.id, "x": x, "y": y, "vx": vx, "vy": vy, "rung": tr.rung, "resolution": tr.resolution,
                "booking_id": tr.matched_booking_id, "lost_at": tr.ended_at, "last_seen": tr.last_seen,
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
        done = [tr for tr in self.tracker.tracks.values() if tr.resolution in ("known", "lost", "handed_off")]
        auto = sum(1 for tr in done if tr.resolution != "handed_off" and tr.rung < 4)
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
