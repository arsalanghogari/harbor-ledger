"""Rung state machine per track, evaluated every second. Rungs only go up on evidence and come
down only through reconcile() or an officer decision. Rungs 1 to 3 have no external effect:
they log, ping (a cooperative handshake) and draft. Only rung 5 releases anything, under a name."""
import math

from .registry import EXTEND_MIN
from .scoring import asset_eta, score
from .tracker import drift_cone

OPTIONS = ["Keep watching and re-ping", "Send patrol boat to observe", "Notify Coast Guard, keep channel open"]
HIDDEN_OPTION = "Close channel"  # behind a second click; needs a typed reason
UNMATCHED_S = 15     # rung 1 -> 2 after this long with no registry match
PING_TIMEOUT_S = 10  # silence declared this long after the first ping
ASK_P = 0.5          # rung 2 -> 3 posterior threshold (with silence)
HOLD_P = 0.8         # rung 3 -> 4 posterior threshold
HOLD_ETA_S = 120     # rung 3 -> 4 when an asset standoff zone is this close
HOLD_TIMEOUT_S = 90  # rung 4 with no click: re-push to duty supervisor


class Ladder:
    def __init__(self, ledger, registry, feed, assets, scoring_cfg, keepout_m=300):
        self.ledger, self.reg, self.feed, self.assets, self.cfg, self.keepout = ledger, registry, feed, assets, scoring_cfg, keepout_m
        self.short = {f[k]: f["short_" + k[5:]] for f in scoring_cfg["features"].values() for k in ("text_yes", "text_no")}
        self.pending = {}   # track_id -> (t_due, ping_id): replies in flight from the mock sensor
        self.seen = set()
        self.external = []  # every external action released, with who authorized it

    def log(self, t, tr, kind, detail):
        self.ledger.log(t, tr.id, kind, detail)

    # --- per-second evaluation -------------------------------------------------------
    def step(self, tr, t):
        if tr.rung == 5 or tr.ended_at is not None:
            return
        if tr.id not in self.seen:
            self.seen.add(tr.id)
            self.log(t, tr, "contact", f"new confirmed track at ({tr.x:.0f}, {tr.y:.0f}) depth {tr.depth:.0f} m")
        self.deliver_reply(tr, t)
        self.rescore(tr, t)
        if tr.rung == 1:
            if tr.matched_booking_id is None:
                b = self.reg.find_match(tr.x, tr.y, t)
                if b:
                    self.tag_known(tr, b, t, "registry match")
                elif t - tr.first_seen >= UNMATCHED_S:
                    self.enter(tr, 2, t, f"no registry match for {UNMATCHED_S} s")
        elif tr.rung == 2:
            if self.silent(tr, t) and tr.posterior > ASK_P:
                self.enter(tr, 3, t, f"silent after ping, posterior {tr.posterior:.2f}")
        elif tr.rung == 3:
            if tr.posterior > HOLD_P:
                self.enter(tr, 4, t, f"posterior {tr.posterior:.2f}")
            elif tr.eta and tr.eta[1] < HOLD_ETA_S:
                self.enter(tr, 4, t, f"reaches {tr.eta[0]} standoff zone in {tr.eta[1]} s")
        elif tr.rung == 4 and tr.escalated_at is None and t - tr.rung_since >= HOLD_TIMEOUT_S:
            tr.escalated_at = t
            self.log(t, tr, "escalated", f"no decision for {HOLD_TIMEOUT_S} s; card re-pushed to duty supervisor, no action taken")

    def silent(self, tr, t):
        return tr.ping_reply is None and tr.ping_sent_at is not None and t - tr.ping_sent_at >= PING_TIMEOUT_S

    def rescore(self, tr, t):
        tr.eta = asset_eta(tr.filter.predict_ahead(HOLD_ETA_S)[:HOLD_ETA_S], self.assets)
        if tr.ping_reply is not None:
            silent = False
        elif tr.ping_sent_at is not None and t - tr.ping_sent_at >= PING_TIMEOUT_S:
            silent = True
        else:
            silent = None  # no ping yet, or reply window still open
        if tr.filter.maneuvering > self.cfg["maneuver_threshold"]:
            tr.maneuver_since = t if tr.maneuver_since is None else tr.maneuver_since
        else:
            tr.maneuver_since = None
        tr.posterior, tr.features = score({
            "unmatched": tr.matched_booking_id is None,
            "silent": silent,
            "heading": tr.eta is not None,
            "maneuver": tr.maneuver_since is not None and t - tr.maneuver_since >= self.cfg["maneuver_hold_s"],
        }, self.cfg)

    # --- transitions and actions (no external effect below rung 5) ----------------------
    def enter(self, tr, rung, t, why):
        tr.rung, tr.rung_since = rung, t
        self.log(t, tr, "rung_change", f"rung {rung}: {why}")
        if rung == 2:
            self.send_ping(tr, t)
        elif rung == 3:
            hdg = math.degrees(math.atan2(tr.vy, tr.vx)) % 360
            tr.drafts = [
                f"Vessel advisory (draft, not sent): unidentified contact {tr.id} near ({tr.x:.0f}, {tr.y:.0f}); "
                f"vessels within {self.keepout} m keep clear",
                f"Patrol notice (draft, not sent): harbor patrol to observe contact {tr.id} at ({tr.x:.0f}, {tr.y:.0f}), "
                f"course {hdg:.0f} deg, {tr.speed:.1f} m/s",
            ]
            for d in tr.drafts:
                self.log(t, tr, "drafted", d)
            self.log(t, tr, "rung_change", f"watch tightened on {tr.id}: feed rate raised (stub), standoff ETA pre-computed")
            self.send_ping(tr, t)
        elif rung == 4:
            tr.card = self.build_card(tr)  # frozen at push: this is the card the officer saw
            self.log(t, tr, "rung_change", f"card pushed to officer: {tr.card['headline']}")

    def send_ping(self, tr, t):
        tr.pings += 1
        if tr.ping_sent_at is None:
            tr.ping_sent_at = t
        self.log(t, tr, "ping_sent", f"acoustic ID ping {tr.pings} sent (cooperative handshake)")
        reply = self.feed.ping(tr.x, tr.y, t)
        if reply:
            self.pending[tr.id] = (t + reply[1], reply[0])

    def deliver_reply(self, tr, t):
        due = self.pending.get(tr.id)
        if not due or t < due[0]:
            return
        del self.pending[tr.id]
        tr.ping_reply = due[1]
        b = self.reg.find_by_ping(due[1])
        self.log(t, tr, "ping_reply", f"reply ping_id={due[1]}" + ("" if b else " (not in registry)"))
        if b:
            self.reconcile(tr, b, t)

    def reconcile(self, tr, b, t) -> bool:
        """Valid ping reply. Inside box and window: match. Outside box (drifted): amend the box.
        Outside window (late): extend it once. All paths drop to rung 1 with no human."""
        if not b.covers(t):
            if not self.reg.extend_window(b, EXTEND_MIN, t, tr.id) or not b.covers(t):
                self.log(t, tr, "reconciled", f"BK-{b.id} window cannot be extended again; still unmatched")
                return False
        if not b.contains(tr.x, tr.y):
            self.reg.amend_box(b, tr.x, tr.y, t, tr.id)
        self.tag_known(tr, b, t, "ping reply reconciled with")
        return True

    def tag_known(self, tr, b, t, why):
        tr.matched_booking_id = b.id
        tr.resolution = tr.resolution or "known"
        self.pending.pop(tr.id, None)
        self.log(t, tr, "reconciled", f"tagged known: {why} BK-{b.id} ({b.operator}, {b.vehicle_id})")
        if tr.rung != 1:
            tr.rung, tr.rung_since = 1, t
            self.log(t, tr, "rung_change", "rung 1: known vehicle, automation stands down")

    def lose(self, tr, t):
        """Tracker deleted a confirmed track. Known vehicles and anything at rung 2+ become a lost
        record with a drift cone; an unmatched rung-1 track is just dropped."""
        if tr.matched_booking_id is None and tr.rung < 2:
            tr.resolution = tr.resolution or "dropped"
            return
        tr.resolution = tr.resolution or "lost"
        tr.cone = drift_cone(tr)
        x, y, vx, vy = tr.last_fix[0]
        hdg = math.degrees(math.atan2(vy, vx)) % 360
        c2, c5 = tr.cone[0], tr.cone[1]
        cone = (f"last fixed at ({x:.0f}, {y:.0f}) t={tr.last_seen:.0f}, drifting {math.hypot(vx, vy):.2f} m/s on {hdg:.0f} deg; "
                f"search {c2['radius']:.0f} m around ({c2['x']:.0f}, {c2['y']:.0f}) within 2 min, {c5['radius']:.0f} m within 5 min")
        if tr.matched_booking_id:
            b = self.reg.get(tr.matched_booking_id)
            tr.recovery = f"Send recovery here: {b.vehicle_id} (BK-{b.id}) {cone}"
        else:
            tr.recovery = f"Drift cone kept for patrol: {tr.id} {cone}"
        self.log(t, tr, "lost", f"no contact for 20 s at rung {tr.rung}; logged as lost. {tr.recovery}")

    def decide(self, tr, option, officer, note, t):
        if tr.rung != 4:
            raise ValueError(f"{tr.id} is at rung {tr.rung}, not waiting on a decision")
        if option not in OPTIONS + [HIDDEN_OPTION]:
            raise ValueError(f"unknown option {option!r}")
        if not officer.strip():
            raise ValueError("a decision needs the officer's name")
        if option == HIDDEN_OPTION and not note.strip():
            raise ValueError("closing the channel needs a typed reason")
        from .models import Decision
        self.ledger.decisions.append(Decision(len(self.ledger.decisions) + 1, tr.id, t, OPTIONS + [HIDDEN_OPTION], option, officer, note))
        tr.rung, tr.rung_since, tr.resolution, tr.decision = 5, t, "handed_off", option
        self.log(t, tr, "decision", f"{officer} chose: {option}" + (f" ({note})" if note else ""))
        if option == OPTIONS[0]:
            self.send_ping(tr, t)
        else:
            self.release(tr, t, option, officer)
        self.log(t, tr, "rung_change", "rung 5: handed off; automation stopped on this track")

    def release(self, tr, t, action, officer):
        """The only place anything leaves the system. Stub: logs it with the authorizing name."""
        self.external.append({"t": t, "track_id": tr.id, "action": action, "officer": officer})
        self.log(t, tr, "external_action", f"released (stub): {action}; authorized by {officer}")

    # --- officer card --------------------------------------------------------------------
    def headline(self, tr):
        where = f"{5 * round(tr.eta[1] / 5)} s from {tr.eta[0].lower()}" if tr.eta else "no asset on its track"
        ping = "no reply to ID ping" if tr.ping_reply is None else f"answered with unregistered ID {tr.ping_reply}"
        return f"Unknown vehicle, {where}, {ping}."

    def build_card(self, tr):
        return {"track_id": tr.id, "headline": self.headline(tr), "posterior": tr.posterior,
                "features": tr.features, "reasons": [self.short.get(f, f) for f in tr.features],
                "prepared": tr.drafts, "eta": tr.eta, "mode": tr.filter.mode,
                "speed": tr.speed, "depth": tr.depth, "pings": tr.pings, "options": OPTIONS,
                "hidden_option": HIDDEN_OPTION, "since": tr.rung_since}

    def card(self, tr):
        return {**tr.card, "escalated_at": tr.escalated_at, "x": tr.x, "y": tr.y}
