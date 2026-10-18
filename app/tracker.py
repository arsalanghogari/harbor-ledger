"""Multi-target tracker over IMMTrack: chi-square gating, global (Hungarian) assignment, an
LLR-style existence score with 3-of-5 confirmation and deletion after ~20 misses, and a
drift cone for deleted tracks."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .imm import MODELS, IMMTrack
from .models import Contact, Track

GATE_CHI2 = 9.21  # 2 dof, 99%
# ponytail: unit-step LLR (hit +1, miss -1, capped). A full-score track dies after 20 misses.
HIT, MISS, SCORE_CAP, DELETE_BELOW = 1.0, 1.0, 5.0, -15.0
CONFIRM_HITS, CONFIRM_SCANS = 3, 5
BIG = 1e9


def assign(tracks, contacts):
    """Global assignment on squared Mahalanobis cost inside the gate.
    Returns ({track index: contact index}, [unassigned contact indexes])."""
    if not tracks or not contacts:
        return {}, list(range(len(contacts)))
    cost = np.full((len(tracks), len(contacts)), BIG)
    for i, tr in enumerate(tracks):
        for j, c in enumerate(contacts):
            d2 = tr.filter.mahalanobis2([c.x, c.y])
            if d2 <= GATE_CHI2:
                cost[i, j] = d2
    rows, cols = linear_sum_assignment(cost)
    pairs = {int(i): int(j) for i, j in zip(rows, cols) if cost[i, j] < BIG}
    return pairs, [j for j in range(len(contacts)) if j not in pairs.values()]


def drift_cone(tr: Track, minutes=(2, 5, 10)):
    """Last fix grown forward under CV-quiet: [{minutes, x, y, radius}] with radius = 1 sigma."""
    x, P = tr.last_fix
    cv = MODELS[0]
    out = []
    for s in range(1, max(minutes) * 60 + 1):
        x, P = cv.predict(x, P)
        if s % 60 == 0 and s // 60 in minutes:
            out.append({"minutes": s // 60, "x": float(x[0]), "y": float(x[1]),
                        "radius": float(np.sqrt(np.linalg.eigvalsh(P[:2, :2]).max()))})
    return out


class Tracker:
    def __init__(self):
        self.tracks: dict[str, Track] = {}
        self.n = 0   # confirmed tracks, for T-n ids
        self.k = 0   # tentative births

    def live(self):
        return [tr for tr in self.tracks.values() if tr.ended_at is None]

    def confirmed(self):
        return [tr for tr in self.live() if tr.confirmed]

    def _hit(self, tr: Track, c: Contact, t):
        tr.filter.update(np.array([c.x, c.y]), c.depth)
        tr.hits += 1
        tr.scans += 1
        tr.score = min(tr.score + HIT, SCORE_CAP)
        tr.last_seen = t
        tr.last_fix = (tr.filter.x.copy(), tr.filter.P.copy())

    def step(self, contacts, t) -> list[Track]:
        """One scan. Confirmed tracks are assigned first, tentative ones take the leftovers,
        leftover contacts become tentative tracks. Returns confirmed tracks deleted this scan."""
        live = self.live()
        for tr in live:
            tr.filter.predict()
        conf = [tr for tr in live if tr.confirmed]
        tent = [tr for tr in live if not tr.confirmed]
        hit = set()
        pairs, rest = assign(conf, contacts)
        for i, j in pairs.items():
            self._hit(conf[i], contacts[j], t)
            hit.add(conf[i].id)
        rest = [contacts[j] for j in rest]
        pairs, unassigned = assign(tent, rest)
        for i, j in pairs.items():
            self._hit(tent[i], rest[j], t)
            hit.add(tent[i].id)
        for j in unassigned:
            c = rest[j]
            self.k += 1
            tr = Track(id=f"tent-{self.k}", first_seen=t, last_seen=t, filter=IMMTrack([c.x, c.y], c.depth))
            tr.last_fix = (tr.filter.x.copy(), tr.filter.P.copy())
            self.tracks[tr.id] = tr
        deleted = []
        for tr in live:
            if tr.id not in hit:
                tr.score -= MISS
                tr.scans += 1
            if tr.confirmed:
                if tr.score < DELETE_BELOW:
                    tr.ended_at = t
                    deleted.append(tr)
            elif tr.hits >= CONFIRM_HITS:
                tr.confirmed = True
                self.n += 1
                self.tracks[f"T-{self.n}"] = self.tracks.pop(tr.id)
                tr.id = f"T-{self.n}"
            elif tr.scans >= CONFIRM_SCANS or tr.score < -1:
                tr.ended_at, tr.resolution = t, "dropped"
        return deleted
