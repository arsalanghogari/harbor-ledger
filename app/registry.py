"""Bookings: create with overlap check, find_match, amend_box, extend_window.
Every change the system makes writes an Event."""
from .models import Booking

MARGIN_M = 20       # slack added around a found position when a box is amended
EXTEND_MIN = 30     # one-time window extension for late vehicles


def boxes_overlap(a: Booking, b: Booking):
    return a.x_min <= b.x_max and b.x_min <= a.x_max and a.y_min <= b.y_max and b.y_min <= a.y_max


def windows_overlap(a: Booking, b: Booking):
    return a.window_start <= b.window_end and b.window_start <= a.window_end


def box_hits_circle(b: Booking, x, y, r):
    cx, cy = min(max(x, b.x_min), b.x_max), min(max(y, b.y_min), b.y_max)
    return (cx - x) ** 2 + (cy - y) ** 2 <= r * r


class Registry:
    def __init__(self, ledger):
        self.ledger = ledger

    def all(self):
        return list(self.ledger.bookings)

    def get(self, bid):
        return next((b for b in self.ledger.bookings if b.id == bid), None)

    def add(self, b: Booking) -> Booking:
        b.id = len(self.ledger.bookings) + 1
        self.ledger.bookings.append(b)
        return b

    def conflict(self, b: Booking, t, keepouts=()):
        """Name of the first conflicting booking or live keep-out zone, else None.
        keepouts: iterable of (x, y, radius, track_id) for tracks at rung 3+."""
        for o in self.ledger.bookings:
            if o.id != b.id and o.status != "closed" and boxes_overlap(b, o) and windows_overlap(b, o):
                return f"booking BK-{o.id} ({o.operator}, {o.vehicle_id})"
        for x, y, r, tid in keepouts:
            if b.covers(t) and box_hits_circle(b, x, y, r):
                return f"keep-out zone around contact {tid}"
        return None

    def create(self, b: Booking, t, keepouts=()):
        """Operator portal entry point. Returns (booking, None) on go, (None, reason) on no-go."""
        if b.x_min >= b.x_max or b.y_min >= b.y_max or b.window_start >= b.window_end:
            return None, "box or window is empty"
        why = self.conflict(b, t, keepouts)
        if why:
            return None, why
        return self.add(b), None

    def find_match(self, x, y, t):
        return next((b for b in self.ledger.bookings if b.status != "closed" and b.contains(x, y) and b.covers(t)), None)

    def find_by_ping(self, ping_id):
        return next((b for b in self.ledger.bookings if b.ping_id == ping_id and b.status != "closed"), None)

    def amend_box(self, b: Booking, x, y, t, track_id):
        """Grow the box to include (x, y) plus margin; status -> amended; log reconciled."""
        dx = max(b.x_min - x, x - b.x_max, 0)
        dy = max(b.y_min - y, y - b.y_max, 0)
        b.x_min, b.x_max = min(b.x_min, x - MARGIN_M), max(b.x_max, x + MARGIN_M)
        b.y_min, b.y_max = min(b.y_min, y - MARGIN_M), max(b.y_max, y + MARGIN_M)
        b.status = "amended"
        b.note = f"vehicle answered ping {round((dx * dx + dy * dy) ** 0.5)} m outside booked box; box extended by system"
        self.ledger.log(t, track_id, "reconciled", f"BK-{b.id} {b.vehicle_id}: {b.note}")
        return b

    def extend_window(self, b: Booking, minutes, t, track_id) -> bool:
        """Extend window_end once. Returns False if the one extension is already used."""
        if b.extended:
            return False
        late = round((t - b.window_end) / 60)
        b.window_end += minutes * 60
        b.extended, b.status = 1, "amended"
        b.note = f"vehicle answered ping {late} min after window; extended {minutes} min by system"
        self.ledger.log(t, track_id, "reconciled", f"BK-{b.id} {b.vehicle_id}: {b.note}")
        return True
