"""Dataclasses for the five ledger tables. All state lives in memory (the Ledger);
the event log is dumped to JSON at the end of a run."""
import json
from dataclasses import asdict, dataclass, field


@dataclass
class Booking:
    id: int | None
    operator: str
    vehicle_id: str
    ping_id: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    window_start: float  # scenario seconds
    window_end: float
    status: str = "scheduled"  # scheduled, active, amended (window extended), closed
    extended: int = 0          # window extensions used (max 1)
    note: str = ""             # reason when amended by the system

    def contains(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def covers(self, t):
        return self.window_start <= t <= self.window_end


@dataclass
class Contact:
    id: str
    t: float
    x: float
    y: float
    depth: float
    speed_est: float | None = None
    source: str = "sonar"


@dataclass
class Track:
    id: str
    first_seen: float
    last_seen: float
    filter: object                     # imm.IMMTrack: combined state, covariance, mode probabilities
    hits: int = 1
    scans: int = 1
    score: float = 1.0                 # existence log-likelihood ratio (unit steps)
    confirmed: bool = False
    last_fix: tuple | None = None      # (state, covariance) at the last hit; seed of the drift cone
    rung: int = 1
    rung_since: float = 0.0
    posterior: float = 0.1
    features: list = field(default_factory=list)   # two strongest features, plain words
    matched_booking_id: int | None = None
    ping_sent_at: float | None = None  # first ping; later pings only count
    pings: int = 0
    ping_reply: str | None = None
    eta: tuple | None = None           # (asset name, seconds) if entering a standoff zone
    maneuver_since: float | None = None  # when non-quiet mode probability first exceeded the threshold
    drafts: list = field(default_factory=list)      # prepared, not sent (rung 3)
    escalated_at: float | None = None  # rung 4 timeout re-push to supervisor
    resolution: str | None = None      # known, lost, dropped, handed_off
    ended_at: float | None = None
    cone: list = field(default_factory=list)        # drift cone once lost
    recovery: str = ""                              # recovery / patrol line once lost
    card: dict | None = None                        # the Hold card as pushed (rung 4)
    decision: str | None = None                     # the option the officer chose (rung 5)
    out_of_area: bool = False                       # identified vehicle outside its booked area

    @property
    def state(self): return self.filter.x
    @property
    def covariance(self): return self.filter.P
    @property
    def x(self): return float(self.filter.x[0])
    @property
    def y(self): return float(self.filter.x[1])
    @property
    def vx(self): return float(self.filter.x[2])
    @property
    def vy(self): return float(self.filter.x[3])
    @property
    def speed(self): return float((self.vx ** 2 + self.vy ** 2) ** 0.5)
    @property
    def depth(self): return float(self.filter.depth)


@dataclass
class Decision:
    id: int
    track_id: str
    t: float
    options_offered: list
    option_chosen: str
    officer: str
    note: str = ""


@dataclass
class Event:
    id: int
    t: float
    track_id: str | None
    kind: str  # contact, rung_change, ping_sent, ping_reply, reconciled, drafted, escalated, decision, external_action, lost
    detail: str


class Ledger:
    """The five tables as lists and dicts."""

    def __init__(self):
        self.bookings: list[Booking] = []
        self.tracks: dict[str, Track] = {}
        self.decisions: list[Decision] = []
        self.events: list[Event] = []

    def log(self, t, track_id, kind, detail) -> Event:
        e = Event(len(self.events) + 1, t, track_id, kind, detail)
        self.events.append(e)
        return e

    def recent(self, n=20, track_id=None):
        evs = [e for e in self.events if track_id is None or e.track_id == track_id]
        return evs[-n:][::-1]

    def dump(self, path):
        with open(path, "w") as f:
            json.dump({"events": [asdict(e) for e in self.events],
                       "decisions": [asdict(d) for d in self.decisions],
                       "bookings": [asdict(b) for b in self.bookings]}, f, indent=1)
