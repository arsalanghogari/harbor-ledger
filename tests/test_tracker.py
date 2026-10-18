import numpy as np

from app.models import Contact
from app.tracker import CONFIRM_HITS, Tracker, drift_cone


def test_two_crossing_contacts_keep_their_ids():
    rng = np.random.default_rng(3)
    tr = Tracker()
    ids = {}
    for t in range(60):  # eastbound along y=0 and northbound along x=0 cross at the origin at t=30
        a = Contact("a", t, -45 + 1.5 * t + rng.normal(0, 0.5), rng.normal(0, 0.5), 5)
        b = Contact("b", t, rng.normal(0, 0.5), -45 + 1.5 * t + rng.normal(0, 0.5), 5)
        tr.step([a, b], t)
        if t == 15:
            ids = {("east" if x.vx > 1 else "north"): x.id for x in tr.confirmed()}
            assert set(ids) == {"east", "north"}
    conf = {x.id: x for x in tr.confirmed()}
    assert len(conf) == 2
    assert conf[ids["east"]].vx > 1.0 and abs(conf[ids["east"]].vy) < 0.4
    assert conf[ids["north"]].vy > 1.0 and abs(conf[ids["north"]].vx) < 0.4


def test_three_of_five_confirmation_then_deletion_after_20_misses():
    tr = Tracker()
    for t in range(5):
        tr.step([Contact("a", t, 1.0 * t, 0.0, 5)], t)
        assert bool(tr.confirmed()) == (t + 1 >= CONFIRM_HITS)
    assert tr.confirmed()[0].id == "T-1"
    deleted = []
    for t in range(5, 40):
        deleted += tr.step([], t)
        if deleted:
            break
    assert deleted and 19 <= t - 4 <= 21


def test_gated_out_return_does_not_become_a_track():
    tr = Tracker()
    for t in range(10):
        tr.step([Contact("a", t, 1.0 * t, 0.0, 5)], t)
    tr.step([Contact("a", 10, 10.0, 0.0, 5), Contact("glitch", 10, 10.0, 40.0, 5)], 10)
    for t in range(11, 16):
        tr.step([Contact("a", t, 1.0 * t, 0.0, 5)], t)
    assert [x.id for x in tr.confirmed()] == ["T-1"]
    assert all(x.resolution == "dropped" for x in tr.tracks.values() if not x.confirmed)


def test_drift_cone_grows_with_horizon():
    tr = Tracker()
    for t in range(10):
        tr.step([Contact("a", t, 0.5 * t, 0.0, 5)], t)
    cone = drift_cone(tr.confirmed()[0])
    assert [c["minutes"] for c in cone] == [2, 5, 10]
    assert cone[0]["radius"] < cone[1]["radius"] < cone[2]["radius"]
    assert cone[2]["x"] > cone[0]["x"]  # drifts along the last velocity
