import pytest

from app.engine import Engine
from app.ladder import HIDDEN_OPTION, OPTIONS


def run(decide=True):
    eng = Engine()
    snap = eng.run(officer=(lambda card: OPTIONS[1]) if decide else None)
    by_birth = {int(tr.first_seen): tr for tr in eng.tracker.tracks.values() if tr.confirmed}
    return eng, snap, [by_birth[t] for t in (0, 10, 30, 60, 90)]


def test_harbor_night_arc():
    eng, snap, (A, B, C, D, E) = run()
    assert A.resolution == "known" and A.matched_booking_id == 1 and A.rung == 1
    assert B.matched_booking_id == 2 and B.rung == 1 and B.out_of_area
    assert eng.reg.get(2).status != "amended" and not eng.reg.get(2).contains(B.x, B.y)  # area kept as registered
    assert "outside the booked area" in eng.reg.get(2).note
    assert C.matched_booking_id == 3 and eng.reg.get(3).extended == 1 and C.resolution == "known"
    assert C.cone and C.recovery.startswith("Send recovery here")
    assert D.rung == 3 and D.resolution == "lost" and D.cone and D.ended_at < 150
    assert E.rung == 5 and E.resolution == "handed_off"
    assert E.card["eta"][0] == "Fuel pier" and E.card["eta"][1] < 120 and "fuel pier" in E.card["headline"]
    assert snap["counters"] == {"auto": 4, "total": 5, "external_without_human": 0, "external_total": 1}
    kinds = [e.kind for e in eng.ledger.events]
    assert kinds.index("external_action") > kinds.index("decision")
    assert eng.ledger.decisions[0].officer == "test officer"


def test_e_reaches_rung_4_and_escalates_without_a_click():
    eng, _, (*_, E) = run(decide=False)
    assert E.rung == 4 and E.escalated_at is None
    while eng.t < E.rung_since + 95:  # keep stepping past the scenario; the feed is silent now
        eng.step()
    assert E.escalated_at == E.rung_since + 90
    assert eng.ladder.external == []


def test_decision_rules():
    eng, _, (A, B, C, D, E) = run(decide=False)
    with pytest.raises(ValueError):
        eng.decide(D.id if D.ended_at is None else A.id, OPTIONS[0], "officer")  # not at rung 4
    with pytest.raises(ValueError):
        eng.decide(E.id, HIDDEN_OPTION, "officer", note="")  # hidden option needs a reason
    with pytest.raises(ValueError):
        eng.decide(E.id, OPTIONS[1], "   ")  # a name is required
    eng.decide(E.id, HIDDEN_OPTION, "duty officer", note="traffic cleared, pier exposed")
    assert E.rung == 5 and eng.ladder.external[0]["officer"] == "duty officer"


def test_registry_overlap_and_keepout_no_go():
    eng, _, (*_, E) = run()
    common = dict(operator="Lab", vehicle_id="v", ping_id="P", window_start=eng.t, window_end=eng.t + 3600)
    made, why = eng.book(x_min=-300, y_min=-400, x_max=-200, y_max=-300, **common)  # inside BK-1's box and window
    assert made is None and why.startswith("booking BK-1")
    made, why = eng.book(x_min=E.x - 50, y_min=E.y - 50, x_max=E.x + 50, y_max=E.y + 50, **common)
    assert made is None and why == f"keep-out zone around contact {E.id}"
    made, why = eng.book(x_min=-900, y_min=-900, x_max=-800, y_max=-800, **common)
    assert made is not None and why is None and made.id == 4
