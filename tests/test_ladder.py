import pytest

from app.engine import Engine
from app.ladder import DRONE_OPTION, HIDDEN_OPTION, OPTIONS


def run(decide=True):
    """Returns the engine, the final snapshot and the four scenario tracks:
    A usc-glider-07 (nominal), B pin-rov-2 (drifts out of area), C sio-auv-3 (goes lost), E the unknown."""
    eng = Engine()
    snap = eng.run(officer=(lambda card: OPTIONS[2]) if decide else None)
    tracks = eng.tracker.tracks.values()
    by_booking = {tr.matched_booking_id: tr for tr in tracks if tr.matched_booking_id and tr.first_seen < 50}
    E = next(tr for tr in tracks if tr.confirmed and tr.first_seen >= 100)
    return eng, snap, (by_booking[1], by_booking[2], by_booking[3], E)


def test_harbor_night_arc():
    eng, snap, (A, B, C, E) = run()
    assert A.resolution == "known" and A.rung == 1 and eng.exception(A) == (None, 0)
    assert B.resolution == "known" and B.out_of_area and eng.exception(B) == ("out_of_area", 4)
    assert eng.reg.get(2).status != "amended" and not eng.reg.get(2).contains(B.x, B.y)  # area kept as registered
    assert C.resolution == "known" and C.ended_at is not None and C.cone and eng.halo_r(C) > 50
    assert E.rung == 5 and E.resolution == "handed_off" and E.matched_booking_id is None
    assert E.card["eta"][0] == "Fuel pier" and E.card["eta"][1] < 120 and "fuel pier" in E.card["headline"]
    assert eng.exception(E)[0] == "keepout_incursion" and [c["vehicle_id"] for c in eng.candidates(E)] == ["sio-auv-3"]
    assert snap["counters"] == {"auto": 3, "total": 4, "external_without_human": 0, "external_total": 1}
    assert snap["counts"] == {"registered": 2, "unknown": 1, "nominal": 1, "lost": 1}
    assert snap["lost"][0]["exception"] == "registered_lost" and snap["lost"][0]["vehicle_id"] == "sio-auv-3"
    assert eng.tasks_json() == []  # the patrol option publishes no drone tasking
    kinds = [e.kind for e in eng.ledger.events]
    assert kinds.index("external_action") > kinds.index("decision")
    assert eng.ledger.decisions[0].officer == "test officer"


def test_confirm_match_reacquires_the_lost_vehicle_and_quiets_the_map():
    eng, _, (A, B, C, E) = run(decide=False)
    assert E.rung == 4 and eng.candidates(E)[0]["booking_id"] == 3
    eng.match(E.id, 3, "confirm", "duty officer")
    assert E.matched_booking_id == 3 and E.rung == 1 and E.human and E.resolution == "known"
    assert C.resolution == "reacquired" and C.cone == [] and eng.snapshot()["lost"] == []
    assert eng.exception(E)[0] == "keepout_approach"  # still heading for the pier, now as a registered vehicle
    assert eng.booking_json(eng.reg.get(3))["recovery"] is None


def test_reject_escalate_and_ack():
    eng, _, (A, B, C, E) = run(decide=False)
    eng.match(E.id, 3, "reject", "duty officer")
    assert 3 in E.rejected and eng.candidates(E) == []
    eng.ack(B.id, "duty officer")
    assert B.acked
    with pytest.raises(ValueError):
        eng.contact_operator(E.id, "duty officer")  # unknowns have no operator
    eng.contact_operator(B.id, "duty officer")
    assert "duty officer" in eng.reg.get(2).note
    eng2 = Engine(); eng2.running = True
    while eng2.t < 130:
        eng2.tick()
    unknown = next(tr for tr in eng2.tracker.confirmed() if not tr.matched_booking_id)
    assert unknown.rung == 3
    eng2.escalate(unknown.id, "duty officer")
    assert unknown.rung == 4 and unknown.card is not None


def test_e_reaches_rung_4_and_escalates_without_a_click():
    eng, _, (*_, E) = run(decide=False)
    assert E.rung == 4 and E.escalated_at is None
    for t in range(int(E.rung_since), int(E.rung_since) + 95):  # ladder only: the card sits unanswered
        eng.ladder.step(E, t)
    assert E.escalated_at == E.rung_since + 90
    assert eng.ladder.external == []


def test_decision_rules():
    eng, _, (A, B, C, E) = run(decide=False)
    with pytest.raises(ValueError):
        eng.decide(A.id, OPTIONS[0], "officer")  # not at rung 4
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


def test_drone_tasking_is_published_on_release_and_takes_reports():
    eng, _, (*_, E) = run(decide=False)
    eng.decide(E.id, DRONE_OPTION, "duty officer")
    tasks = eng.tasks_json()
    assert len(tasks) == 1 and tasks[0]["track_id"] == E.id and tasks[0]["authorized_by"] == "duty officer"
    tgt = tasks[0]["target"]
    assert 33.6 < tgt["lat"] < 33.8 and -118.4 < tgt["lon"] < -118.1 and tgt["live"] and 0 <= tgt["course_deg"] < 360
    assert [p["s"] for p in tasks[0]["predicted"]] == [30, 60, 120]
    assert "DT-1" in eng.ladder.external[0]["action"] and eng.ladder.external[0]["officer"] == "duty officer"
    eng.report("DT-1", "on_station", "visual on target")
    assert eng.tasks_json()[0]["status"] == "on_station" and eng.ledger.events[-1].kind == "drone_report"
    eng.report("DT-1", "complete", "photos taken", "https://example.test/photo.jpg")
    assert eng.tasks_json() == [] and eng.tasks_json(include_done=True)[0]["status"] == "complete"
