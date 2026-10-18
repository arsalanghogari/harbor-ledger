"""Role separation: officers watch contacts and review bookings; operators only request their own slots."""
import asyncio

import pytest
from fastapi import HTTPException

from app import main
from app.main import BookingIn, ReviewIn, engine

OFFICER = {"username": "officer", "operator": "Port of LA Watch Officer", "role": "authority"}
OP = {"username": "usc", "operator": "USC AUV Lab", "role": "operator"}
BOX = dict(vehicle_id="test-auv", ping_id="TEST-AUV", x_min=5000, y_min=5000, x_max=5100, y_max=5100, window_start=0, window_end=3600)


def guards():
    """(path, method) -> names of the dependencies FastAPI runs before the handler."""
    return {(r.path, m): {d.call.__name__ for d in r.dependant.dependencies}
            for r in main.app.routes if hasattr(r, "dependant") for m in r.methods}


def test_routes_are_role_gated():
    g = guards()
    for key in [("/decision", "POST"), ("/control", "POST"), ("/booking/{booking_id}/review", "POST"),
                ("/track/{track_id}/events", "GET"), ("/track/{track_id}/match", "POST"), ("/track/{track_id}/{action}", "POST")]:
        assert "authority" in g[key], key
    assert "operator" in g[("/booking", "POST")]
    with pytest.raises(HTTPException):
        main.authority(OP)
    with pytest.raises(HTTPException):
        main.operator(OFFICER)


def test_operator_booking_needs_officer_approval():
    run = asyncio.run
    b = run(main.booking(BookingIn(**BOX), user=OP))["booking"]
    assert b["status"] == "pending" and engine.reg.find_match(5050, 5050, 10) is None

    snap = run(main.snapshot(user=OP))
    assert "tracks" not in snap and {x["operator"] for x in snap["bookings"]} == {OP["operator"]}

    assert run(main.booking_review(b["id"], ReviewIn(action="approve"), user=OFFICER))["status"] == "scheduled"
    assert engine.reg.find_match(5050, 5050, 10).id == b["id"]
    with pytest.raises(HTTPException):  # reviewed once only
        run(main.booking_review(b["id"], ReviewIn(action="reject"), user=OFFICER))
