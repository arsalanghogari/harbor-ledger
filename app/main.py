"""FastAPI app. The scenario ticks in a background task; the front end polls GET /snapshot at 1 Hz.
POST /decision, POST /booking, POST /control. The single page is served at /, /watch, /register, /control."""
import asyncio
import hmac
import os
import secrets
from dataclasses import asdict
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .auth import COOKIE, SESSION_S, current_operator, issue, load_users, parse, verify
from .engine import ROOT, Engine

engine = Engine()
engine.running = True  # the demo starts on boot; /control reset + start for a clean recording

# Platforms (the observation drone) authenticate with a bearer token, humans with their session.
DRONE_TOKEN = os.environ.get("HARBOR_DRONE_TOKEN") or secrets.token_urlsafe(24)
if "HARBOR_DRONE_TOKEN" not in os.environ:
    print(f"HARBOR_DRONE_TOKEN not set; drone token for this run: {DRONE_TOKEN}", flush=True)


def drone_or_authority(authorization: str | None = Header(default=None), harbor_session: str | None = Cookie(default=None)):
    if authorization and hmac.compare_digest(authorization.removeprefix("Bearer ").strip(), DRONE_TOKEN):
        return {"username": "drone", "operator": "observation drone", "role": "device"}
    username = parse(harbor_session) if harbor_session else None
    user = load_users().get(username) if username else None
    if user and user.get("role") == "authority":
        return {"username": username, "operator": user["operator"], "role": "authority"}
    raise HTTPException(401, "drone bearer token or harbor authority session required")


async def loop():
    while True:
        engine.tick()
        await asyncio.sleep(1 / engine.speed)


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(loop())
    yield
    task.cancel()


app = FastAPI(title="Harbor Ledger", lifespan=lifespan)
app.mount("/config", StaticFiles(directory=os.path.join(ROOT, "config")), name="config")
INDEX = os.path.join(ROOT, "static", "index.html")

Name = Field(min_length=1, max_length=80)
Coord = Field(ge=-20000, le=20000)
Seconds = Field(ge=-7 * 86400, le=7 * 86400)


class DecisionIn(BaseModel):
    track_id: str = Field(max_length=20)
    option: str = Field(max_length=80)
    note: str = Field(default="", max_length=500)


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=200)
    role: Literal["authority", "operator"]  # which door: officer and operator sign-ins are separate


class BookingIn(BaseModel):
    vehicle_id: str = Name
    ping_id: str = Name
    x_min: float = Coord
    y_min: float = Coord
    x_max: float = Coord
    y_max: float = Coord
    window_start: float = Seconds
    window_end: float = Seconds


class ReviewIn(BaseModel):
    action: Literal["approve", "reject"]


class MatchIn(BaseModel):
    booking_id: int
    action: Literal["confirm", "reject"]


def authority(user: dict = Depends(current_operator)):
    if user["role"] != "authority":
        raise HTTPException(403, "harbor authority only")
    return user


def operator(user: dict = Depends(current_operator)):
    if user["role"] != "operator":
        raise HTTPException(403, "registered operators only")
    return user


class ReportIn(BaseModel):
    status: Literal["acknowledged", "en_route", "on_station", "complete", "cancelled"]
    note: str = Field(default="", max_length=500)
    media_url: str = Field(default="", max_length=500)


class ControlIn(BaseModel):
    action: Literal["start", "pause", "reset"] | None = None
    speed: float | None = Field(default=None, gt=0, le=10)


@app.get("/snapshot")
async def snapshot(user: dict = Depends(current_operator)):
    return engine.snapshot() if user["role"] == "authority" else engine.operator_snapshot(user["operator"])


@app.get("/track/{track_id}/events")
async def track_events(track_id: str, user: dict = Depends(authority)):
    """That track's own audit trail, oldest first."""
    return [asdict(e) for e in engine.ledger.events if e.track_id == track_id[:20]]


@app.post("/track/{track_id}/match")
async def track_match(track_id: str, m: MatchIn, user: dict = Depends(authority)):
    """Officer confirms or rejects a candidate registry match for an unknown contact."""
    try:
        engine.match(track_id[:20], m.booking_id, m.action, user["operator"])
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@app.post("/track/{track_id}/{action}")
async def track_action(track_id: str, action: Literal["escalate", "ack", "contact_operator"], user: dict = Depends(authority)):
    try:
        getattr(engine, action)(track_id[:20], user["operator"])
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@app.post("/decision")
async def decision(d: DecisionIn, user: dict = Depends(authority)):
    try:
        engine.decide(d.track_id, d.option, user["operator"], d.note)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@app.post("/login")
async def login(body: LoginIn, response: Response):
    user = verify(body.username, body.password)
    if not user:
        raise HTTPException(401, "wrong username or password")
    if user["role"] != body.role:
        raise HTTPException(403, "this is a harbor officer account; use officer sign-in" if user["role"] == "authority"
                            else "this is an operator account; use operator sign-in")
    response.set_cookie(COOKIE, issue(body.username), max_age=SESSION_S, httponly=True, samesite="lax")
    engine.ledger.log(engine.t, None, "login", f"{user['operator']} signed in as {body.username}")
    return user


@app.post("/logout")
async def logout(response: Response):
    response.delete_cookie(COOKIE)
    return {"ok": True}


@app.get("/me")
async def me(user: dict = Depends(current_operator)):
    return user


@app.post("/booking")
async def booking(b: BookingIn, user: dict = Depends(operator)):
    """An operator requests a slot for itself; it stays pending until an officer reviews it."""
    made, why = engine.book(operator=user["operator"], **b.model_dump())
    return {"ok": made is not None, "conflict": why, "booking": engine.booking_json(made) if made else None}


@app.post("/booking/{booking_id}/review")
async def booking_review(booking_id: int, r: ReviewIn, user: dict = Depends(authority)):
    try:
        b = engine.review(booking_id, r.action == "approve", user["operator"])
    except ValueError as e:
        raise HTTPException(409, str(e))
    return engine.booking_json(b)


@app.get("/tasks")
async def tasks(all: bool = False, who: dict = Depends(drone_or_authority)):
    """Published taskings for external platforms; each is refreshed from the live track."""
    return engine.tasks_json(include_done=all)


@app.get("/tasks/{task_id}")
async def task(task_id: str, who: dict = Depends(drone_or_authority)):
    found = next((t for t in engine.tasks_json(include_done=True) if t["id"] == task_id[:12]), None)
    if found is None:
        raise HTTPException(404, f"no tasking {task_id}")
    return found


@app.post("/tasks/{task_id}/report")
async def task_report(task_id: str, r: ReportIn, who: dict = Depends(drone_or_authority)):
    try:
        engine.report(task_id[:12], r.status, r.note, r.media_url)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"ok": True}


@app.post("/control")
async def control(c: ControlIn, user: dict = Depends(authority)):
    engine.control(c.action, c.speed)
    return {"t": engine.t, "running": engine.running, "speed": engine.speed}


@app.get("/")
@app.get("/watch")
@app.get("/register")
@app.get("/control")
async def index():
    return FileResponse(INDEX)
