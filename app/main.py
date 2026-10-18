"""FastAPI app. The scenario ticks in a background task; the front end polls GET /snapshot at 1 Hz.
POST /decision, POST /booking, POST /control. The single page is served at /, /watch, /register, /control."""
import asyncio
import os
from dataclasses import asdict
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .auth import COOKIE, SESSION_S, current_operator, issue, verify
from .engine import ROOT, Engine

engine = Engine()
engine.running = True  # the demo starts on boot; /control reset + start for a clean recording


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


class BookingIn(BaseModel):
    vehicle_id: str = Name
    ping_id: str = Name
    x_min: float = Coord
    y_min: float = Coord
    x_max: float = Coord
    y_max: float = Coord
    window_start: float = Seconds
    window_end: float = Seconds


class ControlIn(BaseModel):
    action: Literal["start", "pause", "reset"] | None = None
    speed: float | None = Field(default=None, gt=0, le=10)


@app.get("/snapshot")
async def snapshot(user: dict = Depends(current_operator)):
    return engine.snapshot()


@app.get("/track/{track_id}/events")
async def track_events(track_id: str, user: dict = Depends(current_operator)):
    """That track's own audit trail, oldest first."""
    return [asdict(e) for e in engine.ledger.events if e.track_id == track_id[:20]]


@app.post("/decision")
async def decision(d: DecisionIn, user: dict = Depends(current_operator)):
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
async def booking(b: BookingIn, user: dict = Depends(current_operator)):
    made, why = engine.book(operator=user["operator"], **b.model_dump())
    return {"ok": made is not None, "conflict": why, "booking": engine.booking_json(made) if made else None}


@app.post("/control")
async def control(c: ControlIn, user: dict = Depends(current_operator)):
    engine.control(c.action, c.speed)
    return {"t": engine.t, "running": engine.running, "speed": engine.speed}


@app.get("/")
@app.get("/watch")
@app.get("/register")
@app.get("/control")
async def index():
    return FileResponse(INDEX)
