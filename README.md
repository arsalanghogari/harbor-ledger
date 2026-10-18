# Harbor Ledger

Registry-first triage for unidentified autonomous underwater vehicles in a working harbor.
Every sonar contact is checked against a ledger of registered dives before anything escalates:
known vehicles become non-events, unknown ones climb a five-rung ladder, and the watch officer
only sees the contacts that need a human decision. Nothing with an external effect leaves the
system without a named human's click.

Live: https://harbor-ledger.onrender.com (Render free tier; the first request after idle takes about a minute to wake).

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app
```

Open http://localhost:8000. The Harbor Night scenario starts on boot.

| Route | What |
|---|---|
| `/` | Start page: what the system does, and sign-in. Every other route needs a session |
| `/tasks` | Drone taskings published at rung 5, bearer token or authority session (see Drone tasking) |
| `/watch` | Officer console: map, Hold queue, live status board with per-track history |
| `/register` | Operator portal: book a dive, instant go / no-go, booking list with out-of-area flags, window extensions and recovery lines |
| `/control` | Start, pause, reset, 1x / 2x (hidden; for recording) |
| `GET /snapshot` | Full state once a second; the page just renders it |

Sign-in: the start page signs in harbor staff and operators; decisions are logged under the signed-in name and bookings are stamped with it. `python -m app.auth seed` writes three demo accounts
(passwords land in `config/DEMO_ACCOUNTS.md`); `python -m app.auth add <username> "<Operator>"` adds one.
Sessions are HMAC-signed cookies; set `HARBOR_SECRET` to keep them across restarts.

Headless replay with the event log printed and dumped to `scenarios/harbor_night.events.json`:

```bash
.venv/bin/python -m app.engine
```

Tests:

```bash
.venv/bin/python -m pytest -q
```

## The ladder

| Rung | Enter when | Autonomous action (no external effect) | Exit |
|---|---|---|---|
| 1 Watch | New confirmed track | Log, run IMM, check registry | Match: stays 1, tagged known. No match 15 s: rung 2 |
| 2 Ask | Unmatched 15 s | Acoustic ID ping (cooperative handshake), 10 s timer | Valid reply: reconcile. Silence and posterior > 0.5: rung 3 |
| 3 Slow | Posterior > 0.5 | Tighten the watch, second ping, draft the vessel advisory and patrol notice but send nothing, pre-compute standoff ETA | Reply: reconcile. Posterior > 0.8 or asset ETA < 120 s: rung 4 |
| 4 Hold | Posterior > 0.8 or ETA < 120 s | Push the card to the officer. 90 s with no click: re-push to the duty supervisor, still no action | Officer clicks: rung 5 |
| 5 Hand off | Officer decision | Log the Decision under the officer's name, release the chosen drafted action (stub), stop automation | Terminal |

Reconciliation on a valid ping reply is autonomous: inside the booked box and window it is matched;
late, the window is extended once by 30 min. The booked area itself is never amended by the system:
an identified vehicle outside it is flagged on the console and the portal, because being outside the
authorized area is itself a signal.

## Harbor Night

Three contacts over four minutes. Two resolve with no human touch, one reaches the officer.

| Contact | Truth | What happens |
|---|---|---|
| A | USC glider, registered, inside box | Registry match at t+2 s, tagged known |
| B | Inspection ROV, 80 m outside its box | Rung 2 at t+15, answers the ping, identified and back to rung 1, flagged out of area; box unchanged |
| E | Unknown AUV, 1.5 m/s toward the fuel pier, one 30 degree turn | Rung 1 to 4, card: "Unknown vehicle, 105 s from fuel pier, no reply to ID ping." One click |

Counters end at **Resolved without a human: 2 of 3** and **External actions without a human: 0**.
Then on `/register`, a dive booked over E's keep-out circle gets an instant no-go naming the contact.

## Drone tasking

When the officer picks **Task observation drone**, the system publishes a tasking for that one contact.
A drone, or any platform, polls it with a bearer token and reports back; nothing is published without
the officer's release, and the tasking is logged as an external action under their name.

```bash
curl -H "Authorization: Bearer $HARBOR_DRONE_TOKEN" https://harbor-ledger.onrender.com/tasks
```

Each tasking carries the live target (`lat`, `lon`, harbor-local `x_m`/`y_m`, `depth_m`, `course_deg`
as a compass bearing, `speed_mps`), predicted positions at 30, 60 and 120 s, the asset ETA, the standoff
radius and the objective (`observe`, `photograph`). `GET /tasks/DT-1` refreshes it from the tracker on
every call, so the drone follows the estimate, not a stale fix. The drone reports with

```bash
curl -X POST -H "Authorization: Bearer $HARBOR_DRONE_TOKEN" -H "Content-Type: application/json" \
  -d '{"status": "on_station", "note": "visual on target", "media_url": "https://.../photo.jpg"}' \
  https://harbor-ledger.onrender.com/tasks/DT-1/report
```

Statuses: `acknowledged`, `en_route`, `on_station`, `complete`, `cancelled`. Reports land in the audit
log and on the console row. The token is `HARBOR_DRONE_TOKEN` (Render generates one; locally the server
prints one at startup if it is unset). Harbor authority sessions can read the same endpoints.

## How it decides

- `app/imm.py`: IMM estimator, four linear Kalman models (CV-quiet q=0.02, CV-agile q=0.5, two coordinated turns at ±0.08 rad/s), Markov 0.95 diagonal, depth as a separate 1D filter.
- `app/tracker.py`: chi-square 9.21 gating, Hungarian assignment (scipy), existence score with 3-of-5 confirmation and deletion after ~20 misses, drift cone grown 10 min under CV-quiet.
- `app/scoring.py`: naive-Bayes log-odds over four features (registry match, ping reply, heading to asset, maneuver evidence). Ratios in `config/scoring.json`.
- `app/ladder.py`: the rung machine, drafted actions, the 90 s timeout, `reconcile()`.
- `app/registry.py`: bookings, overlap and keep-out checks, window extension, distance outside the booked area.
- `app/feed.py`: the only mock sensor. `python -m app.feed` regenerates the scenario.

Positions are harbor-local meters (east, north); the map layer converts to lat/lon around `config/assets.json` origin.

## Responsible use

Misuse: the tracker and scorer could be pointed at legitimate research vehicles to harass or deny access,
the registry could leak dive schedules, and the maneuver feature could escalate unfamiliar vehicle classes
more often than familiar ones. Limits: nothing with an external effect leaves the system without a named
human's click and a log entry; the registry is opt-in and visible only to the harbor authority and the
booking operator; the acoustic ID check is a cooperative handshake, not a tracking beacon.
