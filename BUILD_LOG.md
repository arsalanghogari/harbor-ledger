# Build log

Built with Claude (Fable 5.1) in Claude Code from the Harbor Ledger design doc (v2, Oct 17 2026).
Stack: Python 3.13, FastAPI + uvicorn, numpy, scipy, pytest; Leaflet from CDN, vanilla JS, polling.

## Build order

| # | Module | Done when | Status |
|---|---|---|---|
| 1 | models.py | Imports cleanly | Done. In-memory `Ledger` holds the five tables as lists/dicts; events dump to JSON |
| 2 | imm.py | Straight track converges in 10 steps; turning track puts > 0.6 on a CT model in 8 steps | Done. 8 steps holds at 3 m/s noiseless; ~13 steps at 1.5 m/s with 2 m noise |
| 3 | tracker.py | Two crossing contacts keep their ids | Done. Two-stage Hungarian: confirmed tracks assigned first, tentative take the leftovers |
| 4 | scoring.py | Unmatched + silent + heading > 0.8 | Done (0.89) |
| 5 | registry.py | Overlap check rejects a conflicting booking | Done, plus keep-out zone no-go for rung 3+ contacts |
| 6 | ladder.py | E reaches rung 4, B reconciles to rung 1, D never passes rung 3 | Done |
| 7 | feed.py + scenario | Replay yields the right row counts per contact | Done: A 241, B 231, E 151 (C and D cut later to declutter the map) |
| 8 | main.py | Snapshot JSON visible in browser | Done |
| 9 | index.html | Full demo scenario runs end to end | Done |

## Hand-tuned, and why

1. **Maneuver evidence uses the single dominant non-quiet model**, not the sum over CV-agile + CT+ + CT-.
   With a 0.95-diagonal switching matrix, three models that a slow drifter cannot distinguish share
   0.75 of the mass by symmetry, so the summed definition fires on every piece of debris. The max of a
   single alternative only exceeds 0.6 with real evidence.
2. **IMM prior mode mix is [0.7, 0.1, 0.1, 0.1]** instead of uniform, for the same reason: a uniform start
   puts 0.75 on the maneuver models before the track has said anything.
3. **E's 30 degree turn sits at t+6 to t+12**, before rung 2. The combined prediction is propagated under
   the mode mix, so while CT mass lingers the predicted path curves and no ETA appears. Early placement lets
   it decay before the rung 4 trigger. CT+ rises to about 0.4 during the turn (visible on hover) but never
   dominates at 1.5 m/s with 2 m sonar noise, so maneuver evidence stays negative in the demo and rung 4
   comes from the ETA, as the demo table says.
4. **Rung 4 for E fires at t+75 (ETA 115 s), not t+55 (ETA 110 s).** The CV-agile model's noisy velocity
   makes the combined ETA flicker, so the first second it dips under 120 s is a little late. Deterministic
   with the baked scenario.
5. **The Hold card is frozen at push time.** Live posterior and ETA wobble with velocity noise; the card the
   officer saw is the card that is logged.
6. **Silence is measured from the first ping.** The second ping at rung 3 does not reset the 10 s timer,
   otherwise the posterior dips for 10 s at every rung 3 entry.
7. **Existence score is a unit-step LLR** (hit +1, miss -1, cap 5, delete below -15: 20 misses from a full
   score). Tentative tracks need 3 hits; they are dropped silently after 5 scans without them.
8. **Keep-out radius 300 m** at rung 3+ is internal only: it draws the drafted advisory zone on the map and
   makes the registry refuse overlapping bookings. Nothing is sent.
9. **Scenario B transits to the end** so only C and D become lost records inside the arc; the run stops with
   the last contact row (no tail), otherwise every known vehicle turns into a lost record at the end.
10. **D plateaus at 0.64**, not the 0.55 in the doc, straight from the likelihood table (8 x 6 x 1/1.5 x 1/2 on
    prior odds 1/9).
11. **Drift cone is literal**: covariance grown 10 min under CV-quiet (about 120, 440 and 1200 m at 2, 5 and
    10 min). The map draws only the 2 and 5 min rings; the 10 min ring covered the whole harbor. All rings
    are non-interactive so they never steal hover from the markers.
12. **Officer name** is required on every decision; the hidden "Close channel" option needs a typed reason,
    checked server side.

13. **Operator sign-in is standard library, not Cognito.** scrypt hashes in `config/users.json`, an
    HMAC-signed cookie, one `current_operator()` dependency. No AWS account, no network on demo day;
    Cognito or SSO can replace the module behind the same dependency. No rate limit on `/login`.

14. **Booked areas are never amended.** The design had the system grow a box to include a drifted
    vehicle that answered its ping. Changed on the team's call: the authorized area stays as the operator
    registered it, the vehicle is identified, and an out-of-area flag shows on the console row, the map
    label and the operator's booking. Being outside the authorized area is a signal, not a correction.
    The 30 min window extension for a late vehicle is unchanged.

15. **Drone tasking is a rung 5 release, not an automatic publish.** "Task observation drone" is the
    second-gentlest option on the card. Confirming it publishes a JSON tasking (live lat/lon from the
    harbor-local origin, compass course, speed, depth, 30/60/120 s predictions, asset ETA) that the drone
    polls with a bearer token and reports back to. Only that contact's data leaves, and only under the
    officer's name.

16. **Scenario trimmed to three contacts and two assets** (the research dock, the late research AUV
    and the debris were cut) so the map reads at a glance. The window-extension and lost-track paths
    still exist and are unit tested; they are no longer shown in the demo arc.

17. **Map symbology per the spec, driven by one derived field.** The engine computes `exception` and
    `rank` per contact each tick (keep-out incursion 1, triage 2, registered lost 3, out of area 4,
    unknown lost 5) and the page is a lookup on it. Hand-tuned: the incursion holds 10 s after the last
    ETA hit so IMM jitter does not blink the zone; dive boxes use a 20/40 m show/hide margin; the
    candidate score is 1 - d / (gate + halo radius). The scenario was rebuilt to the spec's demo
    (three registered, sio-auv-3 lost, the unknown reappearing beside it). Label collision handling is
    not implemented.

## What was not built

- Raising the real feed rate at rung 3 (stub event only).
- Basemap is OpenStreetMap tiles darkened with a CSS filter (CARTO dark tiles now need an API key); if tiles do not load the console still works on a plain dark canvas. Lat/lon alignment is approximate; positions are harbor-local meters around a fixed origin in `config/assets.json`.
- No persistence across restarts; `/control reset` rebuilds everything from the scenario file.
