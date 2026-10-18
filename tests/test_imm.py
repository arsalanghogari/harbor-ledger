import numpy as np

from app.imm import IMMTrack


def truth(speed, omega, n, straight=0):
    """Noiseless 1 Hz positions and velocities; turns at omega rad/s after `straight` steps."""
    psi, pos = 0.0, np.zeros(2)
    for t in range(n):
        if t >= straight:
            psi += omega
        v = speed * np.array([np.cos(psi), np.sin(psi)])
        pos = pos + v
        yield pos.copy(), v


def test_straight_track_converges_within_10_steps():
    f = None
    for k, (z, v) in enumerate(truth(1.5, 0.0, 12)):
        if f is None:
            f = IMMTrack(z)
            continue
        f.step(z)
        if k == 10:
            assert np.linalg.norm(f.x[:2] - z) < 0.5
            assert np.linalg.norm(f.x[2:] - v) < 0.1
            assert f.mode == "CV-quiet"


def test_turning_track_puts_ct_model_above_0_6_within_8_steps():
    # 3 m/s noiseless. At 1.5 m/s with 2 m sonar noise the same threshold takes ~13 steps (see BUILD_LOG).
    f, first = None, None
    for k, (z, _) in enumerate(truth(3.0, 0.08, 45, straight=25)):
        if f is None:
            f = IMMTrack(z)
            continue
        f.step(z)
        if k >= 25 and f.modes["CT+"] > 0.6:
            first = k - 25
            break
    assert first is not None and first <= 8
    assert f.mode == "CT+" and f.maneuvering > 0.6


def test_predict_ahead_follows_the_mode_mix():
    f = None
    for z, v in truth(1.5, 0.0, 30):
        f = IMMTrack(z) if f is None else (f.step(z), f)[1]
    path = f.predict_ahead(60)
    assert path.shape == (61, 2)
    straight = 60 * f.x[2:]
    moved = path[-1] - path[0]
    # Under the mode mix the lingering CT mass (looping branches) shortens the endpoint; it must
    # still head along the velocity and cover most of the straight-line distance.
    cos = moved @ straight / (np.linalg.norm(moved) * np.linalg.norm(straight))
    assert cos > 0.995 and 0.6 < np.linalg.norm(moved) / np.linalg.norm(straight) <= 1.01


def test_depth_is_filtered_separately():
    f = IMMTrack([0, 0], depth=10)
    for _ in range(10):
        f.step([0, 0], depth=12)
    assert abs(f.depth - 12) < 0.5
