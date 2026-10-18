from app.engine import SCORING
from app.scoring import asset_eta, dist_to_polygon, load_config, score

cfg = load_config(SCORING)
SQUARE = [[100, -50], [200, -50], [200, 50], [100, 50]]


def test_unmatched_silent_heading_gives_over_0_8():
    p, top = score({"unmatched": True, "silent": True, "heading": True, "maneuver": False}, cfg)
    assert p > 0.8
    assert top == [cfg["features"]["unmatched"]["text_yes"], cfg["features"]["silent"]["text_yes"]]


def test_matched_and_answering_is_benign():
    p, _ = score({"unmatched": False, "silent": False, "heading": False, "maneuver": False}, cfg)
    assert p <= 0.02


def test_unknown_features_are_skipped_and_clamped():
    p, top = score({"unmatched": True, "silent": None, "heading": None, "maneuver": None}, cfg)
    assert abs(p - (0.1 * 8) / (0.1 * 8 + 0.9)) < 1e-9 and len(top) == 1
    p, _ = score({"unmatched": True, "silent": True, "heading": True, "maneuver": True}, cfg)
    assert abs(p - 40 / 41) < 1e-9  # 1/9 * 8 * 6 * 3 * 2.5 = 40:1, under the 0.99 clamp
    p, _ = score({"unmatched": False, "silent": False, "heading": False, "maneuver": False}, cfg)
    assert p == cfg["clamp"][0]


def test_asset_eta_along_a_straight_path():
    assets = [{"name": "Pier", "polygon": SQUARE, "standoff_m": 20}]
    path = [(1.0 * s, 0.0) for s in range(120)]  # 1 m/s east from the origin toward x=100
    assert asset_eta(path, assets) == ("Pier", 80)
    assert asset_eta([(0.0, 1.0 * s) for s in range(120)], assets) is None
    assert dist_to_polygon(150, 0, SQUARE) == 0 and dist_to_polygon(90, 0, SQUARE) == 10
