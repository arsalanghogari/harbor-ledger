"""Naive-Bayes log-odds threat scorer plus the geometry feature extractor (asset ETA)."""
import json
import math


def load_config(path):
    with open(path) as f:
        return json.load(f)


def score(features: dict, cfg: dict):
    """features: name -> True (evidence for unauthorized) / False / None (unknown, skipped).
    Log-odds sum from the prior, clamped. Returns (posterior, two strongest features in words)."""
    lo = math.log(cfg["prior"] / (1 - cfg["prior"]))
    contrib = []
    for name, val in features.items():
        if val is None:
            continue
        f = cfg["features"][name]
        lr = f["yes"] if val else f["no"]
        lo += math.log(lr)
        contrib.append((math.log(lr), f["text_yes"] if val else f["text_no"]))
    p = 1 / (1 + math.exp(-lo))
    p = min(max(p, cfg["clamp"][0]), cfg["clamp"][1])
    contrib.sort(key=lambda c: -c[0])
    return p, [text for _, text in contrib[:2]]


def point_in_polygon(x, y, poly) -> bool:
    inside = False
    for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1]):
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def dist_to_segment(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    u = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + u * dx), py - (ay + u * dy))


def dist_to_polygon(x, y, poly) -> float:
    if point_in_polygon(x, y, poly):
        return 0.0
    return min(dist_to_segment(x, y, ax, ay, bx, by) for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1]))


def asset_eta(path, assets):
    """path: predicted [x, y] per second from now. Returns (asset name, seconds) for the first
    second at which the prediction is inside an asset's standoff zone, else None."""
    for s, (px, py) in enumerate(path):
        for a in assets:
            if dist_to_polygon(px, py, a["polygon"]) <= a["standoff_m"]:
                return a["name"], s
    return None
