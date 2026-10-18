"""Operator accounts and sessions with the standard library only: scrypt password hashes in
config/users.json and HMAC-signed session cookies. Everything the app needs is behind
current_operator(), so Cognito or SSO can replace this module later without touching routes.

  python -m app.auth add <username> "<Operator name>"   # prompts for a password
  python -m app.auth seed                                # writes the three demo accounts
"""
import getpass
import hashlib
import hmac
import json
import os
import secrets
import sys
import time

from fastapi import Cookie, HTTPException

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS = os.path.join(ROOT, "config", "users.json")
# ponytail: a fresh secret per process signs out everyone on restart; set HARBOR_SECRET to persist
SECRET = os.environ.get("HARBOR_SECRET", secrets.token_hex(32)).encode()
COOKIE = "harbor_session"
SESSION_S = 12 * 3600


def load_users() -> dict:
    try:
        with open(USERS) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def hash_password(password: str, salt: str) -> str:
    return hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1).hex()


def verify(username: str, password: str):
    """{username, operator} if the password matches, else None. Hashes even for unknown users
    so the response time does not reveal which usernames exist."""
    u = load_users().get(username) or {"salt": "00" * 16, "hash": "00" * 64, "operator": ""}
    ok = hmac.compare_digest(hash_password(password, u["salt"]), u["hash"])
    return {"username": username, "operator": u["operator"]} if ok and username in load_users() else None


def issue(username: str) -> str:
    msg = f"{username}:{int(time.time()) + SESSION_S}"
    return f"{msg}:{hmac.new(SECRET, msg.encode(), 'sha256').hexdigest()}"


def parse(token: str):
    try:
        username, exp, sig = token.rsplit(":", 2)
    except (ValueError, AttributeError):
        return None
    good = hmac.compare_digest(hmac.new(SECRET, f"{username}:{exp}".encode(), "sha256").hexdigest(), sig)
    return username if good and int(exp) > time.time() else None


def current_operator(harbor_session: str | None = Cookie(default=None)) -> dict:
    """FastAPI dependency: the signed-in operator or 401."""
    username = parse(harbor_session) if harbor_session else None
    user = load_users().get(username) if username else None
    if not user:
        raise HTTPException(401, "sign in as a registered operator first")
    return {"username": username, "operator": user["operator"]}


def add_user(username: str, operator: str, password: str):
    users = load_users()
    salt = secrets.token_hex(16)
    users[username] = {"operator": operator, "salt": salt, "hash": hash_password(password, salt)}
    with open(USERS, "w") as f:
        json.dump(users, f, indent=2)


if __name__ == "__main__":
    if sys.argv[1:2] == ["add"] and len(sys.argv) == 4:
        add_user(sys.argv[2], sys.argv[3], getpass.getpass("password: "))
        print("added", sys.argv[2])
    elif sys.argv[1:2] == ["seed"]:
        demo = [("officer", "Port of LA Watch Officer"), ("usc", "USC AUV Lab"),
                ("pacific", "Pacific Hull Inspection"), ("scripps", "Scripps Research")]
        lines = ["# Demo accounts (local demo only)", "", "| username | operator | password |", "|---|---|---|"]
        for username, operator in demo:
            pw = f"{username}-{secrets.token_hex(2)}"
            add_user(username, operator, pw)
            lines.append(f"| {username} | {operator} | {pw} |")
        with open(os.path.join(ROOT, "config", "DEMO_ACCOUNTS.md"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("seeded", len(demo), "accounts; passwords in config/DEMO_ACCOUNTS.md")
    else:
        print(__doc__)
