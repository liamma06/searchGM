"""Email + password accounts.

Passwords are hashed with argon2id. A login is a signed, HttpOnly session cookie. Unsafe requests must carry
an `X-Requested-With: fetch` header, which a cross-site form post cannot add (a CSRF guard on top of
SameSite=Lax). Failed logins are throttled, and login always does a hash comparison so response time does
not reveal which emails exist. With AUTH_REQUIRED=0 the app runs open as a single local user."""
import re
import time
from collections import defaultdict, deque

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from . import db
from .config import settings

router = APIRouter()
hasher = PasswordHasher()
DUMMY_HASH = hasher.hash("not-a-real-password")  # checked when the email is unknown, so timing is the same
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")
LOCAL_EMAIL = db.LOCAL_EMAIL


class Limiter:
    """Sliding-window counter kept in memory (per process)."""

    def __init__(self):
        self.hits: dict[str, deque] = defaultdict(deque)

    def count(self, key: str, window: float) -> int:
        q, now = self.hits[key], time.time()
        while q and q[0] < now - window:
            q.popleft()
        return len(q)

    def hit(self, key: str) -> None:
        self.hits[key].append(time.time())

    def allow(self, key: str, limit: int, window: float) -> bool:
        if self.count(key, window) >= limit:
            return False
        self.hit(key)
        return True


login_failures, signups, asks = Limiter(), Limiter(), Limiter()


# ---------- request guards ----------
def require_xhr(request: Request) -> None:
    if request.headers.get("x-requested-with") != "fetch":
        raise HTTPException(403, "Missing X-Requested-With header")


def _local_user() -> dict:
    row = db.user_by_email(LOCAL_EMAIL)
    return db.strip_user(row) if row else db.create_user(LOCAL_EMAIL, "local", "!")  # "!" can never verify


def optional_user(request: Request) -> dict | None:
    if not settings.auth_required:
        return _local_user()
    uid = request.session.get("uid")
    return db.user_by_id(uid) if uid else None


def current_user(user: dict | None = Depends(optional_user)) -> dict:
    if user is None:
        raise HTTPException(401, "Sign in required")
    return user


def admin_user(user: dict = Depends(current_user)) -> dict:
    if not user["is_admin"]:
        raise HTTPException(403, "Admins only")
    return user


def check_ask_rate(user: dict) -> None:
    if not asks.allow(f"u{user['id']}", settings.rate_limit_per_hour, 3600):
        raise HTTPException(429, f"Limit reached: {settings.rate_limit_per_hour} questions per hour")


def _ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


# ---------- routes ----------
class Credentials(BaseModel):
    email: str
    password: str
    name: str = ""


@router.post("/auth/signup", dependencies=[Depends(require_xhr)])
def signup(body: Credentials, request: Request):
    if not settings.auth_required:
        raise HTTPException(400, "Sign-in is turned off (AUTH_REQUIRED=0)")
    if not signups.allow(_ip(request), 10, 3600):
        raise HTTPException(429, "Too many sign-ups from this address. Try again later.")
    email = body.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(400, "Enter a valid email address.")
    if not 8 <= len(body.password) <= 128:
        raise HTTPException(400, "Password must be 8 to 128 characters.")
    name = body.name.strip()[:60] or email.split("@")[0]
    try:
        user = db.create_user(email, name, hasher.hash(body.password))
    except ValueError:
        raise HTTPException(409, "That email is already registered.")
    request.session.clear()
    request.session["uid"] = user["id"]
    return user


@router.post("/auth/login", dependencies=[Depends(require_xhr)])
def login(body: Credentials, request: Request):
    email, ip = body.email.strip().lower(), _ip(request)
    if login_failures.count(f"{ip}|{email}", 900) >= 8 or login_failures.count(ip, 900) >= 30:
        raise HTTPException(429, "Too many attempts. Wait a few minutes and try again.")
    row = db.user_by_email(email)
    try:
        hasher.verify(row["pw_hash"] if row else DUMMY_HASH, body.password)
        ok = row is not None
    except (VerifyMismatchError, InvalidHashError):
        ok = False
    if not ok:
        login_failures.hit(f"{ip}|{email}")
        login_failures.hit(ip)
        raise HTTPException(401, "Wrong email or password.")
    if hasher.check_needs_rehash(row["pw_hash"]):
        db.set_pw_hash(row["id"], hasher.hash(body.password))
    request.session.clear()  # fresh session on login
    request.session["uid"] = row["id"]
    return db.strip_user(row)


@router.post("/auth/logout", dependencies=[Depends(require_xhr)])
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@router.get("/api/me")
def me(user: dict | None = Depends(optional_user)):
    return {"auth_required": settings.auth_required, "user": user}
