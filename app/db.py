"""MongoDB storage for accounts and saved chats.

A run keeps its full event stream (plan, retrieval, verification, citations, ...) so opening a past chat
replays the graph, trace and sources exactly as they were. Everything goes through this module.

With MONGODB_URI set it talks to MongoDB (Atlas). Without it, it falls back to an in-memory mongomock
database, which is fine for local development and tests but loses saved data on restart."""
import json
import re
import secrets
import time
import uuid

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from .config import settings

LOCAL_EMAIL = "local@localhost"
MAX_EVENTS_BYTES = 1_500_000

_db = None
backend = "not started"
_user_cache: dict[str, tuple[float, dict | None]] = {}
USER_TTL = 60  # seconds a signed-in user's record is reused (role changes show up within a minute)


def init(uri: str | None = None, name: str | None = None) -> None:
    """Connect (or reconnect) and make sure the indexes exist. Passing uri="" forces the in-memory store."""
    global _db, backend
    uri = settings.mongodb_uri if uri is None else uri
    if uri:
        client = MongoClient(uri, serverSelectionTimeoutMS=8000, appname="signal")
        client.admin.command("ping")  # fail fast, with the driver's own message (bad password, IP not allowed, ...)
        backend = "mongodb"
    else:
        import mongomock

        client = mongomock.MongoClient()
        backend = "in-memory (MONGODB_URI not set: saved data is lost on restart)"
    _db = client[name or settings.mongodb_db]
    _user_cache.clear()
    _db.users.create_index("email", unique=True)
    _db.chats.create_index([("user_id", 1), ("updated_at", -1)])
    _db.chats.create_index("share_token", unique=True, sparse=True)  # only shared chats carry the field
    _db.runs.create_index([("chat_id", 1), ("_id", 1)])


def _d():
    if _db is None:
        init()
    return _db


# ---------- users ----------
def public_user(doc: dict) -> dict:
    return {"id": str(doc["_id"]), "name": doc["name"], "email": doc["email"], "is_admin": bool(doc.get("is_admin"))}


def strip_user(user: dict) -> dict:
    return {k: user[k] for k in ("id", "name", "email", "is_admin")}


def create_user(email: str, name: str, pw_hash: str) -> dict:
    """The first real account (or any address in ADMIN_USERS) becomes an admin."""
    users = _d().users
    first = users.count_documents({"email": {"$ne": LOCAL_EMAIL}}) == 0  # the local no-login user doesn't count
    try:
        res = users.insert_one({"email": email, "name": name, "pw_hash": pw_hash, "is_admin": first or email in settings.admin_users, "created_at": time.time()})
    except DuplicateKeyError:
        raise ValueError("email already registered")
    return public_user(users.find_one({"_id": res.inserted_id}))


def user_by_email(email: str) -> dict | None:
    doc = _d().users.find_one({"email": email})
    return {**public_user(doc), "pw_hash": doc["pw_hash"]} if doc else None


def user_by_id(uid) -> dict | None:
    hit = _user_cache.get(str(uid))
    if hit and hit[0] > time.time():
        return hit[1]
    try:
        doc = _d().users.find_one({"_id": ObjectId(uid)})
    except (InvalidId, TypeError):
        return None
    user = public_user(doc) if doc else None
    _user_cache[str(uid)] = (time.time() + USER_TTL, user)
    return user


def set_pw_hash(uid: str, pw_hash: str) -> None:
    _d().users.update_one({"_id": ObjectId(uid)}, {"$set": {"pw_hash": pw_hash}})


# ---------- chats ----------
def create_chat(user_id: str, title: str) -> str:
    cid, now = uuid.uuid4().hex, time.time()
    _d().chats.insert_one({"_id": cid, "user_id": user_id, "title": title[:120], "created_at": now, "updated_at": now})
    return cid


def list_chats(user_id: str) -> list[dict]:
    chats = list(_d().chats.find({"user_id": user_id}).sort("updated_at", -1).limit(200))
    counts = {r["_id"]: r["n"] for r in _d().runs.aggregate([
        {"$match": {"chat_id": {"$in": [c["_id"] for c in chats]}}},
        {"$group": {"_id": "$chat_id", "n": {"$sum": 1}}},
    ])}
    return [{"id": c["_id"], "title": c["title"], "preview": c.get("preview", ""), "updated_at": c["updated_at"], "created_at": c.get("created_at", c["updated_at"]),
             "shared": bool(c.get("share_token")), "runs": counts.get(c["_id"], 0)} for c in chats]


def _with_runs(chat: dict) -> dict:
    runs = _d().runs.find({"chat_id": chat["_id"]}).sort("_id", 1)
    return {
        "id": chat["_id"], "title": chat["title"], "shared": bool(chat.get("share_token")),
        "runs": [{"id": str(r["_id"]), "question": r["question"], "kind": r["kind"], "answer": r["answer"], "events": json.loads(r["events"]),
                  "secs": r.get("secs"), "created_at": r["created_at"]} for r in runs],
    }


def get_chat(chat_id: str, user_id: str) -> dict | None:
    chat = _d().chats.find_one({"_id": chat_id, "user_id": user_id})
    return _with_runs(chat) if chat else None


def delete_chat(chat_id: str, user_id: str) -> bool:
    if _d().chats.delete_one({"_id": chat_id, "user_id": user_id}).deleted_count == 0:
        return False
    _d().runs.delete_many({"chat_id": chat_id})
    return True


def rename_chat(chat_id: str, user_id: str, title: str) -> bool:
    return _d().chats.update_one({"_id": chat_id, "user_id": user_id}, {"$set": {"title": title[:120]}}).matched_count > 0


def delete_chats(chat_ids: list[str], user_id: str) -> int:
    """Delete several chats at once. Only ones this user owns are touched, whatever ids are passed."""
    owned = [c["_id"] for c in _d().chats.find({"_id": {"$in": chat_ids}, "user_id": user_id}, {"_id": 1})]
    if owned:
        _d().chats.delete_many({"_id": {"$in": owned}})
        _d().runs.delete_many({"chat_id": {"$in": owned}})
    return len(owned)


def set_share(chat_id: str, user_id: str, on: bool) -> str | None:
    chat = _d().chats.find_one({"_id": chat_id, "user_id": user_id})
    if not chat:
        raise LookupError("chat not found")
    if on:
        token = chat.get("share_token") or secrets.token_urlsafe(16)
        _d().chats.update_one({"_id": chat_id}, {"$set": {"share_token": token}})
        return token
    _d().chats.update_one({"_id": chat_id}, {"$unset": {"share_token": ""}})
    return None


def get_shared(token: str) -> dict | None:
    chat = _d().chats.find_one({"share_token": token})
    return _with_runs(chat) if chat else None


def save_run(chat_id: str, question: str, kind: str, answer: str, events: list[dict], secs: float) -> None:
    blob = json.dumps(events, ensure_ascii=False)  # a JSON string: immune to field-name rules, easy to size-check
    if len(blob) > MAX_EVENTS_BYTES:  # keep the answer and citations, drop the bulky retrieval detail
        blob = json.dumps([e for e in events if e.get("type") in ("plan", "verification", "citations", "grounding", "done", "error")], ensure_ascii=False)
    now = time.time()
    _d().runs.insert_one({"chat_id": chat_id, "question": question, "kind": kind, "answer": answer, "events": blob, "secs": secs, "created_at": now})
    text = re.sub(r"[*`#>]", "", re.sub(r"\[\d+\]", "", answer))  # drop citation markers and markdown symbols
    preview = " ".join(re.sub(r"\s+([.,;:!?])", r"\1", text).split())[:200]  # shown under the title on the chats page
    _d().chats.update_one({"_id": chat_id}, {"$set": {"updated_at": now, "preview": preview}})
