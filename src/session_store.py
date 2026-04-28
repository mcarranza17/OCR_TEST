from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


SESSIONS_DIR = Path("data/sessions")
DEFAULT_TTL_SECONDS = 3600

_SID_LEN = 12


def _is_valid_sid(sid: str) -> bool:
    return bool(sid) and len(sid) == _SID_LEN and sid.isalnum()


def session_dir(sid: str) -> Path:
    if not _is_valid_sid(sid):
        raise ValueError("session id inválido")
    return SESSIONS_DIR / sid


def state_path(sid: str) -> Path:
    return session_dir(sid) / "state.json"


def image_path(sid: str, kind: str) -> Path:
    if kind not in ("document", "selfie"):
        raise ValueError(f"kind inválido: {kind}")
    return session_dir(sid) / f"{kind}.jpg"


def create_session() -> str:
    sid = uuid.uuid4().hex[:_SID_LEN]
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_dir(sid).mkdir(parents=True, exist_ok=True)
    write_state(sid, {
        "created_at": time.time(),
        "document": None,
        "selfie": None,
    })
    return sid


def read_state(sid: str) -> dict[str, Any] | None:
    if not _is_valid_sid(sid):
        return None
    path = state_path(sid)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_state(sid: str, state: dict[str, Any]) -> None:
    path = state_path(sid)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(path)


def update_state(sid: str, **changes: Any) -> dict[str, Any]:
    state = read_state(sid) or {}
    state.update(changes)
    write_state(sid, state)
    return state


def save_image(sid: str, kind: str, image_bytes: bytes) -> None:
    path = image_path(sid, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jpg.tmp")
    tmp.write_bytes(image_bytes)
    tmp.replace(path)
    update_state(sid, **{kind: "ready", f"{kind}_at": time.time()})


def read_image(sid: str, kind: str) -> bytes | None:
    path = image_path(sid, kind)
    if not path.exists():
        return None
    return path.read_bytes()


def reset_kind(sid: str, kind: str) -> None:
    path = image_path(sid, kind)
    if path.exists():
        path.unlink()
    update_state(sid, **{kind: None, f"{kind}_at": None})


def cleanup_old(ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
    if not SESSIONS_DIR.exists():
        return
    now = time.time()
    for entry in SESSIONS_DIR.iterdir():
        if not entry.is_dir() or not _is_valid_sid(entry.name):
            continue
        state = read_state(entry.name)
        created_at = (state or {}).get("created_at", 0)
        if now - created_at > ttl_seconds:
            for child in entry.iterdir():
                try:
                    child.unlink()
                except OSError:
                    pass
            try:
                entry.rmdir()
            except OSError:
                pass
