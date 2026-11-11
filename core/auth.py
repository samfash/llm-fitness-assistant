"""API keys. Only a SHA-256 hash is stored; the raw key is shown once at creation.

    python -m core.auth create alice --label laptop     # prints the key ONCE
    python -m core.auth list [--user alice]
    python -m core.auth revoke fa_AbCdEfGh              # by the 8-character prefix shown in `list`

Keys are 256-bit random values, so a fast hash is appropriate (unlike low-entropy passwords).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import secrets

from . import db

KEY_PREFIX = "fa_"
_USER_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def valid_user_id(user_id: str) -> bool:
    return bool(_USER_RE.match(user_id or ""))


def create_key(user_id: str, label: str | None = None) -> str:
    if not valid_user_id(user_id):
        raise ValueError("user id must be 1-64 characters of letters, digits and _ . @ -")
    raw = KEY_PREFIX + secrets.token_urlsafe(32)
    with db.connect() as c:
        c.execute("INSERT INTO api_keys(key_hash, user_id, label, prefix) VALUES (?,?,?,?)",
                  (_hash(raw), user_id, label, raw[:8]))
    return raw


def verify_key(raw: str | None) -> str | None:
    """Return the key's user id, or None if it is unknown or revoked."""
    if not raw or not raw.startswith(KEY_PREFIX) or len(raw) > 200:
        return None
    with db.connect() as c:
        row = c.execute("SELECT user_id FROM api_keys WHERE key_hash=? AND revoked=0", (_hash(raw),)).fetchone()
    return row["user_id"] if row else None


def revoke_key(prefix: str) -> int:
    if len(prefix) < 8:
        raise ValueError("give the full 8-character prefix, e.g. fa_AbCdEfGh")
    with db.connect() as c:
        cur = c.execute("UPDATE api_keys SET revoked=1 WHERE prefix=? AND revoked=0", (prefix[:8],))
        return cur.rowcount


def list_keys(user_id: str | None = None) -> list[dict]:
    sql, params = "SELECT prefix, user_id, label, revoked, created_at FROM api_keys", ()
    if user_id:
        sql, params = sql + " WHERE user_id=?", (user_id,)
    with db.connect() as c:
        return [dict(r) for r in c.execute(sql + " ORDER BY id", params).fetchall()]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m core.auth")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("user_id")
    c.add_argument("--label")
    ls = sub.add_parser("list")
    ls.add_argument("--user")
    rv = sub.add_parser("revoke")
    rv.add_argument("prefix")
    a = p.parse_args(argv)
    if a.cmd == "create":
        print(f"API key for {a.user_id} (store it now, it cannot be shown again):\n{create_key(a.user_id, a.label)}")
    elif a.cmd == "list":
        for k in list_keys(a.user):
            print(f"{k['prefix']}...  user={k['user_id']}  label={k['label']}  {'REVOKED' if k['revoked'] else 'active'}  {k['created_at']}")
    else:
        print(f"revoked {revoke_key(a.prefix)} key(s)")


if __name__ == "__main__":
    main()
