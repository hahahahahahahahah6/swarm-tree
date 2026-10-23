"""Password hashing, signed seven-day sessions, and login failure tracking."""

import hashlib
import hmac
from http.cookies import CookieError, SimpleCookie
import os
from pathlib import Path
import secrets
import threading
import time

from . import db

SESSION_COOKIE = 'swarm_tree_session'
SESSION_MAX_AGE = 7 * 24 * 60 * 60
_fails = {}
_fail_lock = threading.Lock()
_secret_lock = threading.Lock()


def hash_password(pw):
    """Return (16-byte random salt, PBKDF2-SHA256 hash) for a password."""
    salt = secrets.token_bytes(16)
    return salt, hashlib.pbkdf2_hmac('sha256', pw.encode('utf-8'), salt, 200_000)


def verify_password(pw, salt, hash):
    """Compare password hashes in constant time."""
    candidate = hashlib.pbkdf2_hmac('sha256', pw.encode('utf-8'), salt, 200_000)
    return hmac.compare_digest(candidate, hash)


def _secret():
    """Read an environment secret or create the persistent local secret once."""
    value = os.environ.get('SWARM_TREE_SECRET')
    if value is not None:
        return value.encode('utf-8')
    path = Path.cwd() / 'serve-data/.secret'
    with _secret_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, 'w', encoding='ascii') as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(secrets.token_hex(32))
        return path.read_bytes()


def sign_session(user_id):
    """Return uid|expiry|HMAC for a session valid for seven days."""
    payload = f'{int(user_id)}|{int(time.time()) + SESSION_MAX_AGE}'
    signature = hmac.new(_secret(), payload.encode('ascii'), hashlib.sha256).hexdigest()
    return f'{payload}|{signature}'


def verify_session(value):
    """Return the user ID, or None for malformed, tampered, or expired input."""
    if not isinstance(value, str):
        return None
    try:
        uid, expiry, signature = value.split('|')
        user_id, expires = int(uid), int(expiry)
        if user_id < 1 or expires <= time.time():
            return None
        payload = f'{uid}|{expiry}'.encode('ascii')
        expected = hmac.new(_secret(), payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected.encode('ascii'), signature.encode('ascii')):
            return None
    except (ValueError, UnicodeError):
        return None
    return user_id


def get_user(request):
    """Resolve a BaseHTTPRequestHandler's session cookie to a user row.

    The HTTP layer should set SESSION_COOKIE with HttpOnly, SameSite=Lax,
    Path=/, and Max-Age=SESSION_MAX_AGE, and clear it on logout.
    """
    cookie = SimpleCookie()
    try:
        cookie.load(request.headers.get('Cookie', ''))
    except CookieError:
        return None
    morsel = cookie.get(SESSION_COOKIE)
    user_id = verify_session(morsel.value) if morsel else None
    return db.get_user_by_id(user_id) if user_id is not None else None


def record_fail(ip):
    """Record a failure; return True after more than five failures in 300s."""
    now = time.monotonic()
    with _fail_lock:
        for address in list(_fails):
            recent = [stamp for stamp in _fails[address] if stamp > now - 300]
            if recent:
                _fails[address] = recent
            else:
                del _fails[address]
        failures = _fails.setdefault(ip, [])
        failures.append(now)
        return len(failures) > 5


def clear_fails(ip):
    """Clear an IP's failure history after successful authentication."""
    with _fail_lock:
        _fails.pop(ip, None)
