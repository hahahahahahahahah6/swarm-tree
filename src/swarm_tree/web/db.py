"""SQLite storage with thread-local connections and ownership-scoped reads.

Call init_db(path) before serving to select a database; omitted paths use
<cwd>/serve-data/web.db. CRUD helpers use that database without a connection
argument. Creates return integer IDs; mutations return affected row counts.
"""

import datetime
import os
from pathlib import Path
import sqlite3
import threading

from ..config import PROVIDERS

_local = threading.local()
_path = None
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, pw_hash BLOB NOT NULL,
    salt BLOB NOT NULL, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trees (
    id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, name TEXT NOT NULL,
    yaml_text TEXT NOT NULL, updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, tree_id INTEGER,
    task TEXT NOT NULL, transcript_json TEXT, status TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, provider TEXT NOT NULL,
    model TEXT NOT NULL, prompt TEXT NOT NULL, png_path TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS keys (
    user_id INTEGER NOT NULL, provider TEXT NOT NULL, api_key TEXT NOT NULL,
    PRIMARY KEY(user_id, provider)
);
"""


def init_db(path=None):
    """Initialize and return this thread's connection; accept :memory: too.

    Explicit file paths are resolved against cwd, with parent directories
    created as needed. Each thread opens its own SQLite connection.
    """
    global _path
    target = os.fspath(path) if path is not None else str(Path.cwd() / 'serve-data/web.db')
    if target != ':memory:':
        target = str(Path(target).resolve())
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.fchmod(fd, 0o600)
            os.close(fd)
    old = getattr(_local, 'conn', None)
    if old is not None:
        old.close()
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    _path = target
    _local.conn, _local.path = conn, target
    return conn


def _conn():
    if getattr(_local, 'conn', None) is None or _local.path != _path:
        return init_db(_path)
    return _local.conn


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _write(sql, params):
    conn = _conn()
    with conn:
        cursor = conn.execute(sql, params)
    return cursor


def create_user(email, pw_hash, salt):
    """Insert a user, letting SQLite reject duplicate email addresses."""
    return _write('INSERT INTO users (email, pw_hash, salt, created) VALUES (?, ?, ?, ?)',
                  (email, pw_hash, salt, _now())).lastrowid


def get_user_by_email(email):
    """Return a user row or None."""
    return _conn().execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()


def get_user_by_id(user_id):
    """Return a user row or None."""
    return _conn().execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()


def create_tree(user_id, name, yaml_text):
    """Store a tree and return its ID."""
    return _write('INSERT INTO trees (user_id, name, yaml_text, updated) VALUES (?, ?, ?, ?)',
                  (user_id, name, yaml_text, _now())).lastrowid


def list_trees(user_id):
    """Return owned trees, most recently updated first."""
    return _conn().execute('SELECT * FROM trees WHERE user_id=? ORDER BY updated DESC, id DESC',
                           (user_id,)).fetchall()


def get_tree(id, user_id):
    """Return a tree only when owned by user_id."""
    return _conn().execute('SELECT * FROM trees WHERE id=? AND user_id=?', (id, user_id)).fetchone()


def update_tree(id, user_id, name, yaml_text):
    """Update an owned tree; return zero if missing or not owned."""
    return _write('UPDATE trees SET name=?, yaml_text=?, updated=? WHERE id=? AND user_id=?',
                  (name, yaml_text, _now(), id, user_id)).rowcount


def delete_tree(id, user_id):
    """Delete an owned tree, preserving its historical runs."""
    return _write('DELETE FROM trees WHERE id=? AND user_id=?', (id, user_id)).rowcount


def create_run(user_id, tree_id, task, status='running', transcript_json=None):
    """Create a run, rejecting a tree owned by someone else."""
    if tree_id is not None and get_tree(tree_id, user_id) is None:
        raise ValueError('tree not found')
    return _write('INSERT INTO runs (user_id, tree_id, task, transcript_json, status, created) '
                  'VALUES (?, ?, ?, ?, ?, ?)',
                  (user_id, tree_id, task, transcript_json, status, _now())).lastrowid


def get_run(id, user_id):
    """Return an owned run or None."""
    return _conn().execute('SELECT * FROM runs WHERE id=? AND user_id=?', (id, user_id)).fetchone()


def list_runs(user_id, tree_id=None, limit=10):
    """Return recent owned runs, optionally filtered to one tree."""
    if tree_id is None:
        return _conn().execute('SELECT * FROM runs WHERE user_id=? ORDER BY created DESC, id DESC LIMIT ?',
                               (user_id, limit)).fetchall()
    return _conn().execute('SELECT * FROM runs WHERE user_id=? AND tree_id=? '
                           'ORDER BY created DESC, id DESC LIMIT ?', (user_id, tree_id, limit)).fetchall()


def update_run_status(id, status, transcript_json):
    """Persist worker results; caller must supply an already-authorized run ID."""
    return _write('UPDATE runs SET status=?, transcript_json=? WHERE id=?',
                  (status, transcript_json, id)).rowcount


def create_image(user_id, provider, model, prompt, png_path):
    """Store image metadata and return its ID (does not write PNG bytes)."""
    return _write('INSERT INTO images (user_id, provider, model, prompt, png_path, created) '
                  'VALUES (?, ?, ?, ?, ?, ?)',
                  (user_id, provider, model, prompt, os.fspath(png_path), _now())).lastrowid


def list_images(user_id, limit=8):
    """Return recent owned image metadata."""
    return _conn().execute('SELECT * FROM images WHERE user_id=? ORDER BY created DESC, id DESC LIMIT ?',
                           (user_id, limit)).fetchall()


def get_image(id, user_id):
    """Return image metadata only when owned by user_id."""
    return _conn().execute('SELECT * FROM images WHERE id=? AND user_id=?', (id, user_id)).fetchone()


def set_key(user_id, provider, key):
    """Insert or replace a server-side key without logging its value."""
    _write('INSERT INTO keys (user_id, provider, api_key) VALUES (?, ?, ?) '
           'ON CONFLICT(user_id, provider) DO UPDATE SET api_key=excluded.api_key',
           (user_id, provider, key))


def delete_key(user_id, provider):
    """Remove an owned provider key."""
    return _write('DELETE FROM keys WHERE user_id=? AND provider=?', (user_id, provider)).rowcount


def get_key(user_id, provider):
    """Return the key or None; never pass this value to templates or logs."""
    row = _conn().execute('SELECT api_key FROM keys WHERE user_id=? AND provider=?',
                          (user_id, provider)).fetchone()
    return row['api_key'] if row else None


def keys_status(user_id):
    """Return set/unset booleans for every supported provider."""
    rows = _conn().execute('SELECT provider, LENGTH(api_key)>0 AS is_set FROM keys WHERE user_id=?',
                           (user_id,)).fetchall()
    result = dict.fromkeys(PROVIDERS, False)
    result.update((row['provider'], bool(row['is_set'])) for row in rows)
    return result
