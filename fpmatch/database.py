"""SQLite reference database + audit log for 1:N searches."""
from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cv2
import numpy as np

from .matcher import MatchResult, local_structures, match

SCHEMA = """
CREATE TABLE IF NOT EXISTS prints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id  TEXT NOT NULL,
    name        TEXT,
    finger      TEXT,
    source_file TEXT,
    sha256      TEXT NOT NULL,
    ppi         INTEGER,
    n_minutiae  INTEGER,
    quality     REAL,
    template    BLOB NOT NULL,
    thumbnail   BLOB,
    notes       TEXT,
    enrolled_at TEXT NOT NULL,
    enrolled_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_subject ON prints(subject_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sha ON prints(sha256, finger, subject_id);
CREATE TABLE IF NOT EXISTS audit (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     TEXT NOT NULL,
    user   TEXT,
    action TEXT NOT NULL,
    detail TEXT
);
"""

FINGERS = ["Unknown", "R thumb", "R index", "R middle", "R ring", "R little",
           "L thumb", "L index", "L middle", "L ring", "L little"]


APP_NAME = "FP-Match"


def user_data_dir() -> str:
    """Per-user folder for the database (writable even when the app is installed read-only).

    macOS:   ~/Library/Application Support/FP-Match
    Windows: %LOCALAPPDATA%\\FP-Match
    Linux:   $XDG_DATA_HOME/FP-Match or ~/.local/share/FP-Match
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, APP_NAME)


def default_db_path() -> str:
    """$FPMATCH_DB if set; else an existing data/fingerprints.db next to the app (older installs);
    else fingerprints.db in the per-user data folder."""
    if os.environ.get("FPMATCH_DB"):
        return os.environ["FPMATCH_DB"]
    if not getattr(sys, "frozen", False):
        legacy = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fingerprints.db")
        if os.path.exists(legacy):
            return legacy
    return os.path.join(user_data_dir(), "fingerprints.db")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _arr_to_blob(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, a.astype(np.float32), allow_pickle=False)
    return buf.getvalue()


def _blob_to_arr(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


class PrintDB:
    def __init__(self, path: str = "fingerprints.db"):
        path = os.path.expanduser(path.strip().strip('"\''))  # tolerate "Copy as path" quotes
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.executescript(SCHEMA)
        cols = {r[1] for r in self.con.execute("PRAGMA table_info(prints)")}
        for col in ("enhanced", "mask"):          # ML stage (added in 1.1); older rows are filled on demand
            if col not in cols:
                self.con.execute(f"ALTER TABLE prints ADD COLUMN {col} BLOB")
        self.con.commit()
        self._cache = None
        self._ml_cache = {}

    # ------------------------------------------------------------------ audit
    def log(self, action: str, user: str = "", **detail):
        self.con.execute("INSERT INTO audit(ts,user,action,detail) VALUES (?,?,?,?)",
                         (now(), user, action, json.dumps(detail, default=str)))
        self.con.commit()

    def audit_rows(self, limit=500):
        cur = self.con.execute("SELECT ts,user,action,detail FROM audit ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(ts=r[0], user=r[1], action=r[2], detail=r[3]) for r in cur]

    # ------------------------------------------------------------- enrolment
    def enroll(self, processed, subject_id: str, name: str = "", finger: str = "Unknown",
               source_file: str = "", ppi: int = 500, notes: str = "", user: str = ""):
        thumb = cv2.imencode(".png", _thumb(processed.enh.gray))[1].tobytes()
        enh = cv2.imencode(".png", processed.enh.enhanced)[1].tobytes()
        mask = cv2.imencode(".png", processed.enh.mask.astype(np.uint8) * 255)[1].tobytes()
        try:
            cur = self.con.execute(
                "INSERT INTO prints(subject_id,name,finger,source_file,sha256,ppi,n_minutiae,quality,"
                "template,thumbnail,notes,enrolled_at,enrolled_by,enhanced,mask) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (subject_id, name, finger, source_file, processed.sha256, ppi, processed.n,
                 processed.quality, _arr_to_blob(processed.template), thumb, notes, now(), user, enh, mask))
        except sqlite3.IntegrityError:
            return None  # same image already enrolled for this subject/finger
        self.con.commit()
        self._cache = None
        self.log("enroll", user, print_id=cur.lastrowid, subject_id=subject_id, finger=finger,
                 sha256=processed.sha256, source_file=source_file, minutiae=processed.n)
        return cur.lastrowid

    def delete(self, print_id: int, user: str = ""):
        row = self.con.execute("SELECT subject_id, sha256 FROM prints WHERE id=?", (print_id,)).fetchone()
        self.con.execute("DELETE FROM prints WHERE id=?", (print_id,))
        self.con.commit()
        self._cache = None
        self._ml_cache.pop(print_id, None)
        if row:
            self.log("delete", user, print_id=print_id, subject_id=row[0], sha256=row[1])

    def list(self):
        cur = self.con.execute("SELECT id,subject_id,name,finger,source_file,sha256,n_minutiae,quality,"
                               "enrolled_at,enrolled_by,notes FROM prints ORDER BY id")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur]

    def count(self):
        return self.con.execute("SELECT COUNT(*) FROM prints").fetchone()[0]

    def get(self, print_id: int):
        cur = self.con.execute("SELECT * FROM prints WHERE id=?", (print_id,))
        cols = [d[0] for d in cur.description]
        r = cur.fetchone()
        if not r:
            return None
        d = dict(zip(cols, r))
        d["template"] = _blob_to_arr(d["template"])
        return d

    def thumbnail(self, print_id: int):
        r = self.con.execute("SELECT thumbnail FROM prints WHERE id=?", (print_id,)).fetchone()
        if not r or r[0] is None:
            return None
        return cv2.imdecode(np.frombuffer(r[0], np.uint8), cv2.IMREAD_GRAYSCALE)

    def ml_arrays(self, print_id: int):
        """(enhanced uint8, mask bool) of a reference print for the ML stage."""
        if print_id in self._ml_cache:
            return self._ml_cache[print_id]
        r = self.con.execute("SELECT enhanced, mask FROM prints WHERE id=?", (print_id,)).fetchone()
        if r is None:
            return None
        if r[0] is None or r[1] is None:      # enrolled before 1.1: rebuild from the stored 500 ppi image
            from .enhance import EnhanceParams, enhance
            e = enhance(self.thumbnail(print_id), EnhanceParams())
            enh_png = cv2.imencode(".png", e.enhanced)[1].tobytes()
            mask_png = cv2.imencode(".png", e.mask.astype(np.uint8) * 255)[1].tobytes()
            self.con.execute("UPDATE prints SET enhanced=?, mask=? WHERE id=?", (enh_png, mask_png, print_id))
            self.con.commit()
            r = (enh_png, mask_png)
        enh = cv2.imdecode(np.frombuffer(r[0], np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.frombuffer(r[1], np.uint8), cv2.IMREAD_GRAYSCALE) > 127
        if len(self._ml_cache) > 300:
            self._ml_cache.pop(next(iter(self._ml_cache)))
        self._ml_cache[print_id] = (enh, mask)
        return enh, mask

    # ---------------------------------------------------------------- search
    def _templates(self):
        if self._cache is None:
            rows = self.con.execute("SELECT id, subject_id, name, finger, template FROM prints").fetchall()
            self._cache = []
            for pid, sid, name, finger, blob in rows:
                t = _blob_to_arr(blob)
                self._cache.append((pid, sid, name, finger, t, local_structures(t)))
        return self._cache

    def search(self, query_template: np.ndarray, top_k: int = 10, finger: str | None = None,
               progress=None, query_ml=None, verifier=None, ml_candidates: int = 200) -> list[dict]:
        """Minutiae search over every reference, then (if a verifier and the query's (enhanced, mask) are
        given) ML re-scoring of the best `ml_candidates` (all of them for databases up to 500 prints).
        Hits carry 'result' (MatchResult) and, when scored, 'ml' ({ml, cnn, overlap})."""
        entries = self._templates()
        if finger and finger != "Unknown":
            entries = [e for e in entries if e[3] in (finger, "Unknown")]
        lq = local_structures(query_template)

        def one(e):
            pid, sid, name, fing, t, lt = e
            return e, match(query_template, t, lq, lt)

        results = []
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
            for k, (e, r) in enumerate(ex.map(one, entries)):
                results.append(dict(print_id=e[0], subject_id=e[1], name=e[2], finger=e[3], result=r))
                if progress and k % 25 == 0:
                    progress((k + 1) / max(len(entries), 1))
        results.sort(key=lambda d: (d["result"].score, d["result"].n_supported), reverse=True)
        if verifier is not None and verifier.available and query_ml is not None and results:
            n = len(results) if len(results) <= 500 else ml_candidates
            cand = results[:n]
            pairs = [(query_ml[0], query_ml[1], *self.ml_arrays(h["print_id"]), h["result"]) for h in cand]
            for h, s in zip(cand, verifier.score(pairs)):
                h["ml"] = s
            results = sorted(cand, key=lambda d: (d["ml"]["ml"], d["result"].score), reverse=True) + results[n:]
        return results[:top_k]


def _thumb(gray, max_side=1600):  # keep full 500 ppi image so overlays line up
    h, w = gray.shape
    s = max_side / max(h, w)
    return cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
