"""Content-addressed blob store backed by SQLite and zstd."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from typing import Any

import msgspec
import zstandard

from checkpoint.matching.normalize import canonical_json

_INSERT_SQL = (
    "INSERT OR IGNORE INTO blobs (blob_hash, compression, byte_size, payload, created_at) "
    "VALUES (?, 'zstd', ?, ?, ?)"
)


class BlobStore:
    """Stores immutable, content-addressed payloads in the cassette."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._compressor = zstandard.ZstdCompressor()
        self._decompressor = zstandard.ZstdDecompressor()

    def put(self, data: bytes) -> str:
        """Store bytes; returns the SHA-256 hash. Storing twice keeps one row."""
        blob_hash = hashlib.sha256(data).hexdigest()
        self._conn.execute(
            _INSERT_SQL,
            (blob_hash, len(data), self._compressor.compress(data), time.time()),
        )
        self._conn.commit()
        return blob_hash

    def put_json(self, value: Any) -> str:
        return self.put(canonical_json(value))

    def get(self, blob_hash: str) -> bytes:
        row = self._conn.execute(
            "SELECT payload FROM blobs WHERE blob_hash = ?", (blob_hash,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown blob: {blob_hash}")
        payload: bytes = row["payload"]
        return self._decompressor.decompress(payload)

    def get_json(self, blob_hash: str) -> Any:
        return msgspec.json.decode(self.get(blob_hash))

    def exists(self, blob_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM blobs WHERE blob_hash = ?", (blob_hash,)
        ).fetchone()
        return row is not None

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM blobs").fetchone()
        return int(row["n"])
