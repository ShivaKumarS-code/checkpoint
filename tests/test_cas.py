"""Tests for the content-addressed blob store."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import init_db


@pytest.fixture()
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = init_db(tmp_path / "cassettes.db")
    yield connection
    connection.close()


@pytest.fixture()
def store(conn: sqlite3.Connection) -> BlobStore:
    return BlobStore(conn)


def test_writing_the_same_blob_twice_stores_one_row(store: BlobStore) -> None:
    data = b"same bytes"
    first = store.put(data)
    second = store.put(data)
    assert first == second
    assert store.count() == 1


def test_blob_hash_is_sha256_of_input(store: BlobStore) -> None:
    data = b"hello checkpoint"
    assert store.put(data) == hashlib.sha256(data).hexdigest()


def test_put_stores_compressed_payload_with_original_size(
    conn: sqlite3.Connection, store: BlobStore
) -> None:
    data = b"A" * 10_000
    blob_hash = store.put(data)
    row = conn.execute(
        "SELECT compression, byte_size, payload FROM blobs WHERE blob_hash = ?",
        (blob_hash,),
    ).fetchone()
    assert row["compression"] == "zstd"
    assert row["byte_size"] == len(data)
    assert len(row["payload"]) < len(data)


def test_get_roundtrips_binary_data(store: BlobStore) -> None:
    data = bytes(range(256))
    assert store.get(store.put(data)) == data


def test_get_missing_blob_raises_keyerror(store: BlobStore) -> None:
    with pytest.raises(KeyError):
        store.get("0" * 64)


def test_json_blobs_are_canonicalized_and_deduplicated(store: BlobStore) -> None:
    first = store.put_json({"b": 2, "a": 1})
    second = store.put_json({"a": 1, "b": 2})
    assert first == second
    assert store.count() == 1
    assert store.get_json(first) == {"a": 1, "b": 2}


def test_exists(store: BlobStore) -> None:
    blob_hash = store.put(b"x")
    assert store.exists(blob_hash)
    assert not store.exists("f" * 64)
