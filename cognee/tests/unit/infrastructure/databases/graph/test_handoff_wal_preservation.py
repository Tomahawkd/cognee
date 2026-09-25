"""Do not destroy recovery evidence merely because opening a database failed.

Fault injection reproduces the recovery policy, not the historical corruption.
Neither opening path may unlink the WAL on a RuntimeError containing 'wal'.
"""

import importlib
from unittest.mock import Mock

import ladybug
import pytest

from cognee_db_workers import _kuzu_helpers, kuzu_worker
from cognee_db_workers.harness import HandleRegistry, Request
from cognee_db_workers.kuzu_protocol import OP_OPEN_DATABASE

adapter_module = importlib.import_module("cognee.infrastructure.databases.graph.ladybug.adapter")


@pytest.fixture(params=["in-process", "worker"])
def open_database(request, monkeypatch):
    def open_path(path, database_factory):
        if request.param == "worker":
            monkeypatch.setattr(ladybug, "Database", database_factory)
            return kuzu_worker._open_database(
                HandleRegistry(),
                Request(op=OP_OPEN_DATABASE, kwargs={"database_path": str(path)}),
            )
        monkeypatch.setattr(adapter_module, "Database", database_factory)
        monkeypatch.setattr(_kuzu_helpers, "install_json_extension_local", lambda **kwargs: None)
        # Exercise the real initialization method without allocating the
        # constructor's executor: all native opens below fail intentionally.
        adapter = object.__new__(adapter_module.LadybugAdapter)
        adapter.db_path = str(path)
        adapter.kuzu_buffer_pool_size = 64 * 1024 * 1024
        adapter.kuzu_max_db_size = 256 * 1024 * 1024
        adapter.kuzu_num_threads = 1
        adapter._initialize_connection()

    return open_path


@pytest.mark.parametrize(
    "message",
    [
        "Runtime exception: WAL checksum mismatch",
        "IO exception: permission denied reading graph.db.wal",
        "IO exception: failed to read database /tmp/walnut/graph.db",
    ],
    ids=["checksum", "wal-permission", "wal-in-directory-name"],
)
def test_failed_open_preserves_database_wal_and_original_error(tmp_path, open_database, message):
    path = tmp_path / "graph.db"
    wal = tmp_path / "graph.db.wal"
    path.write_bytes(b"damaged database evidence")
    wal.write_bytes(b"WAL evidence, potentially including committed transactions")
    before = {p.name: p.read_bytes() for p in (path, wal)}
    original = RuntimeError(message)
    # The second error models the handoff's sequence, not its unproven cause.
    factory = Mock(side_effect=[original, RuntimeError("Invalid database header")])

    with pytest.raises(RuntimeError) as raised:
        open_database(path, factory)

    after = {p.name: p.read_bytes() for p in (path, wal) if p.exists()}
    assert after == before, "Failed open must preserve the database and its recovery evidence"
    assert raised.value is original, "A retry must not obscure the first storage error"
    factory.assert_called_once()


def test_invalid_header_alone_leaves_files_untouched(tmp_path, open_database):
    path = tmp_path / "graph.db"
    wal = tmp_path / "graph.db.wal"
    path.write_bytes(b"invalid header evidence")
    wal.write_bytes(b"WAL evidence")
    before = {p.name: p.read_bytes() for p in (path, wal)}
    error = RuntimeError("Invalid database header")
    with pytest.raises(RuntimeError) as raised:
        open_database(path, Mock(side_effect=error))
    assert raised.value is error
    assert {p.name: p.read_bytes() for p in (path, wal)} == before


def test_missing_wal_does_not_turn_failure_into_a_retry(tmp_path, open_database):
    path = tmp_path / "graph.db"
    path.write_bytes(b"database evidence")
    error = RuntimeError("WAL checksum mismatch")
    factory = Mock(side_effect=[error, object()])
    with pytest.raises(RuntimeError) as raised:
        open_database(path, factory)
    assert raised.value is error
    factory.assert_called_once()
    assert path.read_bytes() == b"database evidence"


def test_version_probe_permission_error_preserves_open_error(tmp_path, open_database, monkeypatch):
    from cognee_db_workers import ladybug_migrate

    monkeypatch.setattr(
        ladybug_migrate, "needs_migration", Mock(side_effect=PermissionError("cannot read catalog"))
    )
    # The adapter re-exports the same helper through its legacy module.
    legacy = importlib.import_module(
        "cognee.infrastructure.databases.graph.ladybug.ladybug_migrate"
    )
    monkeypatch.setattr(legacy, "needs_migration", ladybug_migrate.needs_migration)
    original = RuntimeError("IO exception: permission denied reading graph.db")
    factory = Mock(side_effect=[original, object()])
    with pytest.raises(RuntimeError) as raised:
        open_database(tmp_path / "graph.db", factory)
    assert raised.value is original
    factory.assert_called_once()


def test_identified_legacy_format_still_migrates_before_retry(tmp_path, open_database, monkeypatch):
    from cognee_db_workers import ladybug_migrate

    legacy = importlib.import_module(
        "cognee.infrastructure.databases.graph.ladybug.ladybug_migrate"
    )
    migrate = Mock()
    for module in (ladybug_migrate, legacy):
        monkeypatch.setattr(module, "needs_migration", Mock(return_value=(True, "0.9.0")))
        monkeypatch.setattr(module, "ladybug_migration", migrate)

    # Stop at the second opening attempt, before schema initialization.
    retry_error = RuntimeError("sentinel after migration")
    factory = Mock(side_effect=[RuntimeError("incompatible storage version"), retry_error])
    path = tmp_path / "legacy.db"
    with pytest.raises(RuntimeError) as raised:
        open_database(path, factory)
    assert raised.value is retry_error
    assert factory.call_count == 2
    migrate.assert_called_once_with(
        new_db=str(path) + "_new",
        old_db=str(path),
        new_version=ladybug.__version__,
        old_version="0.9.0",
        overwrite=True,
    )
