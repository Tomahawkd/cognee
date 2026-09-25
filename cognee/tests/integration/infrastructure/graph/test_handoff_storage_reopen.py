"""Real local crash/reopen coverage, independent of LLMs and embeddings.

This controlled crash does not reproduce or establish the cause of the damaged
database in HANDOFF-codex_sast_1-2026-09-24.md. It verifies committed graph data
survives with and without an explicit checkpoint, including full property scans.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("ladybug")

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

SEED_AND_EXIT = textwrap.dedent("""
    import asyncio
    import json
    import os
    import sys
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    async def seed():
        graph = LadybugAdapter(
            sys.argv[1], kuzu_num_threads=1,
            kuzu_buffer_pool_size=64 * 1024 * 1024,
            kuzu_max_db_size=256 * 1024 * 1024,
        )
        await graph.checkpoint()
        await graph.query("BEGIN TRANSACTION")
        for id_, name in (("error", "ParseError"), ("lesson", "Patch terminator")):
            await graph.query(
                "CREATE (:Node {id: $id, name: $name, type: 'Entity', properties: $props})",
                {"id": id_, "name": name.lower(), "props": json.dumps({
                    "display_name": name, "description": "Preserve the terminating newline.",
                })},
            )
        await graph.query(
            "MATCH (a:Node {id: 'error'}), (b:Node {id: 'lesson'}) "
            "CREATE (a)-[:EDGE {relationship_name: 'resolved_by', properties: $props}]->(b)",
            {"props": json.dumps({"edge_text": "ParseError is resolved by the patch terminator."})},
        )
        await graph.query("COMMIT")
        if sys.argv[2] == "checkpoint":
            await graph.checkpoint()
        # No adapter.close(), destructors, or graceful process shutdown.
        os._exit(0)

    asyncio.run(seed())
""")


@pytest.mark.asyncio
@pytest.mark.timeout(90)
@pytest.mark.parametrize("mode", ["wal-replay", "checkpoint"])
async def test_committed_graph_survives_abrupt_exit_and_full_scan(tmp_path, monkeypatch, mode):
    from cognee.infrastructure.databases.graph.ladybug import adapter as adapter_module

    monkeypatch.setattr(adapter_module.cache_config, "shared_ladybug_lock", False)
    path = tmp_path / "graph.db"
    env = {**os.environ, "SHARED_LADYBUG_LOCK": "false", "PYTHON_DOTENV_DISABLED": "1"}
    result = subprocess.run(
        [sys.executable, "-c", SEED_AND_EXIT, str(path), mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if mode == "wal-replay":
        wal = tmp_path / "graph.db.wal"
        assert wal.exists() and wal.stat().st_size > 0, "Exercise actual WAL replay"

    # Two complete opens/scans also check that the recovered store can checkpoint
    # and reopen normally, rather than merely answering a count-only query once.
    for _ in range(2):
        graph = LadybugAdapter(
            str(path),
            kuzu_num_threads=1,
            kuzu_buffer_pool_size=64 * 1024 * 1024,
            kuzu_max_db_size=256 * 1024 * 1024,
        )
        try:
            nodes, edges = await graph.get_graph_data()
            assert dict(nodes) == {
                id_: {
                    "name": name.lower(),
                    "type": "Entity",
                    "display_name": name,
                    "description": "Preserve the terminating newline.",
                }
                for id_, name in (("error", "ParseError"), ("lesson", "Patch terminator"))
            }
            assert edges == [
                (
                    "error",
                    "lesson",
                    "resolved_by",
                    {"edge_text": "ParseError is resolved by the patch terminator."},
                )
            ]
            await graph.checkpoint()
        finally:
            await graph.close()
