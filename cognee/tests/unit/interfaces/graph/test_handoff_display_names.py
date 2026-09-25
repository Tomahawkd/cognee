"""Source spelling is presentation metadata, never a new entity identity."""

import importlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.engine.models.Entity import Entity
from cognee.modules.graph.utils.expand_with_nodes_and_edges import construct_data_points_and_edges
from cognee.modules.visualization.preprocessor import (
    COMPACT_PROPERTY_KEYS,
    compact_node,
    preprocess,
)
from cognee.shared.data_models import KnowledgeGraph, Node

formatting = importlib.import_module("cognee.modules.graph.methods.get_formatted_graph_data")


def test_entity_construction_preserves_source_spelling_and_canonical_identity():
    chunks = [
        DocumentChunk(
            text=name,
            chunk_size=len(name),
            chunk_index=0,
            cut_type="test",
            is_part_of=Document(
                name="lesson.txt",
                raw_data_location="lesson.txt",
                external_metadata=None,
                mime_type="text/plain",
            ),
        )
        for name in ("ParseError", "parseerror")
    ]
    graphs = [
        KnowledgeGraph(
            nodes=[Node(id="e", name=name, type="exception", description="Syntax error")], edges=[]
        )
        for name in ("ParseError", "parseerror")
    ]
    points, _ = construct_data_points_and_edges(chunks, graphs)
    entities = [p for p in points.values() if isinstance(p, Entity)]
    assert len(entities) == 1
    entity = entities[0]
    assert entity.id == Entity.id_for("parseerror") == Entity.id_for("ParseError")
    assert entity.name == "parseerror"
    assert entity.metadata["index_fields"] == ["name"]
    assert entity.metadata["identity_fields"] == ["name"]
    assert entity.model_dump().get("display_name") == "ParseError"


@pytest.mark.asyncio
@pytest.mark.parametrize("view", ["graph-json", "visualization", "streamed-visualization"])
@pytest.mark.parametrize(
    "display_name,expected",
    [
        ("ParseError", "ParseError"),
        (None, "parseerror"),
        ("", "parseerror"),
    ],
)
async def test_graph_views_prefer_display_name_and_support_legacy_nodes(
    monkeypatch, view, display_name, expected
):
    properties = {"type": "Entity", "name": "parseerror"}
    if display_name is not None:
        properties["display_name"] = display_name
    raw = ([("e", properties)], [])
    if view == "streamed-visualization":
        projected = {
            key: value for key, value in properties.items() if key in COMPACT_PROPERTY_KEYS
        }
        projected["type"] = properties["type"]
        node = compact_node("e", projected)
        assert node["name"] == expected
    elif view == "visualization":
        node = preprocess(raw).nodes[0]
        assert node["name"] == expected
    else:

        @asynccontextmanager
        async def database_context(*args):
            yield

        monkeypatch.setattr(formatting, "set_database_global_context_variables", database_context)
        monkeypatch.setattr(
            formatting,
            "get_authorized_dataset",
            AsyncMock(return_value=SimpleNamespace(owner_id=uuid4())),
        )
        monkeypatch.setattr(
            formatting,
            "get_graph_engine",
            AsyncMock(return_value=SimpleNamespace(get_graph_data=AsyncMock(return_value=raw))),
        )
        node = (await formatting.get_formatted_graph_data(uuid4(), SimpleNamespace()))["nodes"][0]
        assert node["label"] == expected
    assert node["id"] == "e"
    assert properties["name"] == "parseerror"


@pytest.mark.asyncio
async def test_storage_open_failure_propagates_before_graph_formatting(monkeypatch):
    @asynccontextmanager
    async def database_context(*args):
        yield

    error = RuntimeError("Invalid database header")
    monkeypatch.setattr(formatting, "set_database_global_context_variables", database_context)
    monkeypatch.setattr(
        formatting,
        "get_authorized_dataset",
        AsyncMock(return_value=SimpleNamespace(owner_id=uuid4())),
    )
    monkeypatch.setattr(formatting, "get_graph_engine", AsyncMock(side_effect=error))
    with pytest.raises(RuntimeError) as raised:
        await formatting.get_formatted_graph_data(uuid4(), SimpleNamespace())
    assert raised.value is error
