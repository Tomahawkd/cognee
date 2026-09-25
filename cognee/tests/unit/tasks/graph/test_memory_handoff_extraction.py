"""Handoff contracts at the real extraction boundary; no provider or storage calls.

The fake LLM deliberately returns noise: deterministic filtering must not rely
on a cooperative model. Prompt quality itself needs a separate model evaluation.
"""

import importlib
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.engine.models.node_set import NodeSet
from cognee.shared.data_models import Edge, KnowledgeGraph, Node

extraction = importlib.import_module("cognee.tasks.graph.extract_graph_from_data")

SOURCE = """Require a terminating newline when submitting a unified diff to apply_patch.
ParseError means the patch syntax was rejected; fix the terminator before retrying.
diff --git a/parser.py b/parser.py
index 1234567..abcdef0 100644
--- a/parser.py
+++ b/parser.py
@@ -1 +1 @@
-old
+new
"""


def make_chunk(node_set="session_learnings", *, objects=True):
    return DocumentChunk(
        text=SOURCE,
        chunk_size=len(SOURCE),
        chunk_index=0,
        cut_type="test",
        is_part_of=Document(
            name="lesson.txt",
            raw_data_location="lesson.txt",
            external_metadata=None,
            mime_type="text/plain",
        ),
        belongs_to_set=[NodeSet(name=node_set) if objects else node_set],
    )


def noisy_graph():
    names = [
        ("lesson", "Patch terminator", "practice"),
        ("error", "ParseError", "exception"),
        ("commit", "fix(parser): accept trailing newline", "commit message"),
        ("diff", "diff --git a/parser.py b/parser.py", "diff header"),
        ("git", "Git", "tool"),
        ("hash", "abc1234", "revision"),
    ]
    return KnowledgeGraph(
        nodes=[Node(id=id_, name=name, type=type_, description=name) for id_, name, type_ in names],
        edges=[
            Edge(source_node_id="lesson", target_node_id=target, relationship_name="explains")
            for target in ("error", "commit", "diff", "git", "hash")
        ],
    )


@pytest.fixture
def llm_boundary(monkeypatch):
    llm = AsyncMock(return_value=noisy_graph())
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", llm)
    monkeypatch.setattr(extraction, "get_configured_ontology_resolver", lambda _: None)
    monkeypatch.setattr(extraction, "find_existing_edge_identities", AsyncMock(return_value=set()))
    return llm


@pytest.mark.asyncio
@pytest.mark.parametrize("objects", [True, False], ids=["pipeline-NodeSet", "exported-string"])
@pytest.mark.parametrize(
    "node_set", ["session_learnings", "user_sessions_from_cache", "user_context"]
)
async def test_memory_extraction_removes_diff_headers_only_from_llm_input(
    llm_boundary, node_set, objects
):
    chunk = make_chunk(node_set, objects=objects)
    original = chunk.model_dump()
    await extraction.extract_graph_from_data([chunk], KnowledgeGraph)
    sent = llm_boundary.await_args.args[0]
    assert "diff --git" not in sent
    assert "index 1234567" not in sent
    assert "--- a/parser.py" not in sent
    assert "+++ b/parser.py" not in sent
    assert "@@ -1 +1 @@" not in sent
    assert "ParseError" in sent and "terminating newline" in sent
    assert chunk.text == original["text"]
    assert chunk.is_part_of.model_dump() == original["is_part_of"]


@pytest.mark.asyncio
async def test_artifacts_and_incident_edges_are_removed_before_embedding(llm_boundary):
    chunk = make_chunk()
    # Snapshot arguments at call time: later mutation cannot hide early embedding of noise.
    embedded_graphs = []

    def capture_embeddings(data, **kwargs):
        embedded_graphs.extend(
            g.model_copy(deep=True) for g in data if isinstance(g, KnowledgeGraph)
        )

    await extraction.extract_graph_from_data(
        [chunk], KnowledgeGraph, cache_entity_embeddings=capture_embeddings
    )
    assert embedded_graphs, "The embedding callback must receive the sanitized extraction"
    graph = embedded_graphs[0]
    assert {n.id for n in graph.nodes} == {"lesson", "error", "git", "hash"}
    assert {(e.source_node_id, e.target_node_id) for e in graph.edges} == {
        ("lesson", "error"),
        ("lesson", "git"),
        ("lesson", "hash"),
    }
    assert {entity.name for _, entity in chunk.contains} == {
        "patch terminator",
        "parseerror",
        "git",
        "abc1234",
    }
    assert len(chunk._produced_edge_identities) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node_set,custom_prompt",
    [("technical_manual", None), ("session_learnings", "Extract every diff header literally.")],
)
async def test_general_documents_and_explicit_prompts_keep_literal_content(
    llm_boundary, node_set, custom_prompt
):
    chunk = make_chunk(node_set)
    await extraction.extract_graph_from_data([chunk], KnowledgeGraph, custom_prompt=custom_prompt)
    assert llm_boundary.await_args.args[0] == SOURCE
    if custom_prompt:
        assert llm_boundary.await_args.args[1] == custom_prompt
    assert len(chunk.contains) == 6
    assert len(chunk._produced_edge_identities) == 5


@pytest.mark.asyncio
async def test_custom_model_keeps_memory_input_and_response(llm_boundary):
    class LiteralDocument(BaseModel):
        content: str

    response = LiteralDocument(content=SOURCE)
    llm_boundary.return_value = response
    chunk = make_chunk()
    await extraction.extract_graph_from_data([chunk], LiteralDocument)
    assert llm_boundary.await_args.args[0] == SOURCE
    assert chunk.contains is response


@pytest.mark.asyncio
async def test_memory_policy_is_scoped_per_chunk_in_a_mixed_batch(llm_boundary):
    memory = make_chunk()
    manual = make_chunk("technical_manual")
    llm_boundary.side_effect = lambda *args, **kwargs: noisy_graph()
    await extraction.extract_graph_from_data([memory, manual], KnowledgeGraph)
    prompts = [call.args[1] for call in llm_boundary.await_args_list]
    assert len(prompts) == 2
    assert prompts[0] != prompts[1], "Only the memory chunk should receive the memory policy"


@pytest.mark.asyncio
async def test_extraction_preserves_original_chunk_and_document(llm_boundary):
    chunk = make_chunk()
    original = chunk.model_dump()
    await extraction.extract_graph_from_data([chunk], KnowledgeGraph)
    assert chunk.id == original["id"]
    assert chunk.text == original["text"]
    assert chunk.is_part_of.model_dump() == original["is_part_of"]


@pytest.mark.asyncio
async def test_memory_prompt_requests_reusable_knowledge_and_source_spelling(llm_boundary):
    await extraction.extract_graph_from_data([make_chunk()], KnowledgeGraph)
    prompt = llm_boundary.await_args.args[1].lower()
    assert "reusable" in prompt and "source spelling" in prompt
    assert "commit subjects" in prompt and "diff headers" in prompt
    assert "command outcomes" in prompt
    assert "memory_policy" not in llm_boundary.await_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,type_",
    [
        ("feat(api)!: replace login", "change"),
        ("fix: accept trailing newline", "event"),
        ("diff --git a/a.py b/a.py", "text"),
        ("@@ -1,2 +1,3 @@ function", "text"),
        ("recorded patch", "diff_header"),
        ("recorded subject", "commit-message"),
    ],
)
async def test_artifact_safeguards_work_independently(llm_boundary, name, type_):
    graph = noisy_graph()
    graph.nodes[2].name = name
    graph.nodes[2].type = type_
    llm_boundary.return_value = graph
    chunk = make_chunk()
    await extraction.extract_graph_from_data([chunk], KnowledgeGraph)
    assert len(chunk.contains) == 4
    assert len(chunk._produced_edge_identities) == 3


@pytest.mark.asyncio
async def test_shared_cached_graph_is_not_filtered_for_general_document(llm_boundary):
    # A provider/cache hook may reuse an object. Filtering one memory must not
    # alter the graph the following non-memory chunk receives.
    graph = noisy_graph()
    original = graph.model_dump()
    llm_boundary.return_value = graph
    memory, manual = make_chunk(), make_chunk("technical_manual")
    await extraction.extract_graph_from_data([memory, manual], KnowledgeGraph)
    assert graph.model_dump() == original
    assert len(memory.contains) == 4
    assert len(manual.contains) == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_precomputed_graphs_are_filtered_before_async_embedding(llm_boundary, asynchronous):
    from unittest.mock import Mock

    graph = noisy_graph()
    original = graph.model_dump()
    calculate = (AsyncMock if asynchronous else Mock)(return_value=[graph])
    embedded = []

    async def embed(data, **kwargs):
        embedded.extend(g.model_copy(deep=True) for g in data if isinstance(g, KnowledgeGraph))

    chunk = make_chunk()
    await extraction.extract_graph_from_data(
        [chunk],
        KnowledgeGraph,
        calculate_chunk_graphs=calculate,
        cache_entity_embeddings=embed,
    )
    llm_boundary.assert_not_awaited()
    assert {n.id for n in embedded[0].nodes} == {"lesson", "error", "git", "hash"}
    assert graph.model_dump() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 2])
async def test_precomputed_graph_count_mismatch_is_not_silently_truncated(llm_boundary, count):
    from unittest.mock import Mock

    from cognee.tasks.graph.exceptions import InvalidChunkGraphInputError

    embed = Mock()
    with pytest.raises(InvalidChunkGraphInputError, match="length mismatch"):
        await extraction.extract_graph_from_data(
            [make_chunk()],
            KnowledgeGraph,
            calculate_chunk_graphs=lambda *args, **kwargs: [noisy_graph() for _ in range(count)],
            cache_entity_embeddings=embed,
        )
    embed.assert_not_called()


@pytest.mark.asyncio
async def test_memory_with_no_extracted_knowledge_is_valid(llm_boundary):
    llm_boundary.return_value = KnowledgeGraph(nodes=[], edges=[])
    chunk = make_chunk()
    await extraction.extract_graph_from_data([chunk], KnowledgeGraph)
    assert not chunk.contains
    assert not chunk._produced_edge_identities


@pytest.mark.asyncio
async def test_preparation_preserves_technical_lessons_and_code(llm_boundary):
    chunk = make_chunk()
    chunk.text = (
        "Use diff --git headers when explaining patches.\n"
        "Keep index lookups bounded.\n"
        "---\n"
        "import parser\n"
        "from parser import ParseError\n"
        "+raise ParseError()\n"
    )
    await extraction.extract_graph_from_data([chunk], KnowledgeGraph)
    assert llm_boundary.await_args.args[0] == chunk.text


@pytest.mark.asyncio
async def test_all_artifact_graph_integrates_as_empty_memory(llm_boundary):
    graph = noisy_graph()
    graph.nodes = [node for node in graph.nodes if node.id in {"commit", "diff"}]
    graph.edges = [
        Edge(source_node_id="commit", target_node_id="diff", relationship_name="contains")
    ]
    llm_boundary.return_value = graph
    chunk = make_chunk()
    embedded = []

    def embed(data, **kwargs):
        embedded.extend(g for g in data if isinstance(g, KnowledgeGraph))

    await extraction.extract_graph_from_data([chunk], KnowledgeGraph, cache_entity_embeddings=embed)
    assert embedded[0].nodes == [] and embedded[0].edges == []
    assert not chunk.contains
    assert not chunk._produced_edge_identities
