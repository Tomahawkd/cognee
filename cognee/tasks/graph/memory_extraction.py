"""Narrow extraction policy for conversational memory, not general documents."""

import re

from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.shared.data_models import KnowledgeGraph

_MEMORY_NODE_SETS = {"session_learnings", "user_sessions_from_cache", "user_context"}
_DIFF_HEADER = re.compile(
    r"^(?:diff --git .+|index [0-9a-f]+\.\.[0-9a-f]+(?: \d+)?|"
    r"--- (?:a/\S+|/dev/null)|\+\+\+ (?:b/\S+|/dev/null)|"
    r"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@.*)$"
)
_COMMIT_SUBJECT = re.compile(
    r"^(?:feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?:\([^\n)]+\))?!?:\s+\S",
    re.IGNORECASE,
)
_ARTIFACT_TYPES = {"commit message", "commit subject", "diff header", "patch header", "diff hunk"}


def is_memory_chunk(chunk: DocumentChunk) -> bool:
    """Support both pipeline NodeSets and exported string names."""
    return any(
        (node_set if isinstance(node_set, str) else getattr(node_set, "name", None))
        in _MEMORY_NODE_SETS
        for node_set in (getattr(chunk, "belongs_to_set", None) or [])
    )


def prepare_memory_extraction_text(text: str) -> str:
    """Drop literal diff bookkeeping only from the extraction input."""
    return "".join(
        line for line in text.splitlines(keepends=True) if not _DIFF_HEADER.fullmatch(line.strip())
    )


def filter_memory_graph(graph: KnowledgeGraph) -> KnowledgeGraph:
    """Remove obvious artifacts and incident edges without mutating cached output."""
    removed = {
        node.id
        for node in graph.nodes
        if _COMMIT_SUBJECT.match(node.name.strip())
        or _DIFF_HEADER.fullmatch(node.name.strip())
        or re.sub(r"[_-]+", " ", node.type.strip().lower()) in _ARTIFACT_TYPES
    }
    return graph.model_copy(
        update={
            "nodes": [node for node in graph.nodes if node.id not in removed],
            "edges": [
                edge
                for edge in graph.edges
                if edge.source_node_id not in removed and edge.target_node_id not in removed
            ],
        }
    )
