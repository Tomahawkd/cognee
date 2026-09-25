"""Identifier matching contracts from the 2026-09-24 memory handoff."""

from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.modules.retrieval import lexical_retriever
from cognee.modules.retrieval.lexical_retriever import LexicalRetriever, tokenize_words


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ParseError", ["parseerror", "parse", "error"]),
        ("HTTPServer", ["httpserver", "http", "server"]),
        ("parse_error", ["parse_error", "parseerror", "parse", "error"]),
        ("parse-error", ["parse-error", "parseerror", "parse", "error"]),
    ],
)
def test_identifier_aliases_preserve_whole_token(text, expected):
    assert tokenize_words(text) == expected


def test_aliases_deduplicate_per_occurrence_but_preserve_term_frequency():
    assert Counter(tokenize_words("ParseParse ParseParse")) == {"parseparse": 2, "parse": 2}


def test_stop_words_apply_to_aliases_after_case_splitting():
    assert tokenize_words("ParseError", stop_words={"error"}) == ["parseerror", "parse"]


def test_lowercase_text_does_not_gain_guessed_boundaries():
    assert tokenize_words("parseerror parse error") == ["parseerror", "parse", "error"]
    assert tokenize_words("the parser", stop_words={"the"}) == ["parser"]
    assert tokenize_words("") == []


@pytest.mark.asyncio
async def test_word_query_finds_existing_camelcase_chunk_without_reembedding(monkeypatch):
    graph = SimpleNamespace(
        get_filtered_graph_data=AsyncMock(
            return_value=(
                [
                    ("unrelated", {"type": "DocumentChunk", "text": "Completely unrelated."}),
                    ("identifier", {"type": "DocumentChunk", "text": "ParseError"}),
                ],
                [],
            )
        )
    )
    monkeypatch.setattr(lexical_retriever, "get_graph_engine", AsyncMock(return_value=graph))
    retriever = LexicalRetriever(
        tokenizer=tokenize_words,
        scorer=lambda query, document: len(set(query) & set(document)),
        top_k=1,
        with_scores=True,
    )
    results = await retriever.get_retrieved_objects("parse error")
    assert results[0][0]["id"] == "identifier"
    assert results[0][1] == 2
    assert results[0][0]["text"] == "ParseError"
    graph.get_filtered_graph_data.assert_awaited_once()


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "HTTPServer_parse-error",
            ["httpserver_parse-error", "httpserverparseerror", "http", "server", "parse", "error"],
        ),
        ("HTTP HTTP", ["http", "http"]),
        ("parse--error", ["parse", "error"]),
        ("café 中文", ["café", "中文"]),
        ("version2Parser", ["version2parser", "version2", "parser"]),
    ],
)
def test_identifier_boundaries_preserve_ordinary_text(text, expected):
    assert tokenize_words(text) == expected
