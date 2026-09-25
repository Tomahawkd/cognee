"""Prompt-delivery contracts, not proof that an LLM follows the instructions.

Assert concepts rather than snapshotting exact prompt wording. Evaluation of
generated lessons against real models remains separate from this offline suite.
"""

import re
from unittest.mock import AsyncMock

import pytest

from cognee.modules.session_distillation import distill
from cognee.modules.session_distillation.models import (
    CuratorBatchOutput,
    ProposedLesson,
    WrittenLesson,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["curator", "writer"])
async def test_distillation_delivers_artifact_guidance_to_the_model(monkeypatch, stage):
    llm = AsyncMock()
    monkeypatch.setattr(distill.LLMGateway, "acreate_structured_output", llm)
    if stage == "curator":
        llm.return_value = CuratorBatchOutput(lessons=[])
        await distill.curate_batch(
            "User: Preserve the patch terminator lesson, not the diff header."
        )
    else:
        llm.return_value = WrittenLesson(accept=False, reason="not_durable")
        await distill.write_or_reject(
            ProposedLesson(
                working_statement="The command exited successfully.", member_entry_ids=[]
            ),
            [],
            [],
            [],
        )
    llm.assert_awaited_once()
    prompt = llm.await_args.kwargs["system_prompt"].lower()
    assert re.search(r"durable|reusable|lasting", prompt)
    assert re.search(r"commit.{0,25}(subject|message)|conventional commit", prompt)
    assert re.search(r"diff.{0,25}(header|hunk)|patch.{0,25}(header|hunk)", prompt)
    assert re.search(
        r"bookkeeping|command.{0,25}(outcome|output)|execution.{0,25}(detail|event)", prompt
    )
