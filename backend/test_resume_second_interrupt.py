"""Regression test for a live bug report: POST /resume used to assume
agent.graph.invoke(Command(resume=...)) always produces a final answer. If
resuming one decision immediately ran into ANOTHER request_human_approval
interrupt (e.g. the agent re-evaluates and proposes a new/updated
recommendation), /resume silently ignored the second "__interrupt__" and
fell through to extract_final_text()/extract_tool_outputs(), returning a
stale pre-interrupt "answer" - leaving that second interrupt stuck forever
(no client was ever told to resume it).

This never touches the real Groq API or a real graph run: agent.graph.invoke
is monkeypatched directly, since the point is to test main.py's handling of
whatever the graph returns, not the graph/LLM itself (that's covered by
test_agent_hitl.py and the live manual testing already done for this
feature).

Run with: python3 -m pytest test_resume_second_interrupt.py -v
(or: python3 test_resume_second_interrupt.py, no pytest needed)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import agent
import main


client = TestClient(main.app)


class _FakeInterrupt:
    def __init__(self, value):
        self.value = value


def test_resume_surfaces_a_second_interrupt_instead_of_a_stale_answer():
    """If resuming leads straight into another interrupt, /resume must
    return {"type": "interrupt", ...} - not silently report an answer."""

    def fake_invoke(command, config):
        return {
            "__interrupt__": [
                _FakeInterrupt(
                    {
                        "question": "Risk score 0.4 detected. The agent recommends "
                        "to approve claim CLM-1003. Confirm the expert's decision?"
                    }
                )
            ],
            # A real second-interrupt result would also carry a stale
            # "messages" list from before this new interrupt - included
            # here to prove /resume doesn't fall through and use it.
            "messages": [],
        }

    with patch.object(agent.graph, "invoke", side_effect=fake_invoke):
        response = client.post(
            "/resume",
            json={"thread_id": "CLM-1003", "approved": True, "reason": "проверено вручную"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "interrupt", f"expected a surfaced interrupt, got: {body}"
    assert body["thread_id"] == "CLM-1003"
    assert "0.4" in body["question"]
    assert "judge_score" not in body, "an interrupt response must not carry a judge_score"
    print("test_resume_surfaces_a_second_interrupt_instead_of_a_stale_answer: PASS ->", body)


def test_resume_without_a_second_interrupt_still_returns_a_normal_answer():
    """Sanity check / non-regression: the ordinary single-interrupt resume
    path (no second interrupt) must still return a normal answer with a
    judge_score, unaffected by the new check."""
    fake_messages = [agent.HumanMessage(content="Проверь этот claim на мошенничество")]

    def fake_invoke(command, config):
        return {"messages": fake_messages}

    fake_judge_result = {"groundedness": 1.0, "relevance": 1.0, "verdict": "pass"}

    with patch.object(agent.graph, "invoke", side_effect=fake_invoke), patch.object(
        agent, "extract_final_text", return_value="Claim CLM-1003 processed."
    ), patch("main.judge.judge_response", return_value=fake_judge_result):
        response = client.post(
            "/resume",
            json={"thread_id": "CLM-1003", "approved": True, "reason": "проверено вручную"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "answer"
    assert body["text"] == "Claim CLM-1003 processed."
    assert body["judge_score"] == fake_judge_result
    print("test_resume_without_a_second_interrupt_still_returns_a_normal_answer: PASS ->", body)


if __name__ == "__main__":
    import sys

    failures = 0
    for t in [
        test_resume_surfaces_a_second_interrupt_instead_of_a_stale_answer,
        test_resume_without_a_second_interrupt_still_returns_a_normal_answer,
    ]:
        try:
            t()
            print(f"OK   {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")

    sys.exit(1 if failures else 0)
