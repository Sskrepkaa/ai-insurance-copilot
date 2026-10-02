"""Tests for GET /chat/{claim_id}/history and agent.extract_chat_history().

Covers the feature request: chat history must survive switching claims on
the dashboard and coming back, including the edge case of a claim left
mid-interrupt (human_check pending, never resumed).

Exercises extract_chat_history() directly with fabricated message lists
(no Groq API needed) for the filtering rules, and the FastAPI endpoint
itself with agent.graph.get_state monkeypatched (no real graph/checkpointer
run needed - this is about main.py's response shaping, which
test_agent_hitl.py's live-tested helpers already cover on the agent side).

Run with: python3 -m pytest test_chat_history.py -v
(or: python3 test_chat_history.py, no pytest needed)
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent
import main


client = TestClient(main.app)


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {"id": call_id, "name": name, "args": args}


def _tool_message(call_id: str, name: str, content) -> ToolMessage:
    return ToolMessage(content=json.dumps(content), name=name, tool_call_id=call_id)


def test_extract_chat_history_keeps_only_real_turns():
    """A full thread with tool calls, a HITL round, and a follow-up
    question should reduce to just: user question, final answer, next
    user question, next final answer - nothing else."""
    messages = [
        HumanMessage(content="Проверь этот claim на мошенничество"),
        AIMessage(content="", tool_calls=[_tool_call("c1", "check_fraud_risk", {"claim_id": "CLM-1003"})]),
        _tool_message("c1", "check_fraud_risk", {"risk_score": 0.925, "level": "high"}),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "c2",
                    "request_human_approval",
                    {"claim_id": "CLM-1003", "risk_score": 0.925, "recommendation": "reject"},
                )
            ],
        ),
        _tool_message(
            "c2",
            "request_human_approval",
            {"claim_id": "CLM-1003", "risk_score": 0.925, "recommendation": "reject"},
        ),
        HumanMessage(content="Expert decision recorded: approved=True, reason='проверено вручную'"),
        AIMessage(content="Claim CLM-1003 processed per the expert's decision."),
        HumanMessage(content="Этот клиент уже подавал похожий иск?"),
        AIMessage(content="Yes, two prior claims on record."),
    ]

    history = agent.extract_chat_history(messages)

    assert history == [
        {"role": "user", "text": "Проверь этот claim на мошенничество"},
        {"role": "assistant", "text": "Claim CLM-1003 processed per the expert's decision."},
        {"role": "user", "text": "Этот клиент уже подавал похожий иск?"},
        {"role": "assistant", "text": "Yes, two prior claims on record."},
    ]


def test_extract_chat_history_skips_already_decided_note():
    messages = [
        HumanMessage(content="Проверь этот claim"),
        AIMessage(content="Done."),
        HumanMessage(content="Note: claim CLM-1003 already has a recorded expert decision ..."),
        AIMessage(content="As already decided, this claim was approved."),
    ]
    history = agent.extract_chat_history(messages)
    assert history == [
        {"role": "user", "text": "Проверь этот claim"},
        {"role": "assistant", "text": "Done."},
        {"role": "assistant", "text": "As already decided, this claim was approved."},
    ]


def test_extract_chat_history_on_empty_thread():
    assert agent.extract_chat_history([]) == []


def test_history_endpoint_for_never_chatted_claim_returns_empty():
    """get_state() on a thread_id with no checkpoint returns values={},
    next=(), interrupts=() - the endpoint must turn that into an empty,
    non-error response."""
    fake_state = SimpleNamespace(values={}, next=(), interrupts=())
    with patch.object(agent.graph, "get_state", return_value=fake_state):
        response = client.get("/chat/CLM-9999/history")

    assert response.status_code == 200
    assert response.json() == {"messages": [], "pending_interrupt": None}


def test_history_endpoint_restores_completed_conversation():
    fake_messages = [
        HumanMessage(content="Проверь этот claim на мошенничество"),
        AIMessage(content="Fraud risk is low (0.16)."),
    ]
    fake_state = SimpleNamespace(values={"messages": fake_messages}, next=(), interrupts=())
    with patch.object(agent.graph, "get_state", return_value=fake_state):
        response = client.get("/chat/CLM-1002/history")

    assert response.status_code == 200
    body = response.json()
    assert body["pending_interrupt"] is None
    assert body["messages"] == [
        {"role": "user", "text": "Проверь этот claim на мошенничество"},
        {"role": "assistant", "text": "Fraud risk is low (0.16)."},
    ]


class _FakeInterrupt:
    def __init__(self, value):
        self.value = value


def test_history_endpoint_surfaces_a_pending_interrupt():
    """A claim left mid-HITL (never resumed) must come back with
    pending_interrupt populated in the same shape /chat's own interrupt
    response uses, so the frontend can reuse its Approve/Reject rendering."""
    fake_messages = [
        HumanMessage(content="Проверь этот claim на мошенничество и дай рекомендацию."),
        AIMessage(content="", tool_calls=[_tool_call("c1", "check_fraud_risk", {"claim_id": "CLM-1003"})]),
        _tool_message("c1", "check_fraud_risk", {"risk_score": 0.925, "level": "high"}),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "c2",
                    "request_human_approval",
                    {"claim_id": "CLM-1003", "risk_score": 0.925, "recommendation": "reject"},
                )
            ],
        ),
        _tool_message(
            "c2",
            "request_human_approval",
            {"claim_id": "CLM-1003", "risk_score": 0.925, "recommendation": "reject"},
        ),
    ]
    fake_state = SimpleNamespace(
        values={"messages": fake_messages},
        next=("human_check",),
        interrupts=(
            _FakeInterrupt(
                {
                    "question": "Risk score 0.925 detected. The agent recommends to reject "
                    "claim CLM-1003. Confirm the expert's decision?"
                }
            ),
        ),
    )
    with patch.object(agent.graph, "get_state", return_value=fake_state):
        response = client.get("/chat/CLM-1003/history")

    assert response.status_code == 200
    body = response.json()
    # No visible AI text yet (the only AIMessages so far are pure tool
    # calls) - just the employee's original question.
    assert body["messages"] == [
        {"role": "user", "text": "Проверь этот claim на мошенничество и дай рекомендацию."}
    ]
    assert body["pending_interrupt"] == {
        "type": "interrupt",
        "question": "Risk score 0.925 detected. The agent recommends to reject claim CLM-1003. "
        "Confirm the expert's decision?",
        "thread_id": "CLM-1003",
    }


if __name__ == "__main__":
    import sys

    failures = 0
    for t in [
        test_extract_chat_history_keeps_only_real_turns,
        test_extract_chat_history_skips_already_decided_note,
        test_extract_chat_history_on_empty_thread,
        test_history_endpoint_for_never_chatted_claim_returns_empty,
        test_history_endpoint_restores_completed_conversation,
        test_history_endpoint_surfaces_a_pending_interrupt,
    ]:
        try:
            t()
            print(f"OK   {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")

    sys.exit(1 if failures else 0)
