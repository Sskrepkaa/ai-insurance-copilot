"""Unit tests for the HITL routing logic in agent.py - no real graph
execution, no Groq API, just the pure message-list routing functions.

Regression test for a live bug report: human_check() used to scan the
*entire* thread history for any past check_fraud_risk result above the
threshold, so once a claim had ever been flagged high-risk, a later and
completely unrelated follow-up question (e.g. "has this client filed a
similar claim before?") wrongly re-triggered an Approve/Reject interrupt.

The fix makes the trigger both (a) intent-based - only an explicit
request_human_approval tool call counts, not merely having seen a high
risk_score - and (b) turn-scoped - only the tool-call round that just ran
is inspected, never older rounds further back in the thread.

Run with: python3 -m pytest test_agent_hitl.py -v
(or: python3 test_agent_hitl.py, which runs the same cases without pytest)
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {"id": call_id, "name": name, "args": args}


def _tool_message(call_id: str, name: str, content: dict | list) -> ToolMessage:
    return ToolMessage(content=json.dumps(content), name=name, tool_call_id=call_id)


def test_route_after_tools_ignores_a_stale_approval_request_from_earlier_in_the_thread():
    """Exact repro of the reported bug: a claim was flagged high-risk and
    the agent asked for approval several turns ago (already resolved); a
    brand-new, unrelated question later in the same thread must NOT
    re-trigger the HITL interrupt just because that old history is still
    in state["messages"]."""
    messages = [
        # Turn 1: fraud check -> high risk -> explicit approval request.
        HumanMessage(content="Проверь этот claim на мошенничество"),
        AIMessage(
            content="",
            tool_calls=[_tool_call("c1", "check_fraud_risk", {"claim_id": "CLM-1003"})],
        ),
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
        # Resume happened; decision recorded; agent gave its final answer.
        HumanMessage(content="Expert decision recorded: approved=True, reason='проверено вручную'"),
        AIMessage(content="Claim CLM-1003 processed per the expert's decision.", tool_calls=[]),
        # Turn 2: a totally unrelated follow-up question in the SAME thread.
        HumanMessage(content="Этот клиент уже подавал похожий иск?"),
        AIMessage(
            content="",
            tool_calls=[_tool_call("c3", "get_customer_history", {"client": "Sergei Volkov"})],
        ),
        _tool_message(
            "c3",
            "get_customer_history",
            [{"claim_id": "CLM-0654", "date": "2021-05-30", "outcome": "rejected"}],
        ),
    ]

    state = {"messages": messages}

    assert agent.route_after_tools(state) == "agent", (
        "an unrelated get_customer_history call must not be treated as a "
        "human-approval trigger just because an earlier turn in the same "
        "thread had a high-risk check_fraud_risk + request_human_approval"
    )


def test_route_after_tools_triggers_on_explicit_approval_request():
    """Positive case: when the *current* tool-call round includes
    request_human_approval, routing must go to human_check."""
    messages = [
        HumanMessage(content="Проверь этот claim на мошенничество"),
        AIMessage(
            content="",
            tool_calls=[_tool_call("c1", "check_fraud_risk", {"claim_id": "CLM-1003"})],
        ),
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

    assert agent.route_after_tools({"messages": messages}) == "human_check"


def test_route_after_tools_does_not_trigger_on_plain_fraud_check_without_recommendation():
    """Merely reporting a high risk_score (no recommendation) must not
    trigger HITL - only an explicit request_human_approval call does."""
    messages = [
        HumanMessage(content="Какой риск мошенничества у этого claim?"),
        AIMessage(
            content="",
            tool_calls=[_tool_call("c1", "check_fraud_risk", {"claim_id": "CLM-1003"})],
        ),
        _tool_message("c1", "check_fraud_risk", {"risk_score": 0.925, "level": "high"}),
    ]

    assert agent.route_after_tools({"messages": messages}) == "agent"


def test_should_continue_routes_to_tools_then_end():
    with_tool_calls = {
        "messages": [
            AIMessage(content="", tool_calls=[_tool_call("c1", "check_fraud_risk", {"claim_id": "CLM-1001"})])
        ]
    }
    assert agent.should_continue(with_tool_calls) == "tools"

    without_tool_calls = {"messages": [AIMessage(content="final answer", tool_calls=[])]}
    assert agent.should_continue(without_tool_calls) == agent.END


def test_human_check_builds_question_from_the_approval_payload_and_records_decision(monkeypatch):
    """human_check() must read claim_id/risk_score/recommendation out of the
    request_human_approval ToolMessage (not the stale fraud-check one) and
    surface them in the interrupt question; on "resume" it must record the
    human's decision as a HumanMessage."""
    captured_interrupt_payload = {}

    def fake_interrupt(payload):
        captured_interrupt_payload.update(payload)
        return {"approved": False, "reason": "looks suspicious"}

    monkeypatch.setattr(agent, "interrupt", fake_interrupt)

    messages = [
        HumanMessage(content="Проверь этот claim на мошенничество"),
        AIMessage(
            content="",
            tool_calls=[_tool_call("c1", "check_fraud_risk", {"claim_id": "CLM-1003"})],
        ),
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

    result = agent.human_check({"messages": messages})

    assert "0.925" in captured_interrupt_payload["question"]
    assert "CLM-1003" in captured_interrupt_payload["question"]
    assert "reject" in captured_interrupt_payload["question"]

    recorded = result["messages"][0]
    assert isinstance(recorded, HumanMessage)
    assert "approved=False" in recorded.content
    assert "looks suspicious" in recorded.content


if __name__ == "__main__":
    import sys

    tests = [
        test_route_after_tools_ignores_a_stale_approval_request_from_earlier_in_the_thread,
        test_route_after_tools_triggers_on_explicit_approval_request,
        test_route_after_tools_does_not_trigger_on_plain_fraud_check_without_recommendation,
        test_should_continue_routes_to_tools_then_end,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"OK   {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")

    # The monkeypatch-based test needs a tiny stand-in when run without pytest.
    class _MonkeyPatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    try:
        test_human_check_builds_question_from_the_approval_payload_and_records_decision(_MonkeyPatch())
        print(f"OK   test_human_check_builds_question_from_the_approval_payload_and_records_decision")
    except AssertionError as e:
        failures += 1
        print(f"FAIL test_human_check_builds_question_from_the_approval_payload_and_records_decision: {e}")

    sys.exit(1 if failures else 0)
