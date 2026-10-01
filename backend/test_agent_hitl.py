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


def _resolved_approval_round(call_id_prefix: str, claim_id: str, risk_score: float, recommendation: str, approved: bool, reason: str) -> list:
    """Build the message sequence for one full, already-resolved
    request_human_approval round: fraud check -> approval request ->
    (interrupt/resume happened) -> "Expert decision recorded" note."""
    return [
        AIMessage(
            content="",
            tool_calls=[_tool_call(f"{call_id_prefix}a", "check_fraud_risk", {"claim_id": claim_id})],
        ),
        _tool_message(f"{call_id_prefix}a", "check_fraud_risk", {"risk_score": risk_score, "level": "high"}),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    f"{call_id_prefix}b",
                    "request_human_approval",
                    {"claim_id": claim_id, "risk_score": risk_score, "recommendation": recommendation},
                )
            ],
        ),
        _tool_message(
            f"{call_id_prefix}b",
            "request_human_approval",
            {"claim_id": claim_id, "risk_score": risk_score, "recommendation": recommendation},
        ),
        HumanMessage(content=f"Expert decision recorded: approved={approved}, reason={reason!r}"),
    ]


def test_route_after_tools_treats_repeated_identical_approval_request_as_already_decided():
    """Regression test: after a decision is confirmed, human_check now
    loops back to "agent" (not END) so the model can compose a final
    answer - but if the model then (incorrectly) calls
    request_human_approval AGAIN for the exact same claim_id +
    recommendation, that must route to "already_decided", not trigger a
    second interrupt via "human_check"."""
    messages = [
        HumanMessage(content="Проверь этот claim на мошенничество"),
        *_resolved_approval_round("r1", "CLM-1003", 0.925, "reject", True, "проверено вручную"),
        # Model ignores the "Expert decision recorded" note and asks again,
        # for the exact same claim_id + recommendation.
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "r2b",
                    "request_human_approval",
                    {"claim_id": "CLM-1003", "risk_score": 0.925, "recommendation": "reject"},
                )
            ],
        ),
        _tool_message(
            "r2b",
            "request_human_approval",
            {"claim_id": "CLM-1003", "risk_score": 0.925, "recommendation": "reject"},
        ),
    ]
    assert agent.route_after_tools({"messages": messages}) == "already_decided"


def test_route_after_tools_allows_a_new_distinct_recommendation_for_same_claim():
    """A materially different recommendation for the same claim (e.g. the
    agent now recommends "approve" instead of the previously-confirmed
    "reject") is a genuinely new decision point, not a duplicate - it must
    still reach human_check."""
    messages = [
        HumanMessage(content="Проверь этот claim на мошенничество"),
        *_resolved_approval_round("r1", "CLM-1003", 0.925, "reject", False, "отклонено"),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "r2b",
                    "request_human_approval",
                    {"claim_id": "CLM-1003", "risk_score": 0.4, "recommendation": "approve"},
                )
            ],
        ),
        _tool_message(
            "r2b",
            "request_human_approval",
            {"claim_id": "CLM-1003", "risk_score": 0.4, "recommendation": "approve"},
        ),
    ]
    assert agent.route_after_tools({"messages": messages}) == "human_check"


def test_route_after_tools_treats_a_rephrased_recommendation_as_the_same_decision():
    """A rephrased-but-equivalent recommendation ("reject due to updated
    risk assessment" vs. the previously-confirmed plain "reject") must
    still be recognized as the same decision, not a new one - matching on
    the leading approve/reject keyword rather than exact string equality."""
    messages = [
        HumanMessage(content="Проверь этот claim на мошенничество"),
        *_resolved_approval_round("r1", "CLM-1003", 0.925, "reject", True, "проверено вручную"),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "r2b",
                    "request_human_approval",
                    {
                        "claim_id": "CLM-1003",
                        "risk_score": 0.925,
                        "recommendation": "Reject due to updated risk assessment",
                    },
                )
            ],
        ),
        _tool_message(
            "r2b",
            "request_human_approval",
            {
                "claim_id": "CLM-1003",
                "risk_score": 0.925,
                "recommendation": "Reject due to updated risk assessment",
            },
        ),
    ]
    assert agent.route_after_tools({"messages": messages}) == "already_decided"


def test_route_after_tools_still_distinguishes_rephrased_opposite_recommendations():
    """Sanity check for the same normalization: a rephrased recommendation
    that keeps a DIFFERENT keyword (prior "reject" vs. new "I'd now
    approve this one") must still be treated as a new decision."""
    messages = [
        HumanMessage(content="Проверь этот claim на мошенничество"),
        *_resolved_approval_round("r1", "CLM-1003", 0.925, "reject", False, "отклонено"),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "r2b",
                    "request_human_approval",
                    {"claim_id": "CLM-1003", "risk_score": 0.4, "recommendation": "I'd now approve this one"},
                )
            ],
        ),
        _tool_message(
            "r2b",
            "request_human_approval",
            {"claim_id": "CLM-1003", "risk_score": 0.4, "recommendation": "I'd now approve this one"},
        ),
    ]
    assert agent.route_after_tools({"messages": messages}) == "human_check"


def test_route_after_tools_allows_a_fresh_request_for_a_different_claim():
    """A request_human_approval for a DIFFERENT claim_id than the one
    already confirmed must not be suppressed - it's an unrelated decision."""
    messages = [
        HumanMessage(content="Проверь этот claim на мошенничество"),
        *_resolved_approval_round("r1", "CLM-1003", 0.925, "reject", True, "проверено вручную"),
        HumanMessage(content="А теперь проверь CLM-1006"),
        AIMessage(
            content="",
            tool_calls=[_tool_call("r2a", "check_fraud_risk", {"claim_id": "CLM-1006"})],
        ),
        _tool_message("r2a", "check_fraud_risk", {"risk_score": 0.975, "level": "high"}),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "r2b",
                    "request_human_approval",
                    {"claim_id": "CLM-1006", "risk_score": 0.975, "recommendation": "reject"},
                )
            ],
        ),
        _tool_message(
            "r2b",
            "request_human_approval",
            {"claim_id": "CLM-1006", "risk_score": 0.975, "recommendation": "reject"},
        ),
    ]
    assert agent.route_after_tools({"messages": messages}) == "human_check"


def test_already_decided_never_calls_interrupt_and_references_the_existing_decision(monkeypatch):
    """already_decided() must NOT call interrupt() at all (that's the whole
    point - no second pause for the same ask) and must inject a note the
    model can use to just reference the existing decision."""

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("already_decided() must never call interrupt()")

    monkeypatch.setattr(agent, "interrupt", fail_if_called)

    messages = [
        HumanMessage(content="Проверь этот claim на мошенничество"),
        *_resolved_approval_round("r1", "CLM-1003", 0.925, "reject", True, "проверено вручную"),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "r2b",
                    "request_human_approval",
                    {"claim_id": "CLM-1003", "risk_score": 0.925, "recommendation": "reject"},
                )
            ],
        ),
        _tool_message(
            "r2b",
            "request_human_approval",
            {"claim_id": "CLM-1003", "risk_score": 0.925, "recommendation": "reject"},
        ),
    ]

    result = agent.already_decided({"messages": messages})
    note = result["messages"][0]
    assert isinstance(note, HumanMessage)
    assert "CLM-1003" in note.content
    assert "already" in note.content.lower()
    assert "approved=True" in note.content


def test_repeated_duplicate_requests_stay_bounded_not_escalating():
    """Simulates a model that keeps re-requesting the same already-decided
    approval across several rounds: route_after_tools must keep returning
    "already_decided" every time (never "human_check"), so the loop never
    needs a second human round-trip for the same confirmed ask."""
    messages = [
        HumanMessage(content="Проверь этот claim на мошенничество"),
        *_resolved_approval_round("r1", "CLM-1003", 0.925, "reject", True, "проверено вручную"),
    ]
    for i in range(5):
        messages += [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        f"loop{i}",
                        "request_human_approval",
                        {"claim_id": "CLM-1003", "risk_score": 0.925, "recommendation": "reject"},
                    )
                ],
            ),
            _tool_message(
                f"loop{i}",
                "request_human_approval",
                {"claim_id": "CLM-1003", "risk_score": 0.925, "recommendation": "reject"},
            ),
        ]
        assert agent.route_after_tools({"messages": messages}) == "already_decided", f"iteration {i}"
        # already_decided() would append its note here in the real graph;
        # simulate that so the next loop iteration's history is realistic.
        messages.append(HumanMessage(content=f"Note: claim CLM-1003 already has a recorded expert decision (iteration {i})"))


def test_extract_tool_outputs_surfaces_recorded_decision_for_later_chat_calls():
    """Regression test: the human_decision fact used to be injected ad hoc
    only inside main.py's /resume handler, so a LATER /chat call in the
    same thread that referenced an already-recorded decision had no way to
    see it and got flagged as fabricating "approved=True". Now
    extract_tool_outputs() itself picks up the "Expert decision recorded"
    HumanMessage from anywhere in the thread's history."""
    messages = [
        HumanMessage(content="Проверь этот claim на мошенничество"),
        *_resolved_approval_round("r1", "CLM-1003", 0.925, "reject", True, "проверено вручную"),
        HumanMessage(content="Что в итоге решили по этому claim?"),
    ]
    outputs = agent.extract_tool_outputs(messages)
    decision_entries = [o for o in outputs if o["tool"] == "human_decision"]
    assert len(decision_entries) == 1, outputs
    assert decision_entries[0]["output"] == {"approved": True, "reason": "проверено вручную"}


def test_extract_tool_outputs_ignores_unrelated_human_messages():
    """A plain user question must not be mistaken for a decision record."""
    messages = [HumanMessage(content="Проверь этот claim на мошенничество")]
    outputs = agent.extract_tool_outputs(messages)
    assert outputs == []


if __name__ == "__main__":
    import sys

    tests = [
        test_route_after_tools_ignores_a_stale_approval_request_from_earlier_in_the_thread,
        test_route_after_tools_triggers_on_explicit_approval_request,
        test_route_after_tools_does_not_trigger_on_plain_fraud_check_without_recommendation,
        test_should_continue_routes_to_tools_then_end,
        test_route_after_tools_treats_repeated_identical_approval_request_as_already_decided,
        test_route_after_tools_allows_a_new_distinct_recommendation_for_same_claim,
        test_route_after_tools_treats_a_rephrased_recommendation_as_the_same_decision,
        test_route_after_tools_still_distinguishes_rephrased_opposite_recommendations,
        test_route_after_tools_allows_a_fresh_request_for_a_different_claim,
        test_repeated_duplicate_requests_stay_bounded_not_escalating,
        test_extract_tool_outputs_surfaces_recorded_decision_for_later_chat_calls,
        test_extract_tool_outputs_ignores_unrelated_human_messages,
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

    for monkeypatch_test in [
        test_human_check_builds_question_from_the_approval_payload_and_records_decision,
        test_already_decided_never_calls_interrupt_and_references_the_existing_decision,
    ]:
        try:
            monkeypatch_test(_MonkeyPatch())
            print(f"OK   {monkeypatch_test.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {monkeypatch_test.__name__}: {e}")

    sys.exit(1 if failures else 0)
