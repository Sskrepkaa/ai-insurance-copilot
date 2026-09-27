"""Unit tests for judge.py that never touch the real Groq API.

Run with: python3 -m pytest test_judge.py -v
(or: python3 test_judge.py, which runs the same cases without pytest)

These verify, per the AI Judge task spec (03_prompt_ai_judge_agent.md):
  (a) a well-grounded answer -> a clean dict with groundedness/verdict
  (b) malformed/non-JSON LLM output -> the {"groundedness": None,
      "verdict": "error"} fallback, not a crash
  (c) the prompt actually embeds the real tool_outputs content (not a
      placeholder), so the judge LLM call could actually do the comparison
  plus a couple of extra edge cases (missing GROQ key never needed at
  import time; a badly-grounded/fabricated answer scoring low -> "fail").
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import judge


class _FakeLLM:
    """Stands in for the object judge._get_judge_llm() returns. Records the
    messages it was invoked with and replays a canned response, mirroring
    the .invoke(messages) -> AIMessage(content=...) shape ChatGroq exposes.
    """

    def __init__(self, raw_response: str):
        self._raw_response = raw_response
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return SimpleNamespace(content=self._raw_response)


def _run_with_fake_llm(raw_response: str, question: str, answer: str, tool_outputs: list[dict]):
    fake_llm = _FakeLLM(raw_response)
    with patch.object(judge, "_get_judge_llm", return_value=fake_llm):
        result = judge.judge_response(question=question, answer=answer, tool_outputs=tool_outputs)
    return result, fake_llm


def test_well_grounded_answer_parses_cleanly():
    tool_outputs = [
        {"tool": "get_claim_details", "output": {"id": "CLM-1001", "client": "Ivan Petrov", "amount": 12000}},
        {"tool": "check_fraud_risk", "output": {"risk_score": 0.6, "level": "medium"}},
    ]
    judge_raw = json.dumps(
        {
            "groundedness": 0.98,
            "relevance": 0.95,
            "verdict": "pass",
            "reason": "All facts (amount 12000, risk_score 0.6) match tool_outputs.",
        }
    )
    result, _ = _run_with_fake_llm(
        judge_raw,
        question="What's the fraud risk on CLM-1001?",
        answer="Claim CLM-1001 (Ivan Petrov, $12000) has a medium fraud risk of 0.6.",
        tool_outputs=tool_outputs,
    )
    assert result["groundedness"] == 0.98
    assert result["relevance"] == 0.95
    assert result["verdict"] == "pass"
    assert "reason" in result
    print("test_well_grounded_answer_parses_cleanly: PASS ->", result)


def test_fabricated_answer_scores_low_and_fails():
    tool_outputs = [
        {"tool": "get_claim_details", "output": {"id": "CLM-1001", "client": "Ivan Petrov", "amount": 12000}},
    ]
    # Judge (correctly) notices the $50000 and risk score are not in tool_outputs.
    judge_raw = json.dumps(
        {
            "groundedness": 0.1,
            "relevance": 0.9,
            "verdict": "fail",
            "reason": "Amount $50000 and risk_score 0.95 are not in tool_outputs; fabricated.",
        }
    )
    result, _ = _run_with_fake_llm(
        judge_raw,
        question="What's the fraud risk on CLM-1001?",
        answer="Claim CLM-1001 is $50000 with a fraud risk score of 0.95.",
        tool_outputs=tool_outputs,
    )
    assert result["groundedness"] == 0.1
    assert result["verdict"] == "fail"
    print("test_fabricated_answer_scores_low_and_fails: PASS ->", result)


def test_verdict_is_rederived_from_scores_not_trusted_verbatim():
    # Judge says "pass" but its own scores are below threshold - our code
    # must recompute the verdict rather than trust the LLM's field as-is.
    judge_raw = json.dumps({"groundedness": 0.5, "relevance": 0.9, "verdict": "pass"})
    result, _ = _run_with_fake_llm(judge_raw, "q", "a", [])
    assert result["verdict"] == "fail", "0.5 groundedness is below the 0.7 threshold"
    print("test_verdict_is_rederived_from_scores_not_trusted_verbatim: PASS ->", result)


def test_malformed_json_falls_back_gracefully():
    result, _ = _run_with_fake_llm(
        "Sure, I think this answer is mostly grounded! No JSON here though.",
        "q",
        "a",
        [],
    )
    assert result["groundedness"] is None
    assert result["verdict"] == "error"
    print("test_malformed_json_falls_back_gracefully: PASS ->", result)


def test_json_wrapped_in_markdown_fence_still_parses():
    fenced = "```json\n" + json.dumps({"groundedness": 0.8, "relevance": 0.8, "verdict": "pass"}) + "\n```"
    result, _ = _run_with_fake_llm(fenced, "q", "a", [])
    assert result["groundedness"] == 0.8
    assert result["verdict"] == "pass"
    print("test_json_wrapped_in_markdown_fence_still_parses: PASS ->", result)


def test_llm_call_raising_is_caught_not_crashed():
    with patch.object(judge, "_get_judge_llm", side_effect=RuntimeError("no GROQ_API_KEY set")):
        result = judge.judge_response(question="q", answer="a", tool_outputs=[])
    assert result["groundedness"] is None
    assert result["verdict"] == "error"
    print("test_llm_call_raising_is_caught_not_crashed: PASS ->", result)


def test_prompt_actually_contains_real_tool_outputs_not_a_placeholder():
    tool_outputs = [
        {"tool": "get_claim_details", "output": {"id": "CLM-9999", "client": "Zzyzx Q. Marker", "amount": 424242}},
        {"tool": "get_customer_history", "output": [{"claim_id": "CLM-0001", "outcome": "approved"}]},
    ]
    judge_raw = json.dumps({"groundedness": 1.0, "relevance": 1.0, "verdict": "pass"})
    result, fake_llm = _run_with_fake_llm(
        judge_raw,
        question="Tell me about CLM-9999",
        answer="CLM-9999 belongs to Zzyzx Q. Marker, amount $424242.",
        tool_outputs=tool_outputs,
    )
    # The 2nd message (HumanMessage) is the user prompt carrying question/answer/tool_outputs.
    sent_prompt = fake_llm.last_messages[1].content
    assert "Zzyzx Q. Marker" in sent_prompt, "real client name from tool_outputs must be in the prompt"
    assert "424242" in sent_prompt, "real amount from tool_outputs must be in the prompt"
    assert "CLM-0001" in sent_prompt, "real history entry from tool_outputs must be in the prompt"
    assert "get_claim_details" in sent_prompt
    assert "get_customer_history" in sent_prompt
    # And the system prompt must state the "only tool_outputs counts" rule.
    system_prompt = fake_llm.last_messages[0].content
    assert "tool_outputs" in system_prompt
    assert "fabricat" in system_prompt.lower()
    print("test_prompt_actually_contains_real_tool_outputs_not_a_placeholder: PASS")


def test_import_judge_requires_no_groq_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import importlib

    importlib.reload(judge)  # must not raise even with no key set
    print("test_import_judge_requires_no_groq_api_key: PASS")


if __name__ == "__main__":
    # Allow running without pytest installed.
    test_well_grounded_answer_parses_cleanly()
    test_fabricated_answer_scores_low_and_fails()
    test_verdict_is_rederived_from_scores_not_trusted_verbatim()
    test_malformed_json_falls_back_gracefully()
    test_json_wrapped_in_markdown_fence_still_parses()
    test_llm_call_raising_is_caught_not_crashed()
    test_prompt_actually_contains_real_tool_outputs_not_a_placeholder()
    print("\nAll judge.py tests passed.")
