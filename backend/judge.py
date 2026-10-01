"""AI Judge module (03_prompt_ai_judge_agent.md): an independent LLM call
that grades the insurance copilot's answers for groundedness and relevance
*after* the main agent (agent.py) has already produced a final answer.

This is deliberately a SEPARATE model call from the one in agent.py - the
judge never sees the agent's reasoning/tool-calling process, only:
  - the employee's question (or resume context, for /resume)
  - the agent's final answer text
  - the raw tool_outputs the agent actually retrieved (get_claim_details,
    check_fraud_risk, get_customer_history, plus request_human_approval and
    human_decision - see the system prompt for how those last two differ)

The core check is groundedness: does every factual claim in the answer
(numbers, names, risk scores, history entries, dates, statuses...) trace
back to something literally present in tool_outputs? The judge is
explicitly instructed NOT to use its own general insurance knowledge to
"fill in" or validate facts - only tool_outputs counts as ground truth.
General reasoning/insurance terminology that doesn't assert a new fact
(e.g. "this looks like a routine auto claim", "please review before
approving") is fine and shouldn't hurt groundedness.

On top of the LLM's own judgment, `judge_response()` also runs a cheap,
deterministic check: every "CLM-####"-shaped id mentioned in the answer
must appear somewhere in tool_outputs (in any call's args or output,
recursively - e.g. inside a get_customer_history list entry). If the
answer mentions an id that isn't grounded anywhere in tool_outputs, the
verdict is forced to "fail" regardless of what the LLM judge said - this
catches the specific, high-confidence case of an outright invented claim
id even if the LLM judge itself gets fooled.

Like agent.py's ChatGroq usage, the LLM client is constructed lazily -
`import judge` must work with no GROQ_API_KEY set; the key is only needed
once `judge_response(...)` actually runs.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load backend/.env (GROQ_API_KEY=...) if present, same as agent.py. Never
# commit a real key.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

MODEL_NAME = "openai/gpt-oss-120b"
PASS_THRESHOLD = 0.7

JUDGE_SYSTEM_PROMPT = (
    "You are an AI judge that controls the quality of an insurance support "
    "copilot's answers. You are given the employee's question, the copilot's "
    "final answer, and tool_outputs: a list of every tool call the copilot "
    "made, each with the exact arguments it was called with (\"args\") and "
    "the exact, real data it returned (\"output\") - from "
    "get_claim_details, check_fraud_risk, get_customer_history, and "
    "sometimes two special entries:\n"
    "  - \"request_human_approval\": this is the AGENT'S OWN PROPOSED "
    "recommendation (e.g. \"reject\"), not a confirmed fact about the "
    "claim. The answer saying something like \"I recommend rejecting this "
    "claim, pending your confirmation\" is fine. But if the answer instead "
    "asserts the decision is already made/final (e.g. \"this claim has "
    "been rejected\") and tool_outputs has NO \"human_decision\" entry, "
    "that is an ungrounded/fabricated status claim and must lower "
    "groundedness.\n"
    "  - \"human_decision\": this IS a confirmed fact - the real outcome "
    "an actual human expert entered via the dashboard's Approve/Reject "
    "buttons (approved: bool, reason: str). Stating the claim's final "
    "outcome is grounded when this entry is present.\n\n"
    "Score two criteria, each from 0.0 to 1.0:\n\n"
    "1. groundedness - does the answer contain ONLY facts that are "
    "actually present in tool_outputs (either in a call's \"args\" or its "
    "\"output\")? For example, if check_fraud_risk was called with "
    "args={\"claim_id\": \"CLM-1002\"}, then the answer stating facts about "
    "claim CLM-1002 is grounded even though the tool's own output doesn't "
    "repeat the id - the id came from a real, actually-made tool call, not "
    "from the model's imagination. Treat tool_outputs as the ONLY source of "
    "truth - do NOT use your own general knowledge about insurance, this "
    "client, or typical claim amounts/risk scores to judge whether a fact "
    "is plausible or correct. Any concrete factual claim in the answer (a "
    "claim amount, a client name, a risk score, a status, a history entry, "
    "a date, a count of past claims, etc.) that is not traceable to either "
    "field of tool_outputs must be treated as FABRICATED and should lower "
    "this score. If tool_outputs is empty and the answer still "
    "states specific facts/numbers, that is fabrication (low score). "
    "General insurance reasoning, explanations, recommendations, or "
    "process/etiquette language that does NOT assert a new, specific "
    "fact (e.g. 'please double check with the client', 'this is standard "
    "procedure for auto claims') is NOT fabrication and should not lower "
    "the score by itself.\n\n"
    "2. relevance - does the answer actually address the employee's "
    "question?\n\n"
    "Return ONLY valid JSON, with no markdown fences and no explanation "
    "outside the JSON object, in exactly this shape:\n"
    '{"groundedness": 0.0-1.0, "relevance": 0.0-1.0, "verdict": '
    '"pass"|"fail", "reason": "brief explanation"}\n\n'
    "verdict = \"fail\" if groundedness < 0.7 OR relevance < 0.7, else "
    "\"pass\"."
)


def _build_user_prompt(question: str, answer: str, tool_outputs: list[dict]) -> str:
    """Render the question/answer/tool_outputs the judge must compare.

    tool_outputs is serialized as real JSON (not a placeholder) so the
    judge prompt actually contains the concrete facts to check the answer
    against.
    """
    tool_outputs_json = json.dumps(tool_outputs, indent=2, default=str, ensure_ascii=False)
    return (
        f"Question from the employee:\n{question}\n\n"
        f"Copilot's final answer:\n{answer}\n\n"
        f"tool_outputs (the ONLY ground truth - the real data the copilot's "
        f"tools returned):\n{tool_outputs_json}"
    )


# --------------------------------------------------------------------------
# LLM - constructed lazily so `import judge` never requires GROQ_API_KEY,
# mirroring agent.py's `_get_llm_with_tools()`.
# --------------------------------------------------------------------------

_judge_llm = None


def _get_judge_llm():
    global _judge_llm
    if _judge_llm is None:
        from langchain_groq import ChatGroq

        _judge_llm = ChatGroq(model=MODEL_NAME, temperature=0)
    return _judge_llm


# Matches "CLM-1001"-shaped ids. The dash covers plain ASCII "-" plus the
# Unicode dash variants an LLM sometimes substitutes in prose (e.g.
# "CLM‑1001" with U+2011 non-breaking hyphen) so formatting quirks
# don't let a real, grounded id slip past the check as "not found".
_CLAIM_ID_PATTERN = re.compile(r"CLM[-‐‑‒–—−]\d{3,6}", re.IGNORECASE)
_DASH_VARIANTS = re.compile(r"[‐‑‒–—−]")


def _extract_claim_ids(text: Any) -> set[str]:
    if not isinstance(text, str):
        return set()
    return {_DASH_VARIANTS.sub("-", m).upper() for m in _CLAIM_ID_PATTERN.findall(text)}


def _iter_strings(value: Any):
    """Walk a (possibly nested) tool_outputs value and yield every string
    in it, so claim ids inside e.g. a get_customer_history list entry are
    found just as readily as ones in a top-level field."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_strings(v)


def _grounded_claim_ids(tool_outputs: list[dict]) -> set[str]:
    """Every claim id mentioned anywhere in tool_outputs (args or output,
    at any nesting depth) - the full set of ids the answer is allowed to
    reference."""
    ids: set[str] = set()
    for entry in tool_outputs or []:
        for s in _iter_strings(entry):
            ids |= _extract_claim_ids(s)
    return ids


def _apply_claim_id_check(result: dict, answer: str, tool_outputs: list[dict]) -> dict:
    """Deterministic safety net: force verdict="fail" if the answer
    mentions a claim id that doesn't appear anywhere in tool_outputs. This
    runs regardless of what the LLM judge concluded (even on a parse
    failure/"error" result) because it doesn't depend on the LLM at all -
    it exists specifically for the case where the LLM judge itself misses
    an outright invented id."""
    answer_ids = _extract_claim_ids(answer)
    if not answer_ids:
        return result

    hallucinated = answer_ids - _grounded_claim_ids(tool_outputs)
    if not hallucinated:
        return result

    result = dict(result)
    groundedness = result.get("groundedness")
    result["groundedness"] = 0.0 if groundedness is None else min(groundedness, 0.2)
    result["verdict"] = "fail"
    extra = f"[deterministic check] answer mentions claim id(s) not found in tool_outputs: {sorted(hallucinated)}"
    existing_reason = result.get("reason")
    result["reason"] = f"{existing_reason} {extra}" if existing_reason else extra
    return result


def _fallback(reason: str = "judge output could not be parsed") -> dict:
    """Sane fallback shape when the judge's LLM call fails or returns
    something we can't parse as the expected JSON. Always keeps
    groundedness/verdict present, per the 00_architecture.md contract."""
    return {"groundedness": None, "relevance": None, "verdict": "error", "reason": reason}


def _clamp01(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN check
        return None
    return max(0.0, min(1.0, f))


def _extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from the judge's raw text:
    handles a clean JSON response, one wrapped in ```json ... ``` fences,
    or one with stray text around the { ... } object."""
    text = text.strip()

    # Strip a markdown code fence if the model added one despite instructions.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _parse_judge_output(raw: str) -> dict:
    """Parse the judge's raw text into our result dict, defensively.

    We deliberately RE-DERIVE the pass/fail verdict from the parsed
    groundedness/relevance scores ourselves (rather than trusting the
    judge's own "verdict" field verbatim) so the pass/0.7 threshold rule
    is applied consistently even if the judge's arithmetic is off.
    """
    data = _extract_json_object(raw)
    if data is None:
        return _fallback(f"non-JSON judge output: {raw[:200]!r}")

    groundedness = _clamp01(data.get("groundedness"))
    relevance = _clamp01(data.get("relevance"))

    if groundedness is not None and relevance is not None:
        verdict = "pass" if (groundedness >= PASS_THRESHOLD and relevance >= PASS_THRESHOLD) else "fail"
    else:
        # Missing/unparseable scores - fall back to whatever verdict (if
        # any) the judge itself returned, else mark it an error.
        raw_verdict = data.get("verdict")
        verdict = raw_verdict if raw_verdict in ("pass", "fail") else "error"

    result: dict[str, Any] = {
        "groundedness": groundedness,
        "relevance": relevance,
        "verdict": verdict,
    }
    reason = data.get("reason")
    if isinstance(reason, str) and reason:
        result["reason"] = reason
    return result


def judge_response(question: str, answer: str, tool_outputs: list[dict]) -> dict:
    """Grade `answer` (the copilot's final answer to `question`) for
    groundedness against `tool_outputs` and relevance to `question`, via a
    separate Groq LLM call acting as judge.

    Always returns a dict shaped at least like
    {"groundedness": float | None, "verdict": "pass" | "fail" | "error"}
    (plus "relevance" and "reason" when available) - never raises, even if
    the Groq call fails or returns malformed output.
    """
    user_prompt = _build_user_prompt(question, answer, tool_outputs)

    try:
        llm = _get_judge_llm()
        from langchain_core.messages import HumanMessage, SystemMessage

        response = llm.invoke(
            [SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
        )
        raw = response.content if hasattr(response, "content") else str(response)
        if not isinstance(raw, str):
            raw = str(raw)
        result = _parse_judge_output(raw)
    except Exception as exc:  # noqa: BLE001 - judge must never crash the request
        result = _fallback(f"judge LLM call failed: {exc}")

    # Applied even when the LLM call/parse failed: it's a pure text/data
    # check that doesn't need the LLM judge to have worked at all, so it
    # still catches an outright invented claim id as a last resort.
    return _apply_claim_id_check(result, answer, tool_outputs)
