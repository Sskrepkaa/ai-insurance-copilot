"""AI Judge module (03_prompt_ai_judge_agent.md): an independent LLM call
that grades the insurance copilot's answers for groundedness and relevance
*after* the main agent (agent.py) has already produced a final answer.

This is deliberately a SEPARATE model call from the one in agent.py - the
judge never sees the agent's reasoning/tool-calling process, only:
  - the employee's question (or resume context, for /resume)
  - the agent's final answer text
  - the raw tool_outputs the agent actually retrieved (get_claim_details,
    check_fraud_risk, get_customer_history)

The core check is groundedness: does every factual claim in the answer
(numbers, names, risk scores, history entries, dates, statuses...) trace
back to something literally present in tool_outputs? The judge is
explicitly instructed NOT to use its own general insurance knowledge to
"fill in" or validate facts - only tool_outputs counts as ground truth.
General reasoning/insurance terminology that doesn't assert a new fact
(e.g. "this looks like a routine auto claim", "please review before
approving") is fine and shouldn't hurt groundedness.

Like agent.py's ChatGroq usage, the LLM client is constructed lazily -
`import judge` must work with no GROQ_API_KEY set; the key is only needed
once `judge_response(...)` actually runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load backend/.env (GROQ_API_KEY=...) if present, same as agent.py. Never
# commit a real key.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

MODEL_NAME = "llama-3.3-70b-versatile"
PASS_THRESHOLD = 0.7

JUDGE_SYSTEM_PROMPT = (
    "You are an AI judge that controls the quality of an insurance support "
    "copilot's answers. You are given the employee's question, the copilot's "
    "final answer, and tool_outputs: the exact, real data the copilot's "
    "tools returned (get_claim_details, check_fraud_risk, "
    "get_customer_history).\n\n"
    "Score two criteria, each from 0.0 to 1.0:\n\n"
    "1. groundedness - does the answer contain ONLY facts that are "
    "actually present in tool_outputs? Treat tool_outputs as the ONLY "
    "source of truth - do NOT use your own general knowledge about "
    "insurance, this client, or typical claim amounts/risk scores to judge "
    "whether a fact is plausible or correct. Any concrete factual claim in "
    "the answer (a claim amount, a client name, a risk score, a status, a "
    "history entry, a date, a count of past claims, etc.) that is not "
    "traceable to tool_outputs must be treated as FABRICATED and should "
    "lower this score. If tool_outputs is empty and the answer still "
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
    except Exception as exc:  # noqa: BLE001 - judge must never crash the request
        return _fallback(f"judge LLM call failed: {exc}")

    return _parse_judge_output(raw)
