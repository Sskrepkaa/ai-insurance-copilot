---
name: langgraph-agent
description: Adds LangGraph tool-calling and a human-in-the-loop (HITL) node to the existing FastAPI backend of the AI Insurance Copilot project — creates backend/agent.py and wires POST /chat and POST /resume into backend/main.py using Groq's llama-3.3-70b-versatile. Requires backend-agent to have already created backend/tools.py, db.py and main.py. Do not use for the initial DB/tools setup, the groundedness judge, or frontend work.
tools: Read, Write, Edit, Bash, Glob, Grep, WebFetch, WebSearch
isolation: worktree
---

You extend an existing backend (folder `backend/`, already containing `main.py`, `tools.py`, `db.py` from a prior agent) with a new module `agent.py`, and add `/chat` and `/resume` endpoints to `main.py`. Read `00_architecture.md` at the repo root for the full project contract before starting.

## Context
`tools.py` already has 3 functions: `get_claim_details(claim_id)`, `check_fraud_risk(claim_id)`, `get_customer_history(client)`. Wrap them as LangChain `@tool`s and build a LangGraph graph that uses a real LLM (Groq API) for tool calling.

## 1. Dependencies
Add to `backend/requirements.txt`:
```
langgraph
langchain-groq
langchain-core
```
Get a free API key at console.groq.com and put it in `backend/.env`:
```
GROQ_API_KEY=your_key
```

## 2. agent.py — required logic (sketch below, adapt to the current API)
```python
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
import tools as backend_tools

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

@tool
def get_claim_details(claim_id: str) -> dict:
    """Full claim data (client, amount, status, type, description) by ID."""
    return backend_tools.get_claim_details(claim_id)

@tool
def check_fraud_risk(claim_id: str) -> dict:
    """Fraud risk_score and level for a claim."""
    return backend_tools.check_fraud_risk(claim_id)

@tool
def get_customer_history(client: str) -> list:
    """Past claims for a client by name."""
    return backend_tools.get_customer_history(client)

all_tools = [get_claim_details, check_fraud_risk, get_customer_history]
llm_with_tools = llm.bind_tools(all_tools)

def call_model(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "human_check"

def human_check(state: MessagesState):
    # Look for the last ToolMessage from check_fraud_risk
    for msg in reversed(state["messages"]):
        if getattr(msg, "name", None) == "check_fraud_risk":
            import json
            result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            if result.get("risk_score", 0) > 0.6:
                decision = interrupt({
                    "question": f"Risk score {result['risk_score']} detected. Confirm the expert's decision?",
                })
                return {"messages": [{"role": "user", "content": f"Expert decision: {decision}"}]}
    return {"messages": []}

builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(all_tools))
builder.add_node("human_check", human_check)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "human_check": "human_check"})
builder.add_edge("tools", "agent")
builder.add_edge("human_check", END)

checkpointer = SqliteSaver.from_conn_string("agent_state.db")
graph = builder.compile(checkpointer=checkpointer)
```

The code above is a sketch, not something to copy verbatim — the current LangGraph API may differ, especially `SqliteSaver` construction and `interrupt()`. Use WebFetch/WebSearch to check the current langgraph.io docs and the installed package version, and adapt accordingly. The behavior that must be preserved: the agent calls tools via the LLM, and when `risk_score > 0.6` the graph stops and waits for a human decision.

## 3. main.py additions — strictly per contract

`POST /chat` body `{"claim_id": str, "message": str}`:
- invoke the graph with `thread_id = claim_id`
- if the result contains an interrupt → return `{"type": "interrupt", "question": ..., "thread_id": claim_id}`
- else → return `{"type": "answer", "text": <agent's final message>}`

`POST /resume` body `{"thread_id": str, "approved": bool, "reason": str}`:
- resume the graph with `Command(resume={"approved": approved, "reason": reason})`, config `thread_id`
- return `{"type": "answer", "text": ...}`

Leave a `judge_score` field out for now — a later agent (judge-agent) will add it to these responses.

## Verification
```
curl -X POST http://localhost:8000/chat -d '{"claim_id":"CLM-1001","message":"Check for fraud"}'
```
Should return either a final answer or an interrupt asking the human expert.
