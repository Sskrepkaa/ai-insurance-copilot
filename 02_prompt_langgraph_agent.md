# ПРОМПТ ДЛЯ АГЕНТА 2: LangGraph Agent + tool calling + HITL

Ты дополняешь существующий backend (папка backend/, там уже есть main.py, tools.py, db.py)
новым модулем: agent.py + добавляешь эндпоинты /chat и /resume в main.py.

## Контекст
В tools.py уже есть 3 функции: get_claim_details(claim_id), check_fraud_risk(claim_id),
get_customer_history(client). Ты должен обернуть их в LangChain @tool и построить
LangGraph-граф, который использует настоящую LLM (Groq API) для tool calling.

## 1. Установи зависимости (добавь в requirements.txt backend'а):
langgraph
langchain-groq
langchain-core

## 2. Получи бесплатный API-ключ на console.groq.com, положи в .env файл:
GROQ_API_KEY=твой_ключ

## 3. Создай agent.py со следующей логикой:

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
    """Получить полные данные claim'а (клиент, сумма, статус, тип, описание) по ID."""
    return backend_tools.get_claim_details(claim_id)

@tool
def check_fraud_risk(claim_id: str) -> dict:
    """Рассчитать risk_score мошенничества для claim'а. Возвращает risk_score и level."""
    return backend_tools.check_fraud_risk(claim_id)

@tool
def get_customer_history(client: str) -> list:
    """Получить историю прошлых claims клиента по имени."""
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
    # ВАЖНО: проверяем, не пора ли остановиться и спросить человека
    # (смотрим, был ли в последних сообщениях вызов check_fraud_risk с высоким risk)
    return "human_check"

def human_check(state: MessagesState) -> str:
    # Ищем последний ToolMessage от check_fraud_risk
    for msg in reversed(state["messages"]):
        if getattr(msg, "name", None) == "check_fraud_risk":
            import json
            result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            if result.get("risk_score", 0) > 0.6:
                decision = interrupt({
                    "question": f"Risk score {result['risk_score']} обнаружен. Подтвердить действие эксперта?",
                })
                # decision содержит {"approved": bool, "reason": str} от человека
                return {"messages": [{"role": "user", "content": f"Эксперт решил: {decision}"}]}
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

ПРИМЕЧАНИЕ ДЛЯ АГЕНТА-ИСПОЛНИТЕЛЯ: код выше - это набросок идеи, не копируй буквально.
Актуальный API LangGraph может отличаться (проверь текущую документацию langgraph.io) -
особенно синтаксис SqliteSaver и interrupt(). Разберись и адаптируй под текущую версию,
установленную через pip. Главное сохранить логику: агент вызывает tools через LLM,
и если risk_score > 0.6 - граф останавливается и ждёт человека.

## 4. Добавь в main.py (backend) два эндпоинта СТРОГО по контракту:

POST /chat  body: {"claim_id": str, "message": str}
- invoke граф с thread_id = claim_id
- если результат содержит __interrupt__ -> вернуть {"type": "interrupt", "question": ..., "thread_id": claim_id}
- иначе -> вернуть {"type": "answer", "text": последнее сообщение агента}

POST /resume  body: {"thread_id": str, "approved": bool, "reason": str}
- graph.invoke(Command(resume={"approved": approved, "reason": reason}), config={"thread_id": ...})
- вернуть {"type": "answer", "text": ...}

## Проверка
curl -X POST http://localhost:8000/chat -d '{"claim_id":"CLM-1001","message":"Проверь мошенничество"}'
Должен либо вернуть ответ, либо interrupt с вопросом эксперту.
