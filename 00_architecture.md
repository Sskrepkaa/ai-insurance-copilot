# Архитектура проекта: AI Copilot для сотрудника поддержки страховой компании

## Общая идея
Сотрудник поддержки открывает дашборд, видит список claim'ов (заявок на выплату).
Кликает на claim -> справа открывается чат с AI-агентом, который может:
- посмотреть детали claim'а
- проверить риск мошенничества
- посмотреть историю клиента
- если риск высокий -> ПРИОСТАНОВИТЬСЯ и попросить сотрудника нажать "Одобрить/Отклонить"
После ответа агента, отдельный AI Judge молча проверяет ответ на "не выдумал ли факты"
и результат проверки виден в дашборде маленьким индикатором (зелёный/красный).

## Компоненты и порты (чтобы агенты не путались)
- Backend API (FastAPI): http://localhost:8000
- Frontend (React/Vite): http://localhost:5173
- База данных: SQLite файл claims.db (создаётся backend-агентом)
- LLM: Groq API (бесплатный ключ на console.groq.com), модель llama-3.3-70b-versatile
  (поддерживает tool calling, быстрая, бесплатный тир)

## Контракт API между Frontend и Backend (ЭТО ФИКСИРУЕТСЯ, чтобы агенты не расходились)

### GET /claims
Возвращает список всех claims для дашборда.
Response:
[
  {"id": "CLM-1001", "client": "Ivan Petrov", "amount": 12000, "status": "pending", "type": "Auto"},
  ...
]

### GET /claims/{claim_id}
Детали одного claim'а.
Response:
{"id": "CLM-1001", "client": "Ivan Petrov", "amount": 12000, "status": "pending",
 "type": "Auto", "description": "...", "fraud_risk": null}

### POST /chat
Body: {"claim_id": "CLM-1001", "message": "Проверь этот claim на мошенничество"}
Response (нормальный ответ):
{"type": "answer", "text": "...", "judge_score": {"groundedness": 0.95, "verdict": "pass"}}

Response (если агент остановился и ждёт решения человека - HITL):
{"type": "interrupt", "question": "Risk score 0.85 - подтвердить одобрение claim CLM-1001?",
 "thread_id": "CLM-1001"}

### POST /resume
Body: {"thread_id": "CLM-1001", "approved": true, "reason": "проверено вручную"}
Response: {"type": "answer", "text": "Claim одобрен. ...", "judge_score": {...}}

## Инструменты (tools) агента - 3 штуки, простые Python-функции внутри backend
1. get_claim_details(claim_id) -> данные claim'а из SQLite
2. check_fraud_risk(claim_id) -> фейковый расчёт risk_score по сумме и истории
3. get_customer_history(client_name) -> список предыдущих claims этого клиента

## Порядок разработки (чтобы за 2-3 часа собрать)
1. Backend+DB agent - делает SQLite, сидит данные, пишет 3 Python tool-функции, FastAPI эндпоинты
2. LangGraph agent - берёт tool-функции из backend, строит граф с tool calling + HITL узлом
3. AI Judge agent - пишет функцию judge_response(question, answer, tool_outputs) -> score
4. Frontend agent - React дашборд с списком claims, чатом, кнопками Approve/Reject
