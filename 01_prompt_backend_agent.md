# ПРОМПТ ДЛЯ АГЕНТА 1: Backend + DB

Ты пишешь Python backend на FastAPI для демо-проекта "AI Copilot для страховой поддержки".

## Задача
Создай проект в папке backend/ со следующей структурой:
backend/
  main.py          - FastAPI приложение
  db.py            - работа с SQLite
  tools.py         - 3 tool-функции для агента
  seed_data.py     - скрипт заполнения фейковыми данными
  requirements.txt

## 1. База данных (SQLite через sqlite3 или sqlalchemy, на твой выбор, проще - sqlite3 напрямую)

Таблица claims:
- id (TEXT, PK) - например "CLM-1001"
- client (TEXT) - имя клиента
- amount (REAL) - сумма претензии
- status (TEXT) - "pending" / "approved" / "rejected"
- type (TEXT) - "Auto" / "Health" / "Home"
- description (TEXT) - краткое описание случая

Таблица customer_history:
- id (INTEGER, PK autoincrement)
- client (TEXT)
- claim_id (TEXT)
- date (TEXT)
- outcome (TEXT) - "approved" / "rejected" / "fraud_detected"

## 2. seed_data.py должен создать минимум 5 claims с разными суммами (от 1000 до 20000)
и минимум 8 записей в customer_history, включая хотя бы 1 клиента с историей "fraud_detected"
в прошлом (это важно для демонстрации fraud risk сценария).

## 3. tools.py - три функции, которые ПОТОМ будет использовать LangGraph-агент как tools:

```python
def get_claim_details(claim_id: str) -> dict:
    """Возвращает полные данные claim'а по ID."""
    # запрос к SQLite, вернуть dict или {"error": "not found"}

def check_fraud_risk(claim_id: str) -> dict:
    """
    Простая демо-логика:
    - берёт amount claim'а
    - смотрит customer_history клиента: если там были fraud_detected -> risk += 0.3
    - risk = min(amount / 20000, 1.0) + бонус за историю, capped на 1.0
    - возвращает {"risk_score": float, "level": "low"/"medium"/"high"}
    """

def get_customer_history(client: str) -> list:
    """Возвращает список прошлых claims этого клиента из customer_history."""
```

## 4. main.py - FastAPI эндпоинты СТРОГО по контракту:

- GET /claims -> список всех claims (id, client, amount, status, type)
- GET /claims/{claim_id} -> полные детали одного claim
- Добавь CORS middleware (разрешить http://localhost:5173) - фронт будет на React/Vite

ВАЖНО: эндпоинты /chat и /resume НЕ пиши сам - их добавит следующий агент (LangGraph agent),
который будет импортировать твои tools.py функции. Просто убедись, что tools.py
легко импортируется отдельно (без побочных эффектов при импорте).

## 5. requirements.txt
fastapi
uvicorn
pydantic

## Как запустить (напиши это в конце как README.md комментарий)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 seed_data.py
uvicorn main:app --reload --port 8000

Проверка: открыть http://localhost:8000/claims в браузере, должен вернуться JSON список.
