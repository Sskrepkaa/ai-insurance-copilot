# ПРОМПТ ДЛЯ АГЕНТА 4: Frontend (React + Vite дашборд)

Ты создаёшь React-фронтенд для демо "AI Copilot для страховой поддержки".
Backend уже работает на http://localhost:8000 со следующими эндпоинтами:
GET /claims, GET /claims/{id}, POST /chat, POST /resume (детали контракта - см. architecture.md)

## Стек
Vite + React (без TypeScript для простоты, JSX). Стилизация - обычный CSS или Tailwind
(если Tailwind - настрой через CDN, чтобы не тратить время на конфиг).

## Структура экрана (2 колонки)

### Левая колонка: список claims (карточки)
- Загрузи GET /claims при монтировании
- Каждая карточка показывает: id, client, amount, type, status (цветной badge:
  pending=желтый, approved=зелёный, rejected=красный)
- Клик по карточке -> выбирает claim, подгружает детали через GET /claims/{id},
  открывает правую колонку

### Правая колонка: чат с агентом (появляется после выбора claim)
- Заголовок: "Claim {id} - {client}"
- История сообщений (простой список: user/agent, разным цветом/выравниванием)
- Поле ввода + кнопка "Отправить" -> POST /chat {claim_id, message}
- После ответа агента:
  - если response.type === "answer": показать текст ответа + маленький индикатор
    судьи: зелёная точка если judge_score.verdict === "pass", красная если "fail",
    с tooltip показывающим groundedness/relevance числа
  - если response.type === "interrupt": показать специальный блок с вопросом
    (response.question) и ДВЕ КНОПКИ: "✅ Одобрить" / "❌ Отклонить"
    При клике -> POST /resume {thread_id: response.thread_id, approved: true/false,
    reason: "решение эксперта через дашборд"} -> показать финальный ответ агента

## Компоненты (файлы)
src/
  App.jsx           - общий layout, стейт выбранного claim
  ClaimList.jsx      - левая колонка, список карточек
  ClaimCard.jsx       - одна карточка claim'а
  ChatPanel.jsx       - правая колонка, вся логика чата + HITL кнопок
  api.js              - функции fetchClaims(), fetchClaimDetails(id), sendMessage(claimId, message), resumeDecision(threadId, approved, reason)

## api.js - пример одной функции для ориентира
```javascript
const API_BASE = "http://localhost:8000";

export async function sendMessage(claimId, message) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({claim_id: claimId, message}),
  });
  return res.json();
}
```

## Важные детали UX
- Пока ждём ответ от /chat - показать простой спиннер/"Агент думает..."
- Судья-индикатор должен быть заметным, но не отвлекающим - маленький кружок рядом
  с сообщением агента, не отдельный большой блок
- HITL-блок с кнопками должен визуально отличаться (например, желтая обводка/фон),
  чтобы было понятно "это не обычный ответ, тут нужно решение человека"

## Запуск
npm create vite@latest frontend -- --template react
cd frontend && npm install
npm run dev
Открыть http://localhost:5173, убедиться что видно список claims из backend.
