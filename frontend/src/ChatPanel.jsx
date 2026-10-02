import { useEffect, useRef, useState } from "react";
import { sendMessage, resumeDecision, fetchChatHistory } from "./api";

let messageIdCounter = 0;
function nextMessageId() {
  messageIdCounter += 1;
  return messageIdCounter;
}

const SUGGESTIONS = [
  {
    label: "🔍 Проверить риск мошенничества",
    prompt: "Проверь этот claim на мошенничество.",
  },
  {
    label: "📋 История клиента",
    prompt: "Покажи историю предыдущих claims этого клиента.",
  },
];

function ClaimFactsCard({ claim, onSuggestion }) {
  if (!claim) return null;

  const amountText =
    typeof claim.amount === "number" ? `$${claim.amount.toLocaleString()}` : claim.amount;

  return (
    <div className="claim-facts-card">
      <div className="claim-facts-card__grid">
        <div className="claim-facts-card__fact">
          <span className="claim-facts-card__label">Amount</span>
          <span className="claim-facts-card__value">{amountText}</span>
        </div>
        <div className="claim-facts-card__fact">
          <span className="claim-facts-card__label">Status</span>
          <span className={`status-badge status-badge--${claim.status}`}>{claim.status}</span>
        </div>
        <div className="claim-facts-card__fact">
          <span className="claim-facts-card__label">Type</span>
          <span className="claim-facts-card__value">{claim.type}</span>
        </div>
        {claim.fraud_risk != null && (
          <div className="claim-facts-card__fact">
            <span className="claim-facts-card__label">Fraud risk</span>
            <span className="claim-facts-card__value claim-facts-card__value--fraud">
              {claim.fraud_risk}
            </span>
          </div>
        )}
      </div>

      {claim.description && <p className="claim-facts-card__description">{claim.description}</p>}

      <div className="claim-facts-card__suggestions">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.label}
            type="button"
            className="suggestion-chip"
            onClick={() => onSuggestion(s.prompt)}
          >
            {s.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function JudgeDot({ judge }) {
  if (!judge) return null;
  const verdict = judge.verdict === "pass" ? "pass" : "fail";

  const parts = [];
  if (typeof judge.groundedness === "number") {
    parts.push(`Groundedness: ${judge.groundedness.toFixed(2)}`);
  }
  if (typeof judge.relevance === "number") {
    parts.push(`Relevance: ${judge.relevance.toFixed(2)}`);
  }
  const title = parts.length > 0 ? parts.join(" · ") : `Judge verdict: ${judge.verdict}`;

  return (
    <span
      className={`judge-dot judge-dot--${verdict}`}
      title={title}
      aria-label={`AI judge verdict: ${judge.verdict}. ${title}`}
    />
  );
}

export default function ChatPanel({ claimId, claim, detailsLoading, detailsError }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [resolvingId, setResolvingId] = useState(null);
  const [chatError, setChatError] = useState(null);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState(null);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  // Reset the conversation whenever a different claim is selected, then
  // restore whatever the backend's checkpointer already remembers for this
  // claim's thread - the agent keeps full history server-side, so reopening
  // a claim shouldn't show an empty chat if one already happened.
  useEffect(() => {
    let cancelled = false;
    setMessages([]);
    setInput("");
    setChatError(null);
    setResolvingId(null);
    setHistoryError(null);
    setHistoryLoading(true);

    fetchChatHistory(claimId)
      .then((data) => {
        if (cancelled) return;
        const restored = (data.messages || []).map((m) => ({
          id: nextMessageId(),
          role: m.role === "assistant" ? "agent" : "user",
          text: m.text,
        }));
        // A claim left mid-interrupt (the employee navigated away before
        // clicking Approve/Reject) comes back with pending_interrupt set -
        // render it as the live HITL card, not a plain message, so the
        // decision can still be made.
        if (data.pending_interrupt) {
          restored.push({
            id: nextMessageId(),
            role: "interrupt",
            question: data.pending_interrupt.question,
            threadId: data.pending_interrupt.thread_id,
            resolved: false,
          });
        }
        setMessages(restored);
      })
      .catch((err) => {
        if (!cancelled) setHistoryError(err.message || "Failed to load chat history.");
      })
      .finally(() => {
        if (!cancelled) setHistoryLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [claimId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, sending]);

  const appendMessage = (message) => {
    setMessages((prev) => [...prev, { id: nextMessageId(), ...message }]);
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || sending || historyLoading) return;

    appendMessage({ role: "user", text });
    setInput("");
    setChatError(null);
    setSending(true);

    try {
      const response = await sendMessage(claimId, text);
      if (response.type === "interrupt") {
        appendMessage({
          role: "interrupt",
          question: response.question,
          threadId: response.thread_id,
          resolved: false,
        });
      } else {
        appendMessage({ role: "agent", text: response.text, judge: response.judge_score });
      }
    } catch (err) {
      setChatError(err.message || "Failed to reach the AI copilot.");
    } finally {
      setSending(false);
    }
  };

  const handleDecision = async (message, approved) => {
    setResolvingId(message.id);
    setChatError(null);

    try {
      const response = await resumeDecision(
        message.threadId,
        approved,
        "expert decision via dashboard"
      );
      setMessages((prev) =>
        prev.map((m) =>
          m.id === message.id
            ? { ...m, resolved: true, decision: approved ? "approved" : "rejected" }
            : m
        )
      );
      appendMessage({ role: "agent", text: response.text, judge: response.judge_score });
    } catch (err) {
      setChatError(err.message || "Failed to submit the decision.");
    } finally {
      setResolvingId(null);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  // Suggestion chips only fill the input - the employee still decides
  // whether/when to actually send it, so we don't burn a Groq call just
  // because a claim was opened.
  const handleSuggestion = (prompt) => {
    setInput(prompt);
    textareaRef.current?.focus();
  };

  return (
    <div className="chat-panel">
      <div className="chat-panel__header">
        <h2>
          Claim {claimId}
          {claim ? ` – ${claim.client}` : ""}
        </h2>

        {detailsLoading && <p className="chat-panel__status">Loading claim details...</p>}
        {detailsError && (
          <p className="chat-panel__status chat-panel__status--error">{detailsError}</p>
        )}

        <ClaimFactsCard claim={claim} onSuggestion={handleSuggestion} />
      </div>

      <div className="chat-panel__messages">
        {historyLoading && (
          <p className="chat-panel__status">Loading conversation...</p>
        )}

        {historyError && (
          <p className="chat-panel__status chat-panel__status--error">{historyError}</p>
        )}

        {!historyLoading && !historyError && messages.length === 0 && !sending && (
          <p className="chat-panel__empty">
            Ask the AI copilot about this claim, e.g. "Check this claim for fraud".
          </p>
        )}

        {messages.map((message) => {
          if (message.role === "interrupt") {
            return (
              <div key={message.id} className="hitl-block">
                <div className="hitl-block__label">Human decision required</div>
                <p className="hitl-block__question">{message.question}</p>
                {message.resolved ? (
                  <p className="hitl-block__resolved">
                    Decision: <strong>{message.decision}</strong>
                  </p>
                ) : (
                  <div className="hitl-block__actions">
                    <button
                      type="button"
                      className="btn btn--approve"
                      disabled={resolvingId === message.id}
                      onClick={() => handleDecision(message, true)}
                    >
                      {"✅ Approve"}
                    </button>
                    <button
                      type="button"
                      className="btn btn--reject"
                      disabled={resolvingId === message.id}
                      onClick={() => handleDecision(message, false)}
                    >
                      {"❌ Reject"}
                    </button>
                  </div>
                )}
                {resolvingId === message.id && (
                  <p className="hitl-block__pending">Submitting decision...</p>
                )}
              </div>
            );
          }

          const isAgent = message.role === "agent";
          return (
            <div key={message.id} className={`chat-message chat-message--${message.role}`}>
              <div className="chat-message__bubble">
                <span className="chat-message__text">{message.text}</span>
                {isAgent && <JudgeDot judge={message.judge} />}
              </div>
            </div>
          );
        })}

        {sending && (
          <div className="chat-message chat-message--agent">
            <div className="chat-message__bubble chat-message__bubble--thinking">
              <span className="spinner" aria-hidden="true" />
              Agent is thinking...
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {chatError && (
        <p className="chat-panel__status chat-panel__status--error chat-panel__status--floating">
          {chatError}
        </p>
      )}

      <div className="chat-panel__input">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message to the AI copilot..."
          rows={2}
          disabled={sending || historyLoading}
        />
        <button
          type="button"
          className="btn btn--send"
          onClick={handleSend}
          disabled={sending || historyLoading || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}
