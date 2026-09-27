import { useEffect, useRef, useState } from "react";
import { sendMessage, resumeDecision } from "./api";

let messageIdCounter = 0;
function nextMessageId() {
  messageIdCounter += 1;
  return messageIdCounter;
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
  const bottomRef = useRef(null);

  // Reset the conversation whenever a different claim is selected.
  useEffect(() => {
    setMessages([]);
    setInput("");
    setChatError(null);
    setResolvingId(null);
  }, [claimId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, sending]);

  const appendMessage = (message) => {
    setMessages((prev) => [...prev, { id: nextMessageId(), ...message }]);
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || sending) return;

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

        {claim && (
          <div className="chat-panel__summary">
            <span className="chat-panel__summary-item">{claim.type}</span>
            <span className="chat-panel__summary-item">
              {typeof claim.amount === "number" ? `$${claim.amount.toLocaleString()}` : claim.amount}
            </span>
            <span className={`status-badge status-badge--${claim.status}`}>{claim.status}</span>
            {claim.fraud_risk != null && (
              <span className="chat-panel__summary-item chat-panel__summary-item--fraud">
                Fraud risk: {claim.fraud_risk}
              </span>
            )}
          </div>
        )}

        {claim?.description && <p className="chat-panel__description">{claim.description}</p>}
      </div>

      <div className="chat-panel__messages">
        {messages.length === 0 && !sending && (
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
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message to the AI copilot..."
          rows={2}
          disabled={sending}
        />
        <button
          type="button"
          className="btn btn--send"
          onClick={handleSend}
          disabled={sending || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}
