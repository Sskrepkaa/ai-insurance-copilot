const STATUS_LABELS = {
  pending: "Pending",
  approved: "Approved",
  rejected: "Rejected",
};

function formatAmount(amount) {
  if (typeof amount !== "number") return amount;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(amount);
}

export default function ClaimCard({ claim, selected, onSelect }) {
  const statusLabel = STATUS_LABELS[claim.status] || claim.status;

  return (
    <button
      type="button"
      className={`claim-card${selected ? " claim-card--selected" : ""}`}
      onClick={() => onSelect(claim.id)}
    >
      <div className="claim-card__top">
        <span className="claim-card__id">{claim.id}</span>
        <span className={`status-badge status-badge--${claim.status}`}>{statusLabel}</span>
      </div>
      <div className="claim-card__client">{claim.client}</div>
      <div className="claim-card__meta">
        <span className="claim-card__type">{claim.type}</span>
        <span className="claim-card__amount">{formatAmount(claim.amount)}</span>
      </div>
    </button>
  );
}
