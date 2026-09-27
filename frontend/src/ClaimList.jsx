import ClaimCard from "./ClaimCard";

export default function ClaimList({ claims, loading, error, selectedId, onSelect }) {
  return (
    <div className="claim-list">
      <div className="claim-list__header">
        <h2>Claims</h2>
        <span className="claim-list__count">{claims.length}</span>
      </div>

      {loading && <p className="claim-list__status">Loading claims...</p>}
      {error && <p className="claim-list__status claim-list__status--error">{error}</p>}
      {!loading && !error && claims.length === 0 && (
        <p className="claim-list__status">No claims found.</p>
      )}

      <div className="claim-list__items">
        {claims.map((claim) => (
          <ClaimCard
            key={claim.id}
            claim={claim}
            selected={claim.id === selectedId}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  );
}
