import { useEffect, useState } from "react";
import ClaimList from "./ClaimList";
import ChatPanel from "./ChatPanel";
import { fetchClaims, fetchClaimDetails } from "./api";
import "./App.css";

export default function App() {
  const [claims, setClaims] = useState([]);
  const [claimsLoading, setClaimsLoading] = useState(true);
  const [claimsError, setClaimsError] = useState(null);

  const [selectedId, setSelectedId] = useState(null);
  const [selectedClaim, setSelectedClaim] = useState(null);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [detailsError, setDetailsError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setClaimsLoading(true);
    setClaimsError(null);

    fetchClaims()
      .then((data) => {
        if (!cancelled) setClaims(data);
      })
      .catch((err) => {
        if (!cancelled) setClaimsError(err.message || "Failed to load claims.");
      })
      .finally(() => {
        if (!cancelled) setClaimsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const handleSelect = (claimId) => {
    setSelectedId(claimId);
    setSelectedClaim(null);
    setDetailsError(null);
    setDetailsLoading(true);

    fetchClaimDetails(claimId)
      .then((data) => setSelectedClaim(data))
      .catch((err) => setDetailsError(err.message || "Failed to load claim details."))
      .finally(() => setDetailsLoading(false));
  };

  return (
    <div className="app">
      <header className="app__header">
        <h1>AI Insurance Copilot</h1>
        <p className="app__subtitle">Support dashboard</p>
      </header>

      <div className="app__body">
        <ClaimList
          claims={claims}
          loading={claimsLoading}
          error={claimsError}
          selectedId={selectedId}
          onSelect={handleSelect}
        />

        {selectedId ? (
          <ChatPanel
            key={selectedId}
            claimId={selectedId}
            claim={selectedClaim}
            detailsLoading={detailsLoading}
            detailsError={detailsError}
          />
        ) : (
          <div className="app__placeholder">
            <p>Select a claim on the left to start a conversation with the AI copilot.</p>
          </div>
        )}
      </div>
    </div>
  );
}
