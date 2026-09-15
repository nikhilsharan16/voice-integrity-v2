export const API_BASE_URL = "http://localhost:3001";

// The Python inference service (voice-integrity-inference-service), not
// the Node mock. Only the "upload" flow calls this — the sidebar's canned demo
// buttons still go through API_BASE_URL's mock, since there's no real
// audio behind them.
export const PYTHON_API_BASE_URL = "http://localhost:8000";