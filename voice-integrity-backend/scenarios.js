// Mirrors src/data/scenarios.js on the frontend. This whole file goes
// away once the real model is producing scores from actual audio —
// see the note in server.js.
export const SCENARIOS = {
  genuine: { spoofTarget: 8, matchTarget: 93 },
  cloned: { spoofTarget: 88, matchTarget: 90 },
  mismatch: { spoofTarget: 14, matchTarget: 22 },
  replay: { spoofTarget: 62, matchTarget: 85 },
};
