import express from "express";
import cors from "cors";
import { SCENARIOS } from "./scenarios.js";

const app = express();
app.use(cors());
app.use(express.json());

function jitter(target, amount) {
  const v = target + (Math.random() - 0.5) * amount;
  return Math.max(1, Math.min(99, Math.round(v)));
}

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

/**
 * POST /analyze-window
 * body: { profileKey: "genuine" | "cloned" | "mismatch" | "replay" }
 * response: { spoof_score: number, speaker_similarity: number }
 *
 * THIS IS THE MOCK. Real integration replaces this handler so it:
 *   1. Accepts an actual audio chunk instead of a profileKey
 *      (e.g. multer for multipart, or a raw buffer body)
 *   2. Runs AASIST -> spoof_score and ECAPA-TDNN -> speaker_similarity
 *   3. Returns the same { spoof_score, speaker_similarity } shape
 *
 * Nothing on the frontend needs to change if that response shape is
 * kept — see analyzeWindow() in the frontend's src/lib/scoring.js.
 */
app.post("/analyze-window", (req, res) => {
  const { profileKey } = req.body || {};
  const profile = SCENARIOS[profileKey] || SCENARIOS.genuine;

  const spoof_score = jitter(profile.spoofTarget, 14);
  const speaker_similarity = jitter(profile.matchTarget, 10);

  // Small artificial delay so the frontend behaves the same way it
  // will once real inference has non-zero latency.
  setTimeout(() => {
    res.json({ spoof_score, speaker_similarity });
  }, 120);
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Voice Integrity mock API running on http://localhost:${PORT}`);
});
