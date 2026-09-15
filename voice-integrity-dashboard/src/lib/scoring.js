import { SCENARIOS } from "../data/scenarios.js";
import { API_BASE_URL, PYTHON_API_BASE_URL } from "./config.js";

export const MAX_WINDOWS = 16;
export const TICK_MS = 850;

function jitter(target, amount) {
  const v = target + (Math.random() - 0.5) * amount;
  return Math.max(1, Math.min(99, Math.round(v)));
}

export function computeRisk(spoof, match) {
  const mismatchComponent = 100 - match;
  return Math.round(0.58 * spoof + 0.42 * mismatchComponent);
}

export function riskBand(risk) {
  if (risk < 35) return "safe";
  if (risk < 65) return "warn";
  return "danger";
}

export function reasonsFor(spoof, match, risk, streak) {
  const out = [];
  if (spoof > 55) out.push("Synthesis artifacts detected in acoustic signature");
  else if (spoof > 30) out.push("Minor spectral inconsistencies present");
  if (match < 50) out.push("Speaker identity does not match registered profile");
  else if (match < 75) out.push("Partial inconsistency in voice identity match");
  if (streak >= 3 && riskBand(risk) !== "safe") {
    out.push(`Anomaly persistent across ${streak} consecutive windows`);
  }
  if (out.length === 0) out.push("No anomalies detected in current window");
  return out;
}

export function genPhrase() {
  const words = ["MANGO", "GRANITE", "ORBIT", "COBALT", "TRIVET", "HARBOR", "ZEPHYR", "QUARTZ"];
  const w = words[Math.floor(Math.random() * words.length)];
  const n = Math.floor(10 + Math.random() * 89);
  return `${w}-${n}`;
}

export async function analyzeWindow(profileKey) {
  try {
    const response = await fetch(`${API_BASE_URL}/analyze-window`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profileKey }),
    });
    if (!response.ok) throw new Error("Backend offline");
    
    // Map backend snake_case response to frontend property names
    const data = await response.json();
    return { 
      spoof: data.spoof_score, 
      match: data.speaker_similarity 
    };
  } catch (err) {
    // Fallback simulation if mock server is down
    const profile = SCENARIOS[profileKey] || SCENARIOS.genuine;
    const spoof = jitter(profile.spoofTarget, 14);
    const match = jitter(profile.matchTarget, 10);
    return { spoof, match };
  }
}

/**
 * Real analysis for an uploaded file, via the Python AASIST + ECAPA service
 * (voice-integrity-inference-service, POST /analyze). Unlike analyzeWindow(), this
 * makes ONE request for the whole file - the service already slices it
 * into windows and scores each - and returns the full per-window sequence
 * up front, so App.jsx just steps through it on each tick instead of
 * re-fetching per tick.
 *
 * Score mapping:
 *   - spoof: the service returns bonafide_score/smoothed_score per window,
 *     0-100, where 100 = confidently genuine. This dashboard's convention
 *     is the opposite (spoof: 0-100, high = suspicious), so
 *     spoof = 100 - smoothed_score, computed per window.
 *   - match: the service now runs ECAPA-TDNN speaker identification ONCE
 *     against the whole clip (embeddings need a real utterance, not a 4s
 *     AASIST window) and returns speaker_match.match_score, 0-100. That
 *     single score is applied to every window in this session - risk still
 *     varies tick to tick because spoof does, even though match is flat
 *     for a given clip. If nobody's enrolled yet, match_score comes back
 *     null and is treated as 100 (no identity check performed, not "no
 *     mismatch found" - don't read a flat 100 as a real verification pass
 *     when speaker_match.best_speaker is also null).
 *
 * Throws if the request fails or no speech was detected, so callers can
 * fall back to the simulated scoring path.
 */
export async function analyzeUploadedAudio(file) {
  const formData = new FormData();
  formData.append("audio", file);

  const response = await fetch(`${PYTHON_API_BASE_URL}/analyze`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) throw new Error(`Analysis service returned ${response.status}`);

  const data = await response.json();
  if (!data.windows || data.windows.length === 0) {
    throw new Error("No speech detected in this clip.");
  }

  const speakerMatch = data.speaker_match || {};
  const match =
    speakerMatch.match_score === null || speakerMatch.match_score === undefined
      ? 100 // no one enrolled yet - see note above, this is "not checked", not "verified"
      : Math.max(1, Math.min(99, Math.round(speakerMatch.match_score)));

  return data.windows.map((w) => ({
    spoof: Math.max(1, Math.min(99, Math.round(100 - w.smoothed_score))),
    match,
  }));
}

/**
 * Enroll one audio sample for a named speaker via the Python service's
 * POST /enroll (ECAPA-TDNN embedding, adapted from
 * Tanmay-1404/ecapa-not-automated's enroll.py). Called from the sidebar's
 * "Register voice sample" flow once it captures/uploads real audio instead
 * of just a name.
 */
export async function enrollSpeaker(name, file) {
  const formData = new FormData();
  formData.append("name", name);
  formData.append("audio", file);

  const response = await fetch(`${PYTHON_API_BASE_URL}/enroll`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Enrollment failed: ${response.status}`);
  }
  return response.json(); // { name, samples_enrolled }
}