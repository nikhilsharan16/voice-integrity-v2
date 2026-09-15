import React, { useRef, useState } from "react";
import { RadioTower, UserPlus } from "lucide-react";
import { DEMO_PRESETS } from "../data/scenarios.js";
import UploadPanel from "./UploadPanel.jsx";

export default function Sidebar({ onAnalyze, running, activeDemoKey, registered, onAddRegistered }) {
  const [nameInput, setNameInput] = useState("");
  const [enrollFile, setEnrollFile] = useState(null);
  const [enrolling, setEnrolling] = useState(false);
  const [enrollError, setEnrollError] = useState(null);
  const fileInputRef = useRef(null);

  const addRegistered = async () => {
    const name = nameInput.trim();
    if (!name || !enrollFile) {
      setEnrollError("Name and a voice sample are both required.");
      return;
    }
    setEnrolling(true);
    setEnrollError(null);
    try {
      // onAddRegistered does the actual ECAPA enrollment call (POST
      // /enroll) and only updates the sidebar list once that succeeds -
      // see App.jsx.
      await onAddRegistered(name, enrollFile);
      setNameInput("");
      setEnrollFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setEnrollError(err.message || "Enrollment failed.");
    } finally {
      setEnrolling(false);
    }
  };

  return (
    <aside className="vie-sidebar">
      <div className="vie-brand vie-display">
        <div className="vie-brand-mark">
          <RadioTower size={15} />
        </div>
        Voice Integrity
      </div>

      <UploadPanel disabled={running} onAnalyze={(payload) => onAnalyze(payload)} />

      <div>
        <div className="vie-section-label">Demo scenarios</div>
        <div className="vie-section-sub">Canned examples for a reliable stage demo — not real audio.</div>
        {DEMO_PRESETS.map((p) => (
          <button
            key={p.key}
            className={`vie-scenario-btn${activeDemoKey === p.key && running ? " active" : ""}`}
            onClick={() => onAnalyze({ label: p.label, source: "demo", profileKey: p.key, peaks: null })}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div>
        <div className="vie-section-label">Registered speakers</div>
        <div className="vie-section-sub">A name and a voice sample together create a real ECAPA-TDNN template.</div>
        {registered.map((r) => (
          <div className="vie-reg-item" key={r.id}>
            <span>{r.name}</span>
            <span className="vie-mono">{r.id}</span>
          </div>
        ))}
        <input
          className="vie-input"
          style={{ marginTop: 10 }}
          placeholder="Full name"
          value={nameInput}
          onChange={(e) => setNameInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && addRegistered()}
        />
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          style={{ marginTop: 6, fontSize: 12 }}
          onChange={(e) => setEnrollFile(e.target.files?.[0] || null)}
        />
        {enrollError && (
          <div className="vie-empty" style={{ color: "var(--danger)" }}>
            {enrollError}
          </div>
        )}
        <button className="vie-add-btn" onClick={addRegistered} disabled={enrolling}>
          <UserPlus size={13} /> {enrolling ? "Enrolling…" : "Register voice sample"}
        </button>
      </div>
    </aside>
  );
}
