import React, { useState, useEffect, useRef, useCallback } from "react";
import { PhoneOff, PlayCircle } from "lucide-react";
import Sidebar from "./components/Sidebar.jsx";
import Hero from "./components/Hero.jsx";
import Metrics from "./components/Metrics.jsx";
import Timeline from "./components/Timeline.jsx";
import AssessmentPanel from "./components/AssessmentPanel.jsx";
import SessionLog from "./components/SessionLog.jsx";
import {
  analyzeWindow,
  analyzeUploadedAudio,
  enrollSpeaker,
  computeRisk,
  riskBand,
  reasonsFor,
  genPhrase,
  MAX_WINDOWS,
  TICK_MS,
} from "./lib/scoring.js";

export default function App() {
  const [session, setSession] = useState(null);
  const [running, setRunning] = useState(false);
  const [tick, setTick] = useState(0);
  const [history, setHistory] = useState([]);
  const [bars, setBars] = useState(() => Array.from({ length: 36 }, () => 6));
  const [sessionLog, setSessionLog] = useState([]);
  const [registered, setRegistered] = useState([
    { name: "CFO — R. Malhotra", id: "SPK-0142" },
    { name: "CEO — A. Varma", id: "SPK-0087" },
  ]);
  const [challenge, setChallenge] = useState(null);
  
  const timeoutRef = useRef(null);
  const streakRef = useRef(0);

  const latest = history[history.length - 1];
  const risk = latest ? latest.risk : 0;
  const band = latest ? riskBand(risk) : "idle";

  const startAnalysis = (payload) => {
    clearTimeout(timeoutRef.current);
    setSession(payload);
    setHistory([]);
    setTick(0);
    setChallenge(null);
    streakRef.current = 0;
    setRunning(true);
  };

  const endAnalysis = useCallback(
    (auto = false) => {
      clearTimeout(timeoutRef.current);
      setRunning(false);
      setHistory((h) => {
        if (h.length > 0 && session) {
          const last = h[h.length - 1];
          setSessionLog((log) =>
            [
              {
                id: `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
                label: session.label,
                source: session.source,
                risk: last.risk,
                band: riskBand(last.risk),
                windows: h.length,
                auto,
              },
              ...log,
            ].slice(0, 8)
          );
        }
        return h;
      });
    },
    [session]
  );

  useEffect(() => {
    if (!running || !session) return;

    let active = true;
    // When set, this is the real per-window sequence from the uploaded
    // file (via the Python AASIST service) - runTick steps through it
    // instead of calling the mock analyzeWindow() each tick. Stays null
    // for demo-button sessions, and for uploads where real analysis
    // failed (falls back to the mock below, same as before).
    let sequence = null;
    let idx = 0;

    const runTick = async () => {
      if (!active) return;

      // Fetch the data asynchronously BEFORE updating React state
      const { spoof, match } = sequence ? sequence[idx] : await analyzeWindow(session.profileKey);
      idx += 1;
      if (!active) return;

      const risk = computeRisk(spoof, match);
      const band = riskBand(risk);
      const maxTicks = sequence ? sequence.length : MAX_WINDOWS;

      setTick((t) => {
        const next = t + 1;
        if (next > maxTicks) return t;

        streakRef.current = band !== "safe" ? streakRef.current + 1 : 0;

        setHistory((h) => [...h, { t: next, spoof, match, risk, streak: streakRef.current }].slice(-MAX_WINDOWS));

        if (session.peaks && session.peaks.length > 0) {
          const offset = next % session.peaks.length;
          setBars(
            Array.from({ length: 36 }, (_, i) => {
              const p = session.peaks[(offset + i) % session.peaks.length];
              return 4 + p * 80;
            })
          );
        } else {
          setBars(Array.from({ length: 36 }, () => 4 + Math.random() * 40 + (band === "danger" ? Math.random() * 10 : 0)));
        }

        if (next >= maxTicks) {
          active = false;
          setTimeout(() => endAnalysis(true), 250);
        } else {
          timeoutRef.current = setTimeout(runTick, TICK_MS);
        }

        return next;
      });
    };

    const start = async () => {
      if (session.source === "upload" && session.file) {
        try {
          const real = await analyzeUploadedAudio(session.file);
          // Cap to MAX_WINDOWS - the timeline/history UI is sized around
          // it, same as the demo path.
          sequence = real.slice(0, MAX_WINDOWS);
        } catch (err) {
          // Analysis service unreachable, or no speech detected in the
          // clip - fall back to the simulated scoring path so the demo
          // doesn't just break.
          console.error("Real analysis unavailable, falling back to simulated scoring:", err);
          sequence = null;
        }
      }
      if (active) {
        timeoutRef.current = setTimeout(runTick, TICK_MS);
      }
    };

    start();

    return () => {
      active = false;
      clearTimeout(timeoutRef.current);
    };
  }, [running, session, endAnalysis]);

  const triggerChallenge = () => setChallenge({ phrase: genPhrase(), status: "waiting" });
  const resolveChallenge = (pass) => setChallenge((c) => ({ ...c, status: pass ? "pass" : "fail" }));

  // Actually enrolls the sample with the ECAPA service (POST /enroll) and
  // only adds the sidebar list entry once that succeeds - previously this
  // just added a fake name+ID with no real audio behind it. Throws on
  // failure so Sidebar.jsx can show the error and leave the form filled in
  // for a retry instead of silently pretending it worked.
  const addRegistered = async (name, file) => {
    await enrollSpeaker(name, file);
    setRegistered((r) => [...r, { name, id: `SPK-${Math.floor(1000 + Math.random() * 8999)}` }]);
  };

  const reasons = latest ? reasonsFor(latest.spoof, latest.match, latest.risk, latest.streak) : [];

  return (
    <div className="vie-root">
      <Sidebar
        onAnalyze={startAnalysis}
        running={running}
        activeDemoKey={session?.source === "demo" ? session.profileKey : null}
        registered={registered}
        onAddRegistered={addRegistered}
      />

      <main className="vie-main">
        <div className="vie-header">
          <div>
            <div className="vie-title vie-display">Recording analysis</div>
            <div className="vie-subtitle">Continuous near-real-time voice integrity analysis with rolling risk updates</div>
          </div>
          <div className="vie-call-controls">
            {running ? (
              <button className="vie-btn danger-outline" onClick={() => endAnalysis(false)}>
                <PhoneOff size={14} /> Stop analysis
              </button>
            ) : (
              session && (
                <button className="vie-btn primary" onClick={() => startAnalysis(session)}>
                  <PlayCircle size={14} /> Run again
                </button>
              )
            )}
          </div>
        </div>

        <Hero
          running={running}
          activeSession={!!session} 
          hasCompletedSession={!running && history.length > 0} 
          bars={bars}
          sessionLabel={session?.label}
          sourceTag={session?.source === "upload" ? "uploaded" : session?.source === "demo" ? "demo" : null}
          tick={tick}
          band={band}
          risk={risk}
        />

        <Metrics latest={latest} />

        <div className="vie-lower">
          <div className="vie-panel">
            <div className="vie-panel-title">Rolling risk — last {MAX_WINDOWS} windows</div>
            <Timeline history={history} />
          </div>

          <AssessmentPanel
            latest={latest}
            band={band}
            reasons={reasons}
            challenge={challenge}
            onTriggerChallenge={triggerChallenge}
            onResolveChallenge={resolveChallenge}
          />
        </div>

        <SessionLog log={sessionLog} />
      </main>
    </div>
  );
}