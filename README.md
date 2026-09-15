# voice-integrity

Real-time voice cloning / impersonation detection dashboard (SIH). Three
independent services, run together:

```
voice-integrity/
├── voice-integrity-dashboard/         React frontend (Vite)
├── voice-integrity-backend/           Node/Express mock API (demo scenarios)
└── voice-integrity-inference-service/ Python FastAPI - real AASIST + ECAPA-TDNN
```

## What each piece does

- **`voice-integrity-dashboard`** - the UI. Has two data sources:
  - The sidebar's "Demo scenarios" buttons → call `voice-integrity-backend`
    (the Node mock) for reliable, canned stage-demo numbers.
  - The "Analyze a recording" upload flow → calls
    `voice-integrity-inference-service` (the Python service) for real
    spoof detection (AASIST) and speaker identification (ECAPA-TDNN).
- **`voice-integrity-backend`** - a tiny Express server that fakes scores
  for the demo-scenario buttons. Nothing here needs to change; it's kept
  around specifically so the demo has a reliable fallback path if the real
  model has a bad day on stage.
- **`voice-integrity-inference-service`** - the real ML. Wraps a pretrained
  AASIST checkpoint (spoof/deepfake detection) and SpeechBrain's ECAPA-TDNN
  (speaker verification) behind two endpoints the dashboard calls:
  `POST /analyze` (spoof + speaker match for an uploaded clip) and
  `POST /enroll` (register a speaker's voice sample). See that folder's own
  README for what's been adapted from the original ML notebooks/scripts and
  what's still untested.

## Running all three

You need three terminals open at once.

**1. Python inference service** (port 8000):
```bash
cd voice-integrity-inference-service
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**2. Node mock backend** (port 3001):
```bash
cd voice-integrity-backend
npm install
npm start
```

**3. React dashboard** (port 5173 by default):
```bash
cd voice-integrity-dashboard
npm install
npm run dev
```

Then open the dashboard's dev server URL in your browser. The demo-scenario
buttons work as soon as step 2 is up; the "Analyze a recording" upload flow
needs step 1 up as well (and at least one enrolled speaker, via the
sidebar's "Register voice sample" form, for a real speaker-match score -
without any enrollment it still runs, just with `match` treated as "not
checked" rather than "verified").

## Known gaps / next steps

- AASIST checkpoint is not fine-tuned yet (`AASIST_NOT_FT` in the original
  ML repo) - scores can be noisy on out-of-domain audio. `smoothed_score`
  in the API response is a rolling average meant to make this more usable
  in the UI; swap the checkpoint file in-place once a fine-tuned version
  exists, no other code changes needed.
- ECAPA's cosine-similarity-to-percentage mapping in
  `speaker_model.py::identify_speaker()` is a rough linear rescale, not a
  calibrated threshold - worth checking against real same-speaker vs.
  different-speaker clips once you've enrolled someone.
- No live/streaming path yet - both the AASIST and ECAPA flows are
  file-upload-then-analyze, not real-time mic streaming.
