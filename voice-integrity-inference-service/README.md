# Voice integrity inference service

Wraps the two ML notebooks into a FastAPI service your React dashboard can
call in place of its mocked scoring.

- `app/preprocessing.py` - adapted from `audio-preprocessing-16khz-slidingwindow.ipynb`
- `app/model.py` + `app/aasist_model/aasist.py` - adapted from `aasist-pretrained.ipynb`
- `app/main.py` - the FastAPI app tying them together
- `weights/AASIST.pth`, `weights/AASIST.conf` - pretrained checkpoint + config, copied from `clovaai/aasist` (see "checkpoint source" below)

## What was fixed vs. the raw notebooks

1. **Window size mismatch.** The preprocessing notebook built 3s (48,000-sample)
   windows; AASIST's config (`nb_samp`) requires exactly 64,600 samples per
   input. `preprocessing.py` now derives its window size from
   `AASIST_NB_SAMP = 64600` instead of a hardcoded 3.0s. `model.py` also
   raises a clear error if it's ever handed a window of the wrong length,
   instead of letting a cryptic tensor-shape error surface from inside the
   model. **Confirm 64,600 is still the value your ML team is training/
   evaluating against** - if they change `nb_samp` in their config, update
   `AASIST_NB_SAMP` here to match.

2. **Dead checkpoint URL.** The AASIST notebook's `!wget` pointed at a
   Dropbox link that's no longer valid. The working checkpoint is the one
   committed directly in `clovaai/aasist` at `models/weights/AASIST.pth` -
   that's what's bundled in `weights/` here.

3. **Model loaded once, not per-request.** `model.py` loads the model at
   import time and caches it in a module-level global. The notebook's
   pattern of running the load cells fresh each time would add multi-second
   latency (and eventually OOM) if repeated per request.

4. **Kaggle-specific code removed.** No `!git clone`, `!pip install -r
   requirements.txt`, `/kaggle/input/...` paths, or top-level notebook
   execution - everything is now importable functions.

## What was tested (in this sandbox, CPU-only)

- Model loads and reports `nb_samp = 64600` correctly.
- `aasist_inference()` runs a dummy 64,600-sample tensor through the model
  and returns a valid bonafide score.
- Feeding a 48,000-sample tensor correctly raises the descriptive
  `ValueError` instead of a raw shape mismatch.
- `GET /health` returns 200 with device + expected input length.
- `POST /analyze` with a synthetic WAV runs the full pipeline (load -> mono
  -> resample -> normalize -> VAD -> windowing -> inference) without errors
  and returns a well-formed (empty, since the test clip was a pure tone
  with no real speech) response.

**Not tested here:** real speech audio (I don't have a way to source a
speech sample in this environment), GPU execution, and actual browser ->
service network calls (CORS is configured but only verified by reading the
FastAPI/Starlette docs, not by hitting it from a browser). Run `/analyze`
against your genuine/cloned/mismatch/replay scenario clips before the demo
to confirm VAD picks up real speech and the scores look sane.

## Running it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/analyze \
  -F "audio=@your_clip.wav"
```

## Wiring up the React dashboard

Replace your mocked scoring call with something like:

```js
async function analyzeClip(audioBlob) {
  const formData = new FormData();
  formData.append("audio", audioBlob, "clip.wav");

  const res = await fetch("http://localhost:8000/analyze", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Analysis failed: ${res.status}`);
  return res.json();
}
```

The response shape (`num_windows`, `windows[]` with `start_time`/`end_time`/
`bonafide_score`, `mean_bonafide_score`, `min_bonafide_score`) is a
reasonable default - rename fields in `main.py`'s `WindowScore`/
`AnalyzeResponse` models to match whatever your dashboard's timeline/risk
components currently expect from the mock, so the swap is a one-line
change on the frontend.

## Still open

- **REST vs. streaming.** This is a file-upload endpoint (`POST /analyze`),
  which is the safer bet for a demo. If you need true live-mic streaming, a
  WebSocket endpoint that buffers incoming chunks and emits a score every
  time enough samples accumulate for one window is the next step - not
  included here since it's a bigger surface area to debug under time
  pressure.
- **Hop size.** Preprocessing still uses a 0.5s hop against a 4.04s window,
  which is ~87% overlap between consecutive windows - a lot of redundant
  inference calls. Worth widening once you know your latency budget.
- **ECAPA-TDNN / temporal fusion / risk engine** from your original pipeline
  design aren't part of either notebook and aren't implemented here - this
  service only covers capture-through-AASIST (bonafide scoring), matching
  what your friends have actually shared so far.
