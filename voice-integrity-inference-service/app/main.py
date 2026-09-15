"""
Voice-integrity inference service.

Wraps preprocessing.py + model.py + speaker_model.py behind a small FastAPI
app. Run with:

    uvicorn app.main:app --reload --port 8000

Endpoints:
  GET  /health          - liveness check + model info
  POST /analyze         - upload an audio file, get back a per-window
                          bonafide score timeline PLUS a speaker_match
                          block (best enrolled speaker + similarity) -
                          this is the one your React dashboard should
                          call in place of its mocked scoring function
  POST /enroll          - register one audio sample for a named speaker
                          (ECAPA-TDNN embedding, averaged into that
                          speaker's template if they're already enrolled)
  GET  /speakers        - list currently enrolled speakers

The response shape below mirrors what a scenario-simulator dashboard
typically expects (overall verdict + a timeline of window scores). Adjust
field names to match whatever your dashboard's mock currently returns -
that's the main thing that keeps this swap-in low-effort.
"""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import model, preprocessing, speaker_model

app = FastAPI(title="Voice Integrity Inference Service")

# Loosen this to your actual dashboard origin(s) before the demo - "*" is
# fine for local hackathon testing but not for anything beyond that.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Number of recent windows averaged into smoothed_score. At a 0.5s hop this
# is a 2.5s trailing window - short enough to react to a real change in
# speaker, long enough to iron out single-window noise. Tune once you have
# a feel for the fine-tuned model's score distribution.
SMOOTHING_WINDOW = 5


class WindowScore(BaseModel):
    start_time: float
    end_time: float
    speech_ratio: float
    bonafide_score: float
    bonafide_probability: float
    smoothed_score: float


class SpeakerMatch(BaseModel):
    best_speaker: str | None
    best_similarity: float | None
    match_score: float | None


class AnalyzeResponse(BaseModel):
    num_windows: int
    speech_ratio_overall: float
    num_speech_segments: int
    windows: list[WindowScore]
    mean_bonafide_score: float | None
    min_bonafide_score: float | None
    speaker_match: SpeakerMatch


class EnrollResponse(BaseModel):
    name: str
    samples_enrolled: int


@app.on_event("startup")
def startup():
    # AASIST is already loaded at import time in model.py; this just
    # confirms it and fails fast if the checkpoint is missing. ECAPA is
    # deliberately NOT loaded here - see speaker_model.load_classifier()
    # for why - so startup stays fast even before anyone enrolls/identifies.
    model.load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(model._device),
        "expected_input_samples": model.get_expected_input_length(),
        "enrolled_speakers": len(speaker_model.list_enrolled_speakers()),
    }


@app.get("/speakers")
def speakers():
    return {"speakers": speaker_model.list_enrolled_speakers()}


@app.post("/enroll", response_model=EnrollResponse)
async def enroll(name: str = Form(...), audio: UploadFile = File(...)):
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Speaker name is required.")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        waveform, original_sr = preprocessing.load_audio_bytes(audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}")

    result = speaker_model.enroll_speaker(name, waveform, original_sr)
    return EnrollResponse(**result)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        result = preprocessing.preprocess_audio_bytes(audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {exc}")

    # Speaker identification runs once against the whole clip (ECAPA
    # embeddings are meant for longer utterances, not 4s AASIST windows),
    # not per-window like the spoof score. If nobody's enrolled yet, this
    # comes back all-None and the frontend should treat that as "no
    # identity check performed" rather than "no match".
    try:
        waveform, original_sr = preprocessing.load_audio_bytes(audio_bytes)
        speaker_result = speaker_model.identify_speaker(waveform, original_sr)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Speaker identification failed: {exc}")

    speaker_match = SpeakerMatch(
        best_speaker=speaker_result["best_speaker"],
        best_similarity=speaker_result["best_similarity"],
        match_score=speaker_result["match_score"],
    )

    windows = result["windows"]
    metadata = result["metadata"]

    if not windows:
        return AnalyzeResponse(
            num_windows=0,
            speech_ratio_overall=result["speech_ratio_overall"],
            num_speech_segments=result["num_speech_segments"],
            windows=[],
            mean_bonafide_score=None,
            min_bonafide_score=None,
            speaker_match=speaker_match,
        )

    raw_scores: list[float] = []
    window_scores: list[WindowScore] = []
    for window_tensor, meta in zip(windows, metadata):
        try:
            inference = model.aasist_inference(window_tensor.squeeze())
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        # --- UPDATED INVERSION LOGIC ---
        # Probability is a decimal (0 to 1), so subtract from 1.0
        corrected_prob = 1.0 - inference["bonafide_probability"]
        # Score is a percentage (0 to 100), so subtract from 100.0
        corrected_score = 100.0 - inference["bonafide_score"]

        raw_scores.append(corrected_score)
        
        # Trailing rolling average over the last SMOOTHING_WINDOW windows
        # (including this one). Only looks backward so it works the same
        # way in a future streaming/live setup as it does here on a full file.
        recent = raw_scores[-SMOOTHING_WINDOW:]
        smoothed = sum(recent) / len(recent)

        window_scores.append(
            WindowScore(
                start_time=meta["start_time"],
                end_time=meta["end_time"],
                speech_ratio=meta["speech_ratio"],
                bonafide_score=corrected_score,
                bonafide_probability=corrected_prob,
                smoothed_score=smoothed,
            )
        )

    scores = raw_scores

    return AnalyzeResponse(
        num_windows=len(window_scores),
        speech_ratio_overall=result["speech_ratio_overall"],
        num_speech_segments=result["num_speech_segments"],
        windows=window_scores,
        mean_bonafide_score=sum(scores) / len(scores),
        min_bonafide_score=min(scores),
        speaker_match=speaker_match,
    )