
"""
Voice-integrity inference service.

Wraps preprocessing.py + model.py + speaker_model.py behind a small FastAPI
app. Run with:

    uvicorn app.main:app --reload --port 8000

Endpoints:
  GET  /health          - liveness check + model info
  POST /analyze         - upload audio and get per-window bonafide scores
                          plus speaker identification
  POST /enroll          - register an audio sample for a named speaker
  GET  /speakers         - list currently enrolled speakers
"""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import model, preprocessing, speaker_model

app = FastAPI(title="Voice Integrity Inference Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    # AASIST is loaded at import time in model.py.
    # Confirm it and fail fast if the checkpoint is missing.
    # ECAPA is loaded only when speaker identification is needed.
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
        raise HTTPException(
            status_code=400,
            detail="Speaker name is required."
        )

    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty upload."
        )

    try:
        waveform, original_sr = preprocessing.load_audio_bytes(audio_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio: {exc}"
        )

    result = speaker_model.enroll_speaker(name, waveform, original_sr)

    return EnrollResponse(**result)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty upload."
        )

    # Audio preprocessing
    try:
        result = preprocessing.preprocess_audio_bytes(audio_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Preprocessing failed: {exc}"
        )

    # Speaker identification:
    # Only load/use ECAPA when at least one speaker is enrolled.
    enrolled_speakers = speaker_model.list_enrolled_speakers()

    if enrolled_speakers:
        try:
            waveform, original_sr = preprocessing.load_audio_bytes(audio_bytes)

            speaker_result = speaker_model.identify_speaker(
                waveform,
                original_sr
            )

            speaker_match = SpeakerMatch(
                best_speaker=speaker_result["best_speaker"],
                best_similarity=speaker_result["best_similarity"],
                match_score=speaker_result["match_score"],
            )

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Speaker identification failed: {exc}"
            )

    else:
        speaker_match = SpeakerMatch(
            best_speaker=None,
            best_similarity=None,
            match_score=None,
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
            raise HTTPException(
                status_code=500,
                detail=str(exc)
            )

        # Convert bonafide probability into spoof probability.
        corrected_prob = 1.0 - inference["bonafide_probability"]

        # Convert bonafide score into spoof score.
        corrected_score = 100.0 - inference["bonafide_score"]

        raw_scores.append(corrected_score)

        # Trailing rolling average over the last SMOOTHING_WINDOW windows.
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
