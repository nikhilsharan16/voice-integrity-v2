"""
Audio preprocessing pipeline for the voice-integrity inference service.

Adapted from HarshChaurasiya1/Audio_Preprocessing_Pipeline
(audio-preprocessing-16khz-slidingwindow.ipynb). Changes made when porting
out of the notebook:

  - Removed Kaggle-specific paths and !pip installs.
  - WINDOW_SECONDS is now derived from AASIST_NB_SAMP (64,600 samples)
    instead of a hardcoded 3.0s. The original notebook's 3s / 48,000-sample
    windows do NOT match what AASIST expects as input (nb_samp: 64600 in
    config/AASIST.conf) - feeding those windows straight into the model will
    raise a shape error. Confirm 64,600 is still correct with whoever owns
    the AASIST training config before relying on this in the final demo.
  - Everything is wrapped in functions that accept in-memory bytes (for a
    FastAPI upload) rather than only a filesystem path.
"""

import io
from pathlib import Path
from typing import List, TypedDict

import numpy as np
import soundfile as sf
import torch
import torchaudio

from silero_vad import get_speech_timestamps, load_silero_vad

TARGET_SR = 16000

# AASIST's fixed input length (see aasist/config/AASIST.conf -> nb_samp).
AASIST_NB_SAMP = 64600

# Hop between windows. The notebook used 0.5s; kept as-is, but at a 4.04s
# window this gives ~87% overlap between consecutive windows, which is a lot
# of redundant inference. Consider widening this (e.g. 1.0-2.0s) once you've
# checked how the demo's latency budget looks end to end.
HOP_SECONDS = 0.5
HOP_SAMPLES = int(TARGET_SR * HOP_SECONDS)

MIN_SPEECH_DURATION_MS = 250
MIN_SILENCE_DURATION_MS = 100
SPEECH_PAD_MS = 50
MIN_SPEECH_RATIO = 0.30

_vad_model = None


def get_vad_model():
    """Lazily load Silero VAD once and cache it at module scope."""
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad()
    return _vad_model


class WindowMeta(TypedDict):
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    speech_ratio: float


def to_mono(audio: torch.Tensor) -> torch.Tensor:
    """Public alias - also used by speaker_model.py for ECAPA input prep."""
    return _to_mono(audio)


def resample(audio: torch.Tensor, original_sr: int, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Public alias - also used by speaker_model.py for ECAPA input prep."""
    return _resample(audio, original_sr, target_sr)


def load_audio_bytes(audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    """Public loader for other modules (e.g. speaker_model.py) that need
    the raw decoded waveform without running the full VAD/windowing
    pipeline."""
    buffer = io.BytesIO(audio_bytes)
    return _load_with_soundfile(buffer)


def _to_mono(audio: torch.Tensor) -> torch.Tensor:
    if audio.shape[0] == 1:
        return audio
    return audio.mean(dim=0, keepdim=True)


def _resample(audio: torch.Tensor, original_sr: int, target_sr: int = TARGET_SR) -> torch.Tensor:
    if original_sr == target_sr:
        return audio
    return torchaudio.functional.resample(audio, orig_freq=original_sr, new_freq=target_sr)


def _normalize(audio: torch.Tensor) -> torch.Tensor:
    peak = torch.max(torch.abs(audio))
    if peak < 1e-8:
        return audio
    return audio / peak


def _create_speech_mask(num_samples: int, speech_timestamps: list) -> torch.Tensor:
    mask = torch.zeros(num_samples, dtype=torch.bool)
    for segment in speech_timestamps:
        start = max(int(segment["start"]), 0)
        end = min(int(segment["end"]), num_samples)
        mask[start:end] = True
    return mask


def _create_sliding_windows(
    audio: torch.Tensor,
    speech_mask: torch.Tensor,
    window_samples: int,
    hop_samples: int,
    min_speech_ratio: float,
) -> tuple[List[torch.Tensor], List[WindowMeta]]:
    total_samples = audio.shape[-1]
    windows: List[torch.Tensor] = []
    metadata: List[WindowMeta] = []

    if total_samples < window_samples:
        padded_audio = torch.zeros(1, window_samples, dtype=audio.dtype)
        padded_audio[:, :total_samples] = audio

        padded_mask = torch.zeros(window_samples, dtype=torch.bool)
        padded_mask[:total_samples] = speech_mask

        ratio = padded_mask.float().mean().item()
        if ratio >= min_speech_ratio:
            windows.append(padded_audio)
            metadata.append(
                {
                    "start_sample": 0,
                    "end_sample": total_samples,
                    "start_time": 0.0,
                    "end_time": total_samples / TARGET_SR,
                    "speech_ratio": ratio,
                }
            )
        return windows, metadata

    start = 0
    while start + window_samples <= total_samples:
        end = start + window_samples
        audio_window = audio[:, start:end]
        mask_window = speech_mask[start:end]
        ratio = mask_window.float().mean().item()

        if ratio >= min_speech_ratio:
            windows.append(audio_window.clone())
            metadata.append(
                {
                    "start_sample": start,
                    "end_sample": end,
                    "start_time": start / TARGET_SR,
                    "end_time": end / TARGET_SR,
                    "speech_ratio": ratio,
                }
            )
        start += hop_samples

    return windows, metadata


def preprocess_audio_tensor(
    audio: torch.Tensor,
    original_sr: int,
    window_samples: int = AASIST_NB_SAMP,
    hop_samples: int = HOP_SAMPLES,
    min_speech_ratio: float = MIN_SPEECH_RATIO,
) -> dict:
    """
    Run the full pipeline (mono -> resample -> normalize -> VAD -> windows)
    on an already-loaded waveform tensor of shape [channels, samples].

    Returns a dict with:
      windows: list[torch.Tensor], each shape [1, window_samples]
      metadata: list[WindowMeta] aligned with windows
      sample_rate: int
      speech_ratio_overall: float
    """
    audio = _to_mono(audio)
    audio = _resample(audio, original_sr, TARGET_SR)
    audio = _normalize(audio)

    waveform = audio.squeeze()
    vad_model = get_vad_model()
    speech_timestamps = get_speech_timestamps(
        waveform,
        vad_model,
        sampling_rate=TARGET_SR,
        min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
        speech_pad_ms=SPEECH_PAD_MS,
    )

    speech_mask = _create_speech_mask(audio.shape[-1], speech_timestamps)
    speech_ratio_overall = speech_mask.float().mean().item() if speech_mask.numel() else 0.0

    windows, metadata = _create_sliding_windows(
        audio, speech_mask, window_samples, hop_samples, min_speech_ratio
    )

    return {
        "windows": windows,
        "metadata": metadata,
        "sample_rate": TARGET_SR,
        "speech_ratio_overall": speech_ratio_overall,
        "num_speech_segments": len(speech_timestamps),
    }


def _load_with_soundfile(source) -> tuple[torch.Tensor, int]:
    """
    Decode audio via soundfile/libsndfile instead of torchaudio.load().

    Recent torchaudio versions route .load() through TorchCodec, which
    needs a matching system FFmpeg install - a common source of pain on
    Windows. soundfile bundles its own libsndfile (no separate FFmpeg
    needed) and its wheel here (1.2.2) decodes wav, flac, ogg, and mp3
    natively, which covers everything this service needs.
    """
    data, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    # soundfile gives [samples, channels]; the rest of this pipeline
    # expects torchaudio's [channels, samples] convention.
    audio = torch.from_numpy(data.T.copy())
    return audio, sample_rate


def preprocess_audio_file(file_path: str, **kwargs) -> dict:
    """Convenience wrapper: load from a path on disk, then preprocess."""
    audio, original_sr = _load_with_soundfile(file_path)
    return preprocess_audio_tensor(audio, original_sr, **kwargs)


def preprocess_audio_bytes(audio_bytes: bytes, **kwargs) -> dict:
    """Convenience wrapper for a FastAPI upload: load from an in-memory buffer."""
    buffer = io.BytesIO(audio_bytes)
    audio, original_sr = _load_with_soundfile(buffer)
    return preprocess_audio_tensor(audio, original_sr, **kwargs)
