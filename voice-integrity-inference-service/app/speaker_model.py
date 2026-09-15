"""
Speaker verification via ECAPA-TDNN, adapted from
Tanmay-1404/ecapa-not-automated (enroll.py + identify.py).

Changes made when porting out of the standalone scripts:

  - enroll.py read a folder of per-speaker audio files and wrote a single
    speakers.pkl in one batch run. Here, enrollment happens one speaker/
    sample at a time via POST /enroll (see main.py), and each new sample
    is averaged into that speaker's existing template (if any) rather than
    requiring every sample to be present up front.
  - identify.py hardcoded a 16kHz requirement and raised if the input
    didn't match. Real uploads won't reliably be exactly 16kHz, so audio
    here goes through the same mono + resample helpers as the AASIST
    pipeline (preprocessing.to_mono / preprocessing.resample) instead of
    rejecting anything off-rate.
  - The ECAPA classifier is loaded once at import time (same pattern as
    model.py), not reloaded per request.
  - The speaker database persists to disk (database/speakers.pkl, same
    filename the original scripts used) so enrollments survive a server
    restart.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier

from . import preprocessing

_SERVICE_ROOT = Path(__file__).resolve().parent.parent
_DATABASE_DIR = _SERVICE_ROOT / "database"
_DATABASE_FILE = _DATABASE_DIR / "speakers.pkl"

_classifier = None
_speaker_db: Dict[str, torch.Tensor] = {}
# How many raw samples have been averaged into each speaker's template so
# far - needed to correctly weight a new sample into a running average
# instead of just averaging the two most recent ones together.
_sample_counts: Dict[str, int] = {}


def load_classifier():
    """Load the ECAPA-TDNN classifier once. Safe to call repeatedly."""
    global _classifier
    if _classifier is None:
        _classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    return _classifier


def _load_database():
    global _speaker_db, _sample_counts
    if _DATABASE_FILE.exists():
        with open(_DATABASE_FILE, "rb") as f:
            saved = pickle.load(f)
        # Support both the plain {name: embedding} format the original
        # enroll.py wrote, and this service's {name: (embedding, count)}.
        for name, value in saved.items():
            if isinstance(value, tuple):
                _speaker_db[name], _sample_counts[name] = value
            else:
                _speaker_db[name] = value
                _sample_counts[name] = 1


def _save_database():
    _DATABASE_DIR.mkdir(exist_ok=True)
    combined = {name: (_speaker_db[name], _sample_counts[name]) for name in _speaker_db}
    with open(_DATABASE_FILE, "wb") as f:
        pickle.dump(combined, f)


def compute_embedding(audio: torch.Tensor, original_sr: int) -> torch.Tensor:
    """
    Mono + resample to 16kHz (matching the AASIST pipeline's helpers), then
    run through ECAPA to get a normalized 192-dim speaker embedding.
    """
    classifier = load_classifier()

    audio = preprocessing.to_mono(audio)
    audio = preprocessing.resample(audio, original_sr, preprocessing.TARGET_SR)

    with torch.no_grad():
        embedding = classifier.encode_batch(audio)  # [1, 1, 192]

    embedding = embedding.squeeze()
    embedding = F.normalize(embedding, p=2, dim=0)
    return embedding


def enroll_speaker(name: str, audio: torch.Tensor, original_sr: int) -> dict:
    """
    Add one audio sample for `name` to the speaker database. If the
    speaker already has samples enrolled, the new embedding is averaged
    into the existing template (weighted by sample count so far) and
    re-normalized - same averaging idea as the original enroll.py, just
    incremental instead of batch.
    """
    embedding = compute_embedding(audio, original_sr)

    if name in _speaker_db:
        prior_count = _sample_counts[name]
        combined = _speaker_db[name] * prior_count + embedding
        new_count = prior_count + 1
        combined = F.normalize(combined / new_count, p=2, dim=0)
        _speaker_db[name] = combined
        _sample_counts[name] = new_count
    else:
        _speaker_db[name] = embedding
        _sample_counts[name] = 1

    _save_database()

    return {"name": name, "samples_enrolled": _sample_counts[name]}


def identify_speaker(audio: torch.Tensor, original_sr: int) -> dict:
    """
    Compare incoming audio against every enrolled speaker (cosine
    similarity, same as identify.py's torch.dot on normalized embeddings)
    and return the ranked results plus the best match.

    Returns:
      best_speaker: name of the closest match, or None if nobody's enrolled
      best_similarity: raw cosine similarity of the best match, -1 to 1
      match_score: best_similarity rescaled to 0-100 for the dashboard
      all_scores: {name: similarity} for every enrolled speaker
    """
    if not _speaker_db:
        return {
            "best_speaker": None,
            "best_similarity": None,
            "match_score": None,
            "all_scores": {},
        }

    incoming = compute_embedding(audio, original_sr)

    scores: Dict[str, float] = {}
    for speaker_name, speaker_embedding in _speaker_db.items():
        score = torch.dot(incoming, speaker_embedding).item()
        scores[speaker_name] = score

    results: List[Tuple[str, float]] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_speaker, best_similarity = results[0]

    # Cosine similarity for normalized ECAPA embeddings is typically in a
    # much narrower band than the full [-1, 1] range in practice (genuine
    # same-speaker comparisons often land ~0.4-0.9). A flat linear rescale
    # of [-1, 1] -> [0, 100] will under-report confident matches - treat
    # match_score as a rough dashboard-friendly indicator, not a calibrated
    # probability, until you've checked real score distributions for your
    # registered speakers and adjusted this mapping (or picked a proper
    # decision threshold) accordingly.
    match_score = max(0, min(100, round((best_similarity + 1) * 50)))

    return {
        "best_speaker": best_speaker,
        "best_similarity": best_similarity,
        "match_score": match_score,
        "all_scores": scores,
    }


def list_enrolled_speakers() -> List[dict]:
    return [{"name": name, "samples_enrolled": _sample_counts[name]} for name in _speaker_db]


# Load the database at import time so /identify works immediately if
# speakers were enrolled in a previous run. The classifier itself is NOT
# loaded here - it's a much bigger download/load, so it's deferred to
# first use (see load_classifier()) rather than paid at every server
# startup regardless of whether enrollment/identification is used yet.
_load_database()
