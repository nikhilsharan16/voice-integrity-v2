"""
AASIST model loading and inference.

Adapted from alooudit/AASIST_NOT_FT (aasist-pretrained.ipynb). Changes made
when porting out of the notebook:

  - The model and checkpoint are loaded once, at import time, into module
    level globals - not reloaded on every request. Loading + moving weights
    to device is the expensive part; doing it per-request would tank
    latency and eventually memory.
  - Paths point at the weights/ directory bundled with this service instead
    of a Kaggle input path. The original notebook's `!wget` for AASIST.pth
    pointed at a dead Dropbox link - the working checkpoint is the one
    committed inside clovaai/aasist itself, under models/weights/AASIST.pth.
  - aasist_inference() now takes a numpy array as an alternative to a
    tensor, and raises immediately (with a clear message) if it's handed a
    window of the wrong length instead of letting a cryptic shape error
    surface from inside the model.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .aasist_model.aasist import Model

_SERVICE_ROOT = Path(__file__).resolve().parent.parent
_WEIGHTS_DIR = _SERVICE_ROOT / "weights"
_CONFIG_PATH = _WEIGHTS_DIR / "AASIST.conf"
_CHECKPOINT_PATH = _WEIGHTS_DIR / "AASIST.pth"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_nb_samp = None


def load_model():
    """
    Load the AASIST architecture + pretrained weights once. Safe to call
    multiple times - only loads on the first call.
    """
    global _model, _nb_samp

    if _model is not None:
        return _model

    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"AASIST config not found at {_CONFIG_PATH}")
    if not _CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"AASIST checkpoint not found at {_CHECKPOINT_PATH}. "
            "Copy models/weights/AASIST.pth from clovaai/aasist into this "
            "service's weights/ directory."
        )

    with open(_CONFIG_PATH, "r") as f:
        config = json.load(f)
    model_config = config["model_config"]
    _nb_samp = model_config["nb_samp"]

    model = Model(model_config).to(_device)
    state_dict = torch.load(_CHECKPOINT_PATH, map_location=_device)
    model.load_state_dict(state_dict)
    model.eval()

    _model = model
    return _model


def get_expected_input_length() -> int:
    """Number of samples AASIST expects per window (64,600 for AASIST.conf)."""
    if _nb_samp is None:
        load_model()
    return _nb_samp


def aasist_inference(audio_window) -> dict:
    """
    Run one window through AASIST.

    audio_window: 1-D torch.Tensor or numpy array of exactly
                  get_expected_input_length() samples.

    Returns dict with:
      bonafide_logit: raw class-1 logit
      bonafide_probability: softmax probability for the bona-fide class (0-1)
      bonafide_score: bonafide_probability * 100 (0-100 scale)
    """
    model = load_model()

    if isinstance(audio_window, np.ndarray):
        audio_tensor = torch.from_numpy(audio_window)
    else:
        audio_tensor = audio_window

    audio_tensor = audio_tensor.float().reshape(-1)

    expected_len = get_expected_input_length()
    if audio_tensor.shape[0] != expected_len:
        raise ValueError(
            f"AASIST expects exactly {expected_len} samples per window, "
            f"got {audio_tensor.shape[0]}. Check preprocessing.AASIST_NB_SAMP "
            "is wired to the same value as this model's config."
        )

    audio_tensor = audio_tensor.to(_device).unsqueeze(0)

    with torch.no_grad():
        _, logits = model(audio_tensor)

    bonafide_logit = logits[0, 1]
    probability = torch.softmax(logits, dim=1)[0, 1]

    return {
        "bonafide_logit": bonafide_logit.item(),
        "bonafide_probability": probability.item(),
        "bonafide_score": probability.item() * 100,
    }


# Load at import time so the first request doesn't pay the load cost.
# Comment this out during local dev if you want faster reload cycles.
load_model()
