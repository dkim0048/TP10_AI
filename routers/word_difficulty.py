import re
import json
import numpy as np
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, field_validator

import wordfreq

MODEL_PATH = Path(__file__).parent.parent / "model" / "word_difficulty_model.json"
try:
    with open(MODEL_PATH, "r") as f:
        artifact = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")
except json.JSONDecodeError:
    raise RuntimeError(f"Model file is corrupted: {MODEL_PATH}")

W = np.array(artifact["w"], dtype=float)
B = float(artifact["b"])
SCALER_MEAN = np.array(artifact["scaler_mean"], dtype=float)
SCALER_SCALE = np.array(artifact["scaler_scale"], dtype=float)
TARGET_AGE = float(artifact["target_age"])
MIN_PRED_AOA = float(artifact["min_pred_aoa"])
MAX_PRED_AOA = float(artifact["max_pred_aoa"])
FEATURES = artifact["features"]

if artifact.get("model_type") != "ridge_regression":
    raise RuntimeError("Invalid model artifact: expected ridge_regression")
if W.shape[0] != len(FEATURES):
    raise RuntimeError(f"Model feature mismatch: weights={W.shape[0]}, features={len(FEATURES)}")
if SCALER_MEAN.shape[0] != len(FEATURES) or SCALER_SCALE.shape[0] != len(FEATURES):
    raise RuntimeError("Scaler feature mismatch with feature list")
if np.any(SCALER_SCALE == 0):
    raise RuntimeError("Scaler contains zero scale value")

MAX_WORD_LENGTH = 50

router = APIRouter()


class PredictRequest(BaseModel):
    word: str

    @field_validator("word")
    @classmethod
    def validate_word(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("word must not be empty")
        if len(v) > MAX_WORD_LENGTH:
            raise ValueError(f"word must be {MAX_WORD_LENGTH} characters or fewer")
        if not re.search(r"[a-zA-Z]", v):
            raise ValueError("word must contain at least one alphabetic character")
        return v


class PredictResponse(BaseModel):
    word: str
    normalized_word: str
    predicted_aoa: float
    category: str
    message: str


def clean_word(word: str) -> str:
    return re.sub(r"[^a-z]", "", str(word).lower().strip())


def estimate_syllables(word: str) -> int:
    w = clean_word(word)
    if not w:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def build_features(word: str) -> np.ndarray:
    cw = clean_word(word)
    n_letters = len(cw)
    n_syll_est = estimate_syllables(cw)
    zipf_score = wordfreq.zipf_frequency(cw, "en")
    vowels = "aeiouy"
    n_vowels = sum(1 for ch in cw if ch in vowels)
    vowel_ratio = n_vowels / n_letters if n_letters > 0 else 0.0
    max_consonant_run = 0
    current_run = 0
    for ch in cw:
        if ch not in vowels:
            current_run += 1
            max_consonant_run = max(max_consonant_run, current_run)
        else:
            current_run = 0
    feature_dict = {
        "n_letters": float(n_letters),
        "n_syll_est": float(n_syll_est),
        "zipf_score": float(zipf_score),
        "vowel_ratio": float(vowel_ratio),
        "max_consonant_run": float(max_consonant_run),
    }
    return np.array([feature_dict[f] for f in FEATURES], dtype=float)


def normalize_word_form(word: str) -> str:
    w = clean_word(word)
    if not w:
        return w
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if (
        len(w) > 4
        and w.endswith("s")
        and not w.endswith(("ss", "sis", "ous", "us", "is"))
    ):
        return w[:-1]
    return w


def predict_aoa(x: np.ndarray) -> float:
    x_scaled = (x - SCALER_MEAN) / SCALER_SCALE
    raw = float(x_scaled @ W + B)
    return float(np.clip(raw, MIN_PRED_AOA, MAX_PRED_AOA))


def aoa_category(pred_aoa: float) -> str:
    diff = pred_aoa - TARGET_AGE
    if diff <= -2.0:
        return "very_likely_familiar"
    if diff <= -0.5:
        return "likely_familiar"
    if diff < 0.5:
        return "around_target_age"
    if diff < 2.0:
        return "likely_unfamiliar"
    return "very_likely_unfamiliar"


def aoa_message(pred_aoa: float) -> str:
    diff = pred_aoa - TARGET_AGE
    age = int(TARGET_AGE)
    if diff <= -2.0:
        return f"Most children aged {age} would likely know this word"
    if diff <= -0.5:
        return f"Children aged {age} would likely know this word"
    if diff < 0.5:
        return f"This word may be around the expected level for children aged {age}"
    if diff < 2.0:
        return f"This word may be unfamiliar to children aged {age}"
    return f"This word is likely unfamiliar to children aged {age}"


@router.get("/")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Lexical Bridge Word Difficulty API is running",
    }


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    raw_word = req.word.lower().strip()
    cleaned_word = clean_word(raw_word)
    norm_word = normalize_word_form(raw_word)

    raw_features = build_features(cleaned_word)
    aoa_raw = predict_aoa(raw_features)

    if norm_word and norm_word != cleaned_word:
        norm_features = build_features(norm_word)
        aoa_norm = predict_aoa(norm_features)
        aoa_final = round(0.5 * aoa_raw + 0.5 * aoa_norm, 2)
    else:
        aoa_final = round(aoa_raw, 2)

    return PredictResponse(
        word=req.word,
        normalized_word=norm_word,
        predicted_aoa=aoa_final,
        category=aoa_category(aoa_final),
        message=aoa_message(aoa_final),
    )
