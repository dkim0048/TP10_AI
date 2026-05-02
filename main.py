import re
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

import wordfreq

# Set the path of the saved ridge regression model artifact
MODEL_PATH = Path(__file__).parent / "model" / "word_difficulty_model.json"
# Load the saved model artifact from JSON
try:
    with open(MODEL_PATH, "r") as f:
        artifact = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")
except json.JSONDecodeError:
    raise RuntimeError(f"Model file is corrupted: {MODEL_PATH}")

# Load model weights and configuration
W = np.array(artifact["w"], dtype=float)
B = float(artifact["b"])
# Load scaler values used during training
SCALER_MEAN = np.array(artifact["scaler_mean"], dtype=float)
SCALER_SCALE = np.array(artifact["scaler_scale"], dtype=float)
# Load target age and AoA clamp range
TARGET_AGE = float(artifact["target_age"])
MIN_PRED_AOA = float(artifact["min_pred_aoa"])
MAX_PRED_AOA = float(artifact["max_pred_aoa"])
FEATURES = artifact["features"]
if artifact.get("model_type") != "ridge_regression":
    raise RuntimeError("Invalid model artifact: expected ridge_regression")

if W.shape[0] != len(FEATURES):
    raise RuntimeError(
        f"Model feature mismatch: weights={W.shape[0]}, features={len(FEATURES)}"
    )

if SCALER_MEAN.shape[0] != len(FEATURES) or SCALER_SCALE.shape[0] != len(FEATURES):
    raise RuntimeError("Scaler feature mismatch with feature list")

if np.any(SCALER_SCALE == 0):
    raise RuntimeError("Scaler contains zero scale value")

# Deployment safety limits
MAX_WORD_LENGTH = 50
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW = 60  # seconds

# In-memory rate limit store
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
# Create FastAPI application
app = FastAPI(title="Lexical Bridge Word Difficulty API")
# Allow frontend requests to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
# Simple IP-based rate limiting middleware
@app.middleware("http")
async def rate_limit(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    # Keep only requests made within the current time window
    timestamps = _rate_limit_store[ip]
    timestamps[:] = [t for t in timestamps if t > window_start]
    # Reject request if the client exceeds the request limit
    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please try again later."}
        )
    # Store current request timestamp and continue
    timestamps.append(now)
    return await call_next(request)

# Catch unexpected server errors and return a safe response
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."}
    )

# Request body format for word prediction
class PredictRequest(BaseModel):
    word: str
    # Validate user input before prediction
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

# Response body format for word prediction
class PredictResponse(BaseModel):
    word: str
    normalized_word: str
    predicted_aoa: float
    category: str
    message: str

# Extract only lowercase alphabet characters
def clean_word(word):
    return re.sub(r"[^a-z]", "", str(word).lower().strip())

# Estimate syllable count using rule-based vowel group detection
def estimate_syllables(word):
    w = clean_word(word)
    if not w:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    # Count a new syllable when a new vowel group starts
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    # Remove silent trailing e
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)

# Extract linguistic features from a single word
def build_features(word):
    cw = clean_word(word)
    n_letters = len(cw)
    n_syll_est = estimate_syllables(cw)
    zipf_score = wordfreq.zipf_frequency(cw, "en")
    vowels = "aeiouy"
    # Calculate vowel ratio
    n_vowels = sum(1 for ch in cw if ch in vowels)
    vowel_ratio = n_vowels / n_letters if n_letters > 0 else 0.0
    # Calculate maximum consecutive consonant run
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
    # Return features in the same order used during training
    return np.array([feature_dict[f] for f in FEATURES], dtype=float)

# Normalize simple plural or inflected word forms
def normalize_word_form(word):
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

# Predict continuous AoA using saved scaler and ridge regression weights
def predict_aoa(x):
    x_scaled = (x - SCALER_MEAN) / SCALER_SCALE
    raw = float(x_scaled @ W + B)
    return float(np.clip(raw, MIN_PRED_AOA, MAX_PRED_AOA))

# Convert predicted AoA into a broad difficulty category
def aoa_category(pred_aoa):
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

# Convert predicted AoA into a user-friendly message
def aoa_message(pred_aoa):
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

# Health check endpoint for deployment testing
@app.get("/")
def health():
    return {
        "status": "ok",
        "message": "Lexical Bridge Word Difficulty API is running"
    }

# Word difficulty prediction endpoint
@app.post("/predict", response_model=PredictResponse)
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
        message=aoa_message(aoa_final)
    )

