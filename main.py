import re
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from PIL import Image
import pytesseract
import pdf2image
import wordfreq
import io

# Set the path of the saved logistic regression model artifact
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
# Load prediction threshold, age threshold, and feature order
BEST_THRESHOLD = float(artifact["best_threshold"])
THRESHOLD_AGE = float(artifact["threshold_age"])
FEATURES = artifact["features"]

# Output message based on predicted probability
HEDGE_BANDS = [
    (0.25, "Most children aged {age} would likely know this word"),
    (0.40, "Children aged {age} would likely know this word"),
    (0.60, "This word may or may not be familiar to children aged {age}"),
    (0.75, "This word may be unfamiliar to children aged {age}"),
    (1.00, "This word is likely unfamiliar to children aged {age}"),
]

# Deployment safety limits
MAX_WORD_LENGTH = 50
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
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
    probability: float
    label: str
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
    w = str(word).lower().strip()
    cw = clean_word(w)
    n_letters = len(cw)
    n_syll_est = estimate_syllables(w)
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

    # Convert -ies to -y, for example babies -> baby
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    # Remove -es, for example boxes -> box
    if len(w) > 4 and w.endswith("es") and w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    # Remove plural -s, for example cats -> cat
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and not w.endswith("sis"):
        return w[:-1]
    return w

# Stable sigmoid function
def sigmoid(z):
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )

# Predict difficult probability using saved scaler and logistic regression weights
def predict_probability(x):
    x_scaled = (x - SCALER_MEAN) / SCALER_SCALE
    z = x_scaled @ W + B
    return float(sigmoid(z))

# Convert predicted probability into a user-friendly message
def hedge_message(p):
    age = int(THRESHOLD_AGE)
    for upper, message in HEDGE_BANDS:
        if p < upper:
            return message.format(age=age)
    return HEDGE_BANDS[-1][1].format(age=age)

# Extract text from an image using Tesseract OCR
def extract_text_from_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image).strip()

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
    norm_word = normalize_word_form(raw_word)
    # Predict probability from the original input word
    raw_features = build_features(raw_word)
    p_raw = predict_probability(raw_features)
    # If normalized form is different, also predict using the normalized word
    if norm_word and norm_word != raw_word:
        norm_features = build_features(norm_word)
        p_norm = predict_probability(norm_features)
        p_final = 0.5 * p_raw + 0.5 * p_norm
    else:
        p_final = p_raw
    # Classify using the tuned threshold selected during validation
    label = "difficult" if p_final >= BEST_THRESHOLD else "easy"
    return PredictResponse(
        word=req.word,
        normalized_word=norm_word,
        probability=round(p_final, 4),
        label=label,
        message=hedge_message(p_final)
    )

# Image text extraction endpoint
@app.post("/extract-image")
async def extract_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size must be 10MB or less")

    image = Image.open(io.BytesIO(contents))
    text = extract_text_from_image(image)

    if not text:
        raise HTTPException(status_code=422, detail="No text could be extracted from the image")

    return {"text": text}

# PDF text extraction endpoint
@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.content_type or file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size must be 10MB or less")

    pages = pdf2image.convert_from_bytes(contents, dpi=200)
    extracted = [extract_text_from_image(page) for page in pages]
    text = "\n\n".join(block for block in extracted if block)

    if not text:
        raise HTTPException(status_code=422, detail="No text could be extracted from the PDF")

    return {"text": text, "pages": len(pages)}
