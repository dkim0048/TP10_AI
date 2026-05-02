# TP10_AI - Lexical Bridge Word Difficulty API

A FastAPI service that estimates the Age of Acquisition (AoA) of an English word and converts it into a difficulty category for children at a given age threshold, using a Ridge Regression model trained on Age of Acquisition data.


## Tech Stack

| | |
|--|--|
| Language | Python 3.11 |
| Framework | FastAPI |
| Inference | NumPy (ridge regression, no sklearn at runtime) |
| Word Frequency | wordfreq |
| Deployment | Render |


## Project Structure

main.py                              # FastAPI app (inference + validation + rate limiting)
requirements.txt
model/
├── word_difficulty_model.json       # Trained artifact (weights, scaler, target_age, aoa clamp range)
└── Ridge_Regression.ipynb          # Training notebook (Google Colab)


## API Endpoints

### `GET /`

Health check.

### `POST /predict`


Predicts word difficulty.

**Request**
```json
{ "word": "elephant" }
```

**Response**
```json
{
  "word": "elephant",
  "normalized_word": "elephant",
  "predicted_aoa": 9.23,
  "category": "very_likely_unfamiliar",
  "message": "This word is likely unfamiliar to children aged 7"
}
```

**AoA difference bands** (`diff = predicted_aoa − target_age`)

| diff | category | message |
|---|---|---|
| ≤ −2.0 | `very_likely_familiar` | Most children aged 7 would likely know this word |
| ≤ −0.5 | `likely_familiar` | Children aged 7 would likely know this word |
| < +0.5 | `around_target_age` | This word may be around the expected level for children aged 7 |
| < +2.0 | `likely_unfamiliar` | This word may be unfamiliar to children aged 7 |
| ≥ +2.0 | `very_likely_unfamiliar` | This word is likely unfamiliar to children aged 7 |

All routes apply IP-based rate limiting (60 req/min).


## Model

Ridge Regression (MSE + L2 regularization, gradient descent) trained on the Kuperman AoA dataset (~5,253 words after filtering).

**Pipeline**

```
raw word → feature extraction (5) → standardization → ridge regression → predicted AoA → category
```

**Features**

| Feature | Description |
|---|---|
| `n_letters` | Character count |
| `n_syll_est` | Rule-based syllable estimate |
| `zipf_score` | Word frequency (Zipf scale) |
| `vowel_ratio` | Vowel-to-letter ratio |
| `max_consonant_run` | Longest consecutive consonant sequence |

**Target**: continuous AoA value (Kuperman). Predicted AoA is clipped to [1.0, 20.0] before category assignment.

**Hyperparameters**: α = 0.001, lr = 0.001, epochs = 3000, lr decay = lr / (1 + 0.001 · epoch)


## Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Server runs at `http://127.0.0.1:8000` (default port when using `--reload`).

## Deployment

| Environment | Start command | URL |
|---|---|---|
| Local | `uvicorn main:app --reload` | `http://127.0.0.1:8000` |
| Render | `uvicorn main:app --host 0.0.0.0 --port $PORT` | https://tp10-ai.onrender.com |


## Architecture Notes
- The model provides an indicative difficulty estimate only. It is not intended to diagnose dyslexia, assess a child's reading ability, or replace professional judgement.
- Model weights, scaler parameters, target age, and AoA clamp range are stored as plain JSON and loaded once at startup.
- Both the raw and base-normalised form of a word are scored. The final predicted AoA is their average.
- Input is validated (max 50 chars, must contain a letter) before any inference runs.
