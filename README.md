# TP10_AI - Lexical Bridge Word Difficulty API

A FastAPI service that predicts whether an English word is easy or difficult for children at a given age threshold, using a logistic regression model trained on Age of Acquisition data.


## Tech Stack

| | |
|--|--|
| Language | Python 3.11 |
| Framework | FastAPI |
| Inference | NumPy (logistic regression, no sklearn at runtime) |
| Word Frequency | wordfreq |
| Deployment | Render |


## Project Structure

main.py                              # FastAPI app (inference + validation + rate limiting)
requirements.txt
model/
└── word_difficulty_model.json       # Trained artifact (weights, scaler, threshold)
train/
└── notebooks/
    └── Logistic_Regression.ipynb    # Training notebook (Google Colab)


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
  "probability": 0.751,
  "label": "difficult",
  "message": "This word is likely unfamiliar to children aged 7"
}
```

**Probability bands**

| Range | Message |
|---|---|
| < 0.25 | Most children aged 7 would likely know this word |
| 0.25 - 0.40 | Children aged 7 would likely know this word |
| 0.40 - 0.60 | This word may or may not be familiar to children aged 7 |
| 0.60 - 0.75 | This word may be unfamiliar to children aged 7 |
| ≥ 0.75 | This word is likely unfamiliar to children aged 7 |

All routes apply IP-based rate limiting (60 req/min).


## Model

Binary logistic regression trained on the Kuperman AoA dataset (~4,253 words after filtering).

**Features**

| Feature | Description |
|---|---|
| `n_letters` | Character count |
| `n_syll_est` | Rule-based syllable estimate |
| `zipf_score` | Word frequency (Zipf scale) |
| `vowel_ratio` | Vowel-to-letter ratio |
| `max_consonant_run` | Longest consecutive consonant sequence |

**Label**: `1 (difficult)` if AoA ≥ 7, else `0 (easy)`

**Test set performance** (threshold = 0.50): Accuracy 0.696 · F1 0.700 · MCC 0.391


## Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Server runs at `http://127.0.0.1:8000`.


## Architecture Notes
- The model provides an indicative difficulty estimate only. It is not intended to diagnose dyslexia, assess a child's reading ability, or replace professional judgement.
- Model weights are stored as plain JSON and loaded once at startup.
- Both the raw and base-normalised form of a word are scored. The final probability is their average.
- Input is validated (max 50 chars, must contain a letter) before any inference runs.
