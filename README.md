# TP10_AI - Lexical Bridge Word Difficulty API

A FastAPI service that estimates the Age of Acquisition (AoA) of an English word and converts the prediction into a cautious age-level difficulty category for children around a selected target age.

The model is trained on Age of Acquisition data using Ridge Regression. It does not diagnose dyslexia or assess a child's reading ability. It only provides a broad word-level difficulty indicator.

---

## Tech Stack

| | |
|--|--|
| Language | Python 3.11 |
| Framework | FastAPI |
| Inference | NumPy (Ridge Regression, no sklearn at runtime) |
| Word Frequency | wordfreq |
| Deployment | Render |

---

## Project Structure

```text
main.py                              # FastAPI app (inference + validation + rate limiting)
requirements.txt
model/
└── word_difficulty_model.json       # Trained Ridge Regression artifact
train/
└── notebooks/
    └── Ridge_Regression.ipynb       # Training notebook (Google Colab)
```

---

## API Endpoints

### `GET /`

Health check endpoint.

**Response**

```json
{
  "status": "ok",
  "message": "Lexical Bridge Word Difficulty API is running"
}
```

---

### `POST /predict`

Predicts the estimated Age of Acquisition of a word and returns a broad age-level difficulty category.

**Request**

```json
{
  "word": "elephant"
}
```

**Response**

```json
{
  "word": "elephant",
  "normalized_word": "elephant",
  "predicted_aoa": 8.34,
  "category": "likely_unfamiliar",
  "message": "This word may be unfamiliar to children aged 7"
}
```

---

## Output Categories

The model predicts a continuous AoA value. This value is compared with the selected target age, currently 7 years old, and converted into a cautious user-facing category.

| Predicted AoA compared with target age | Category | Message |
|---|---|---|
| ≤ target age - 2.0 | `very_likely_familiar` | Most children aged 7 would likely know this word |
| target age - 2.0 to target age - 0.5 | `likely_familiar` | Children aged 7 would likely know this word |
| target age - 0.5 to target age + 0.5 | `around_target_age` | This word may be around the expected level for children aged 7 |
| target age + 0.5 to target age + 2.0 | `likely_unfamiliar` | This word may be unfamiliar to children aged 7 |
| ≥ target age + 2.0 | `very_likely_unfamiliar` | This word is likely unfamiliar to children aged 7 |

All routes apply IP-based rate limiting: 60 requests per minute.

---

## Model

The final model uses Ridge Regression to predict a continuous Age of Acquisition value from word-level linguistic features.

Ridge Regression was selected because the target variable, `AoA_Kup`, is continuous. Instead of converting AoA into only easy/difficult labels, the model predicts an estimated AoA score and then converts it into broad support categories.

The predicted AoA should not be interpreted as an exact developmental age. It is used only as a broad word-level reference.

---

## Training Pipeline

```text
raw word
→ cleaning
→ feature extraction
→ standardization
→ Ridge Regression
→ predicted AoA
→ age-level category/message
```

---

## Features

The model only uses features that can be extracted from a single input word. This is important because the user enters one word at a time, and the system does not rely on sentence context, reading history, or child-specific data.

| Feature | Description |
|---|---|
| `n_letters` | Number of letters in the cleaned word |
| `n_syll_est` | Rule-based syllable estimate |
| `zipf_score` | Word frequency score from `wordfreq` |
| `vowel_ratio` | Ratio of vowels to total letters |
| `max_consonant_run` | Longest consecutive consonant sequence |

**Target:** Continuous `AoA_Kup` value  
**Model output:** Predicted AoA  
**User-facing output:** Age-level category and message

---

## Example Predictions

| Word | Predicted AoA | Category |
|---|---:|---|
| `the` | 3.33 | `very_likely_familiar` |
| `cat` | 5.91 | `likely_familiar` |
| `dog` | 5.63 | `likely_familiar` |
| `apple` | 6.25 | `likely_familiar` |
| `strength` | 6.55 | `around_target_age` |
| `beautiful` | 7.78 | `likely_unfamiliar` |
| `elephant` | 8.34 | `likely_unfamiliar` |
| `metamorphosis` | 11.04 | `very_likely_unfamiliar` |
| `dyslexia` | 9.33 | `very_likely_unfamiliar` |
| `meaningful` | 8.65 | `likely_unfamiliar` |

---

## Model Artifact

The trained model is saved as:

```text
model/word_difficulty_model.json
```

The artifact stores:

```text
model_type
weights
bias
scaler mean
scaler scale
target age
minimum / maximum predicted AoA range
feature order
```

The API loads this artifact once at startup and performs inference using NumPy.

---

## Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Server runs at:

```text
http://127.0.0.1:8000
```

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"word\":\"elephant\"}"
```

---

## Deployment

| Environment | Start command | URL |
|---|---|---|
| Local | `uvicorn main:app --reload` | `http://127.0.0.1:8000` |
| Render | `uvicorn main:app --host 0.0.0.0 --port $PORT` | `https://tp10-ai.onrender.com` |

---

## Architecture Notes

- The model provides an indicative word-level difficulty estimate only.
- It is not intended to diagnose dyslexia, assess a child's reading ability, or replace professional judgement.
- The API predicts the difficulty of the word, not the condition or ability of the child.
- Model weights, scaler parameters, feature order, target age, and AoA clamp range are stored in JSON and loaded once at startup.
- Both the raw word and a simple base-normalised form may be scored. If the normalised form differs from the original word, the final predicted AoA is the average of both predictions.
- Input is validated before inference. The input must be no more than 50 characters and contain at least one alphabetic character.
- The API returns cautious messages such as “likely familiar”, “around target age”, and “likely unfamiliar” to avoid overclaiming.

---