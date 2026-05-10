# TP10 AI

AI components for the Lexical Bridge application — a reading support tool designed for children with dyslexia.

This repository contains three AI modules, each addressing a different feature of the product.

---

## Modules

| Module | Task | Algorithm |
|---|---|---|
| [`word_difficulty_model`](./word_difficulty_model/) | Predict word difficulty (Age of Acquisition) | Ridge Regression |
| [`library_clustering`](./library_clustering/) | Organize children's stories by topic and reading level | Hard-EM / Soft-EM |
| [`photo_mission_image_classification`](./photo_mission_image_classification/) | Classify uploaded photos by object class | CNN (PyTorch) |

---

## Project Structure

```text
main.py                                          # FastAPI inference server (word difficulty)
requirements.txt
word_difficulty_model/
├── data/                                        # AoA dataset
├── model/                                       # Trained model artifact (JSON)
├── notebooks/                                   # Training notebooks (Google Colab)
│   └── experiments/                             # Earlier experiment notebooks
├── outputs/                                     # Training plots
└── src/
    └── Ridge_Regression.py                      # Training script

library_clustering/
├── data/                                        # GlotStoryBook dataset
├── notebooks/
│   └── Document_Clustering.ipynb               # Clustering notebook (Google Colab)
├── outputs/                                     # Cluster plots and exported JSON
└── src/
    └── document_clustering.py                   # Clustering script

photo_mission_image_classification/
├── data/
│   ├── notebooks/                               # Dataset collection and analysis
│   └── src/                                     # Data collection and analysis scripts
└── cnn/
    └── training/
        └── baseline/
            ├── cnn_baseline.py                  # Training script
            └── cnn_baseline.ipynb               # Training notebook (Google Colab)
```

---

## Word Difficulty Model

Estimates the Age of Acquisition (AoA) of an English word and converts the prediction into a broad difficulty category relative to a target child age (default: 7 years old).

Deployed as a FastAPI service on Render.

**Features used:**

| Feature | Description |
|---|---|
| `n_letters` | Number of letters |
| `n_syll_est` | Rule-based syllable estimate |
| `zipf_score` | Word frequency from `wordfreq` |
| `vowel_ratio` | Ratio of vowels to total letters |
| `max_consonant_run` | Longest consecutive consonant sequence |

**Output categories:**

| Category | Condition |
|---|---|
| `very_likely_familiar` | Predicted AoA ≤ target age − 2.0 |
| `likely_familiar` | Predicted AoA ≤ target age − 0.5 |
| `around_target_age` | Within ±0.5 of target age |
| `likely_unfamiliar` | Predicted AoA < target age + 2.0 |
| `very_likely_unfamiliar` | Predicted AoA ≥ target age + 2.0 |

See [`word_difficulty_model/`](./word_difficulty_model/) for training details.

---

## Library Clustering

Automatically organizes children's stories from the GlotStoryBook dataset into topic categories and reading difficulty levels for the in-app reading library.

- **Topic clustering:** Hard-EM and Soft-EM on a bag-of-words document-term matrix (K=8 clusters, manually mapped to 3 categories)
- **Difficulty levels:** Dale-Chall Readability Score → Level 1 / 2 / 3
- **Output:** `stories_final.json` structured by category and difficulty, ready for frontend import

**Categories:** `ANIMALS` · `NATURE` · `DAILY_LIFE`

See [`library_clustering/`](./library_clustering/) for details.

---

## Photo Mission Image Classification

Trains a baseline CNN to classify uploaded images into one of nine object classes for the photo mission game feature. A child is asked to photograph a specific object, and the model checks whether the uploaded image matches the requested class.

**Classes:** `cat` · `dog` · `bird` · `flower` · `fruit` · `vegetable` · `car` · `bicycle` · `shoe`

**Architecture:** 4-block CNN (Conv2d → ReLU → MaxPool) + fully connected classifier, trained with CrossEntropyLoss and Adam.

See [`photo_mission_image_classification/`](./photo_mission_image_classification/) for details.

---

## API (Word Difficulty)

The word difficulty model is served via FastAPI.

### `GET /`

Health check.

### `POST /predict`

**Request**
```json
{ "word": "elephant" }
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

Rate limit: 60 requests per minute per IP.

---

## Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

```text
http://127.0.0.1:8000
```

---

## Tech Stack

| | |
|---|---|
| Language | Python 3.11 |
| API Framework | FastAPI |
| Word Difficulty Inference | NumPy (no sklearn at runtime) |
| CNN Training | PyTorch |
| Clustering | NumPy (custom Hard-EM / Soft-EM) |
| Word Frequency | wordfreq |
| Readability | textstat |
| Deployment | Render |
