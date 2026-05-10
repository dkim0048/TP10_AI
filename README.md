# TP10 AI

AI components for the Lexical Bridge application — a reading support tool designed for children with dyslexia.

This repository contains three AI modules, each addressing a different feature of the product.

---

## Background

Parents of children with dyslexia often face a common problem: when their child struggles with a word, it is hard to tell whether the difficulty comes from dyslexia itself or whether the word is simply hard for most children that age. Without a reference point, parents cannot easily distinguish between the two.

Beyond that, many parents want to read alongside their child but lack suitable materials, and children with reading difficulties often avoid reading altogether — making sustained engagement a key challenge.

The three modules in this repository each address one of these problems.

---

## Modules

| Module | Problem | Algorithm |
|---|---|---|
| [`word_difficulty_model`](./word_difficulty_model/) | Is this word hard because of dyslexia, or is it hard for most kids? | Ridge Regression |
| [`library_clustering`](./library_clustering/) | Parents need accessible reading material to share with their child | Hard-EM / Soft-EM |
| [`photo_mission_image_classification`](./photo_mission_image_classification/) | Children with reading difficulties tend to avoid reading | CNN (PyTorch) |

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

**Problem:** When a child struggles with a word, parents cannot easily tell whether it is because of dyslexia or because the word is genuinely difficult for most children that age. This makes it hard to offer the right kind of support.

**Solution:** A parent types in a word, and the model estimates how difficult it generally is for children around a given age, based on Age of Acquisition (AoA) data. The output is a broad, cautious category — not a diagnosis — that helps parents understand whether the word is expected to be hard for most kids or not.

The model is trained on AoA data using Ridge Regression and deployed as a FastAPI service on Render. It uses only word-level features that can be extracted from a single input word, with no sentence context or child-specific data required.

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

**Problem:** Parents who want to read alongside their child often lack suitable reading material — content that is age-appropriate, accessible, and actually available.

**Solution:** An in-app reading library built from the open [GlotStoryBook](https://github.com/cisnlp/GlotStoryBook) dataset. Stories are filtered to English-language, CC-BY licensed texts of at least 50 words, then organized by topic and reading difficulty so parents can quickly find a story that fits.

- **Dataset:** GlotStoryBook (English, CC-BY, ≥ 50 words)
- **Topic clustering:** Hard-EM and Soft-EM on a bag-of-words document-term matrix (K=8 clusters, manually mapped to 3 categories)
- **Difficulty levels:** Dale-Chall Readability Score → Level 1 / 2 / 3
- **Output:** `stories_final.json` structured by category and difficulty, ready for frontend import

**Categories:** `ANIMALS` · `NATURE` · `DAILY_LIFE`

See [`library_clustering/`](./library_clustering/) for details.

---

## Photo Mission Image Classification

**Problem:** Children with reading difficulties often avoid reading as a coping strategy, making sustained engagement difficult.

**Solution:** A photo-taking game that encourages interaction without putting reading at the centre. A child is given a prompt to photograph a specific object, and the model checks whether the uploaded image matches the requested class. This serves as an engagement mechanic that keeps children active in the app without requiring them to read.

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

## Deployment

The word difficulty API is deployed on Render and publicly accessible at:

```text
https://tp10-ai.onrender.com
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
