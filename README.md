# TP10 AI

AI components for the Lexical Bridge application - a parent-facing reading support tool for children who experience dyslexia-related reading difficulties.

This repository contains three AI components, each supporting a different feature area of the product.

---

## Background

Parents of children who experience dyslexia-related reading difficulties often face a common problem: when their child struggles with a word, it can be hard to tell whether the word reflects an individual reading difficulty or whether it is generally unfamiliar for children around the same age. Without a reference point, parents may find it difficult to decide how to support the child during reading.

Beyond individual word difficulty, parents may also want to read with their child but lack suitable reading materials. At the same time, children who experience reading difficulties may avoid reading tasks when they feel stressful or discouraging, making sustained engagement an important challenge.

The three modules in this repository address these needs through word difficulty prediction, reading library organization, and photo-based engagement.

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

**Problem:** When a child struggles with a word, parents may find it difficult to tell whether the word is part of their child’s individual reading struggle or whether it is generally unfamiliar for children around the same age. This can make it harder to decide what kind of support to provide.

**Solution:** A parent enters a word, and the model estimates how difficult the word is likely to be for children around a selected age level, based on Age of Acquisition (AoA) data. The output is a broad, cautious category, not a diagnosis that helps parents understand whether the word may be commonly familiar or unfamiliar for children at that age.

The model is trained on AoA data using Ridge Regression and is deployed as a FastAPI service on Render. It uses only word-level features that can be extracted from a single input word, with no sentence context or child-specific data required.

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

**Problem:** Parents who want to read with their child may not always have suitable reading materials prepared, especially content that is accessible, engaging, and legally reusable within the application.

**Solution:** An in app reading library built from the open [GlotStoryBook](https://github.com/cisnlp/GlotStoryBook) dataset. Stories are filtered to English-language, CC-BY licensed texts of at least 50 words, then organised by topic and reading difficulty so parents can quickly find story materials that fit the reading session.

- **Dataset:** GlotStoryBook (English, CC-BY, ≥ 50 words)
- **Topic clustering:** Hard EM and Soft EM on a bag-of-words document-term matrix (K=8 clusters, manually mapped to 3 categories)
- **Difficulty levels:** Dale-Chall Readability Score → Level 1 / 2 / 3
- **Output:** `stories_final.json` structured by category and difficulty, ready for frontend import

**Categories:** `ANIMALS` · `NATURE` · `DAILY_LIFE`

See [`library_clustering/`](./library_clustering/) for details.

---

## Photo Mission Image Classification

**Problem:** Children with reading difficulties may avoid reading tasks when they feel difficult or stressful, which can make sustained engagement challenging.

**Solution:** A photo taking challenge that introduces a playful activity before or alongside reading. The child is given a prompt to photograph a specific object, and the model checks whether the uploaded image matches the requested class. This works as an engagement mechanic that helps children stay active in the app while reducing pressure around reading.

**Classes:** `cat` · `dog` · `bird` · `flower` · `fruit` · `vegetable` · `car` · `bicycle` · `shoe`

**Architecture:** Basic 4-block CNN baseline: Conv2d → ReLU → MaxPool, followed by a fully connected classifier. The model is trained with CrossEntropyLoss and Adam.

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
