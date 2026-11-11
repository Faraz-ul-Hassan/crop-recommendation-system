# 🌱 Crop Recommendation System

An end-to-end machine learning system that recommends the most suitable crop based on soil nutrients and environmental conditions. Built with **Random Forest**, deployed with **FastAPI**, and served through a **plain-JS web UI**.

**Live API docs (Swagger):** `http://localhost:8000/docs` after running locally.

---

## 📊 Overview

Given 7 input features:

- **N** — Nitrogen (mg/kg)
- **P** — Phosphorus (mg/kg)
- **K** — Potassium (mg/kg)
- **temperature** — °C
- **humidity** — %
- **ph** — soil pH
- **rainfall** — mm

…the model predicts one of **22 crops** (rice, maize, chickpea, … coffee).

### Results

| Model | Test accuracy | 5-fold CV (mean ± std) |
|---|---|---|
| Logistic Regression | 97.27% | 96.76% ± 1.13% |
| KNN | 97.95% | 96.53% ± 0.75% |
| Decision Tree | 97.95% | 98.47% ± 0.71% |
| **Random Forest (chosen)** | **99.55%** | **99.32% ± 0.43%** |
| SVM | 98.41% | 97.61% ± 0.75% |
| Naive Bayes | 99.55% | 99.43% ± 0.51% |
| Gradient Boosting | 98.86% | 98.75% ± 0.69% |

**Why Random Forest?** Equal test accuracy to Naive Bayes (99.55%), but **lower CV variance** (0.43% vs 0.51%), no naive-independence assumption (our `P`–`K` correlation is 0.74), and better behavior under feature drift. Full analysis in `notebooks/crop_recommendation.ipynb`.

---

## 🗂️ Project Structure

```
crop-recommendation-system/
├── app/
│   ├── main.py              FastAPI app + /predict endpoint
│   ├── schemas.py           Pydantic request/response models
│   └── static/
│       ├── index.html       Frontend form
│       ├── style.css
│       └── script.js        Fetch + render results
├── models/
│   ├── crop_model.pkl       Trained Random Forest
│   ├── scaler.pkl           StandardScaler (fit on train only)
│   └── metadata.json        Feature order, classes, metrics
├── notebooks/
│   └── crop_recommendation.ipynb   Full ML pipeline (EDA → training → eval)
├── figures/                 EDA and evaluation plots
├── data/
│   └── Crop_recommendation.csv
├── requirements.txt         Runtime dependencies
├── requirements-full.txt    Full notebook environment
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Run Locally

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/crop-recommendation-system.git
cd crop-recommendation-system
```

### 2. Create a virtual environment and install

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start the API

```bash
uvicorn app.main:app --reload --port 8000
```

Startup output should include:

```
[startup] Model loaded from .../models/crop_model.pkl
[startup] Scaler loaded from .../models/scaler.pkl
```

### 4. Open the app

- **Web UI:** http://localhost:8000/
- **Swagger docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

---

## 🔌 API Usage

### `POST /predict`

**Request:**

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```

**Response:**

```json
{
  "predicted_crop": "rice",
  "confidence": 0.9,
  "top_3": [
    {"crop": "rice", "probability": 0.9},
    {"crop": "jute", "probability": 0.1},
    {"crop": "pomegranate", "probability": 0.0}
  ]
}
```

**Invalid input** (e.g., `humidity > 100`) returns **HTTP 422** with details.

### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

---

## 🧠 How It Works

1. **EDA** — 2200 samples, 22 balanced crops, no nulls/duplicates. Some right-skew in `N`, `K`, `rainfall`; `P`–`K` correlated at 0.74. Outliers kept (real agronomy).
2. **Preprocessing** — `StandardScaler` **fit only on the training split** to prevent data leakage. Stratified 80/20 split preserves class proportions.
3. **Model selection** — 7 classifiers compared on **accuracy, per-class precision/recall/F1, and 5-fold CV**. Random Forest chosen for accuracy + stability.
4. **Hyperparameter tuning** — 27-combination grid search (`n_estimators`, `max_depth`, `min_samples_leaf`). No meaningful improvement — the dataset's remaining errors are inherent to the features.
5. **Ensemble experiment** — soft-voting RF+NB matched 99.55% but adopted NB's systematic `rice→jute` error due to NB overconfidence. Kept RF.
6. **Deployment** — Model + scaler serialized with `joblib`, loaded **once at FastAPI startup**, applied in the same order as training.

---

## 🛠️ Tech Stack

- **ML:** scikit-learn, pandas, numpy, joblib
- **API:** FastAPI, Uvicorn, Pydantic v2
- **Frontend:** plain HTML/CSS/JS (no framework, no build step)

---

## 🎯 What's Next

- Containerize with Docker
- Add CI (GitHub Actions) — run tests on every push
- Deploy to a public host (Render, Fly.io, Railway)
- Swap the frontend for React (API contract unchanged)
- Add more features to separate rice vs. jute (the model's only systematic confusion)

---

## 📄 License

MIT (or your choice — add `LICENSE` file if you want a formal one)
