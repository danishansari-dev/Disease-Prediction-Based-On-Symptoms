# Disease Prediction Using Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://disease-prediction-based-on-symptoms-tb54b3snkpncduuwybu9w2.streamlit.app/)

## 📋 Project Description

This project provides an advanced machine learning diagnostic system that predicts potential diseases based on user-reported symptoms. Using 5 state-of-the-art machine learning algorithms, the system classifies health conditions into 41 distinct categories based on 132 symptoms.

The project features **symptom category filtering**, **differential diagnosis with percentage confidence scores**, **multi-model consensus comparison**, **Streamlit performance caching**, and **compressed model serialization**.

### Algorithms Supported
1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **XGBoost Classifier (XGBClassifier)**
4. **K-Nearest Neighbors (KNN)**
5. **Multi-Layer Perceptron (MLP Neural Network)**

---

## ✨ Key Features

- **Symptom Category Filtering**: Search and filter 132 symptoms across 6 medical categories (*Skin & Hair*, *Respiratory & ENT*, *Digestive & Abdominal*, *Neurological & Mood*, *Musculoskeletal*, *Urinary & Endocrine*).
- **Differential Diagnosis & Probabilities**: Ranks and displays Top-5 likely candidate diagnoses with percentage confidence progress bars.
- **Multi-Model Consensus View**: Side-by-side diagnostic comparison matrix across all 5 trained algorithms to evaluate model agreement.
- **Sub-Second Performance & Caching**: Models and symptom metadata are cached in memory (`@st.cache_resource` & `@st.cache_data`) for instant UI re-renders without reading large CSV datasets at runtime.
- **Compressed Model Weights**: Joblib model binaries are compressed (~98% reduction for KNN model from 35 MB to 0.68 MB).
- **Automated Unit Testing**: Includes a `pytest` test suite verifying symptom metadata integrity and prediction pipelines.

---

## 🛠️ Tech Stack

- **Frontend / Web Framework**: Streamlit
- **ML Libraries**: scikit-learn, XGBoost, NumPy, pandas
- **Model Serialization & Optimization**: joblib (compression level 3), JSON metadata
- **Testing**: pytest

---

## 📁 Project Structure

```
Disease-Prediction-Based-On-Symptoms/
├── data/
│   └── Augmented_Data.csv                # Dataset (132 symptoms, 41 diseases)
├── models/
│   ├── symptoms.json                     # Symptom feature keys, display titles & category mappings
│   ├── decision_tree.joblib              # Compressed Decision Tree model (0.02 MB)
│   ├── random_forest.joblib              # Compressed Random Forest model (0.39 MB)
│   ├── xgboost.joblib                    # Compressed XGBoost model (0.65 MB)
│   ├── knn.joblib                        # Compressed KNN model (0.68 MB)
│   └── mlp.joblib                        # Compressed MLP model (0.39 MB)
├── notebooks/
│   ├── 01_decision_tree.ipynb            # Decision Tree training notebook
│   ├── 02_random_forest.ipynb            # Random Forest training notebook
│   ├── 03_xgboost.ipynb                  # XGBoost training notebook
│   ├── 04_knn.ipynb                      # KNN training notebook
│   └── 05_mlp.ipynb                      # MLP training notebook
├── scripts/
│   └── export_and_compress.py            # Metadata extraction and joblib compression script
├── tests/
│   └── test_app.py                       # Automated pytest test suite
├── docs/                                 # Technical documentation & interview preparation guides
├── final_disease_prediction_app.py       # Main Streamlit web application
├── requirements.txt                      # Python dependencies (pinned for reproducibility)
├── .gitignore
└── README.md
```

---

## 🚀 Setup & Installation Instructions

### Prerequisites

Ensure you have Python 3.9+ and pip installed.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/danishansari-dev/Disease-Prediction-Based-On-Symptoms.git
   cd Disease-Prediction-Based-On-Symptoms
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Run Metadata Export & Model Compression Script**:
   ```bash
   python scripts/export_and_compress.py
   ```

---

## 🧪 Running Unit Tests

Run the automated test suite to verify metadata integrity and model predictions:

```bash
pytest tests/
```

---

## 🖥️ Running the Streamlit Application

Start the interactive Streamlit web app locally:

```bash
streamlit run final_disease_prediction_app.py
```

The application will launch automatically in your browser at `http://localhost:8501`.

---

## 🤖 Models Summary & Binary Sizes

| Model | File Path | Size |
|---|---|---|
| Decision Tree | `models/decision_tree.joblib` | 0.02 MB |
| Random Forest | `models/random_forest.joblib` | 0.39 MB |
| XGBoost | `models/xgboost.joblib` | 0.65 MB |
| K-Nearest Neighbors (KNN) | `models/knn.joblib` | 0.68 MB |
| MLP Neural Network | `models/mlp.joblib` | 0.39 MB |

---

## 📊 Dataset Specifications

- **132 binary symptom features** (0 = absent, 1 = present)
- **41 target disease categories** (`prognosis`)
- Categorized into 6 medical body systems for structured selection.

---

## 📄 License & Disclaimer

This project is for educational and research purposes only. It is not intended to replace professional medical diagnosis, advice, or treatment.
