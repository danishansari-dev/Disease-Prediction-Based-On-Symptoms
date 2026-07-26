# Disease Prediction Using Machine Learning

## Project Description

This project focuses on developing a machine learning model to predict diseases based on a set of symptoms provided by the user. Using various algorithms, the system classifies diseases into one of the 41 categories based on 132 different symptoms. The main goal is to provide a simple yet effective tool for healthcare diagnostics using state-of-the-art machine learning techniques. The models used in this project include **Decision Tree**, **Random Forest**, **XGBClassifier**, **K-Nearest Neighbors (KNN)**, and **Multi-Layer Perceptron (MLP)**.

The project utilizes a dataset, `Augmented_Data.csv`, which contains 132 symptom columns and a target column, "Prognosis", with 41 unique diseases. The final web application, built using **Streamlit**, allows users to input symptoms and receive a predicted diagnosis.

## Features

- **Symptom Input**: Users can input symptoms to predict possible diseases.
- **Disease Prediction**: Based on the input symptoms, the model predicts the most likely disease.
- **Algorithms Used**: Implements Decision Tree, Random Forest, XGBClassifier, KNN, and MLP for disease classification.
- **Model Evaluation**: Performance metrics like accuracy, precision, recall, and F1-score are used to evaluate model performance.

## Setup Instructions

### Prerequisites

Before setting up the project, ensure that you have the following installed:

- Python 3.x
- pip (Python package installer)

### Installation

1. **Clone the Repository**:  
   Clone the repository to your local machine using the following command:

   ```bash
   git clone https://github.com/danishansari-dev/Disease-Prediction-Based-On-Symptoms.git
   cd Disease-Prediction-Based-On-Symptoms
   ```

2. **Install Dependencies**:  
   Install all required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the Streamlit web app:

```bash
streamlit run final_disease_prediction_app.py
```

The app will open in your browser. Select a model, choose your symptoms, and click **Predict Disease**.

## Project Structure

```
Disease-Prediction-Based-On-Symptoms/
├── final_disease_prediction_app.py   # Streamlit web application
├── final_model1.ipynb                # Decision Tree notebook
├── final_model2.ipynb                # Random Forest notebook
├── final_model3.ipynb                # XGBoost notebook
├── final_model4.ipynb                # KNN notebook
├── final_model5.ipynb                # MLP Neural Network notebook
├── final_model1.joblib               # Trained Decision Tree model
├── final_model2.joblib               # Trained Random Forest model
├── final_model3.joblib               # Trained XGBoost model
├── final_model4.joblib               # Trained KNN model
├── final_model5.joblib               # Trained MLP model
├── Augmented_Data.csv                # Dataset (132 symptoms, 41 diseases)
├── requirements.txt                  # Python dependencies
└── README.md
```

## Models Used

| # | Model | File |
|---|-------|------|
| 1 | Decision Tree | `final_model1.joblib` |
| 2 | Random Forest | `final_model2.joblib` |
| 3 | XGBoost (XGBClassifier) | `final_model3.joblib` |
| 4 | K-Nearest Neighbors (KNN) | `final_model4.joblib` |
| 5 | Multi-Layer Perceptron (MLP) | `final_model5.joblib` |

## Dataset

The dataset (`Augmented_Data.csv`) contains:
- **132 symptom columns** (binary: 0 or 1)
- **1 target column** (`prognosis`) with **41 unique diseases**

## License

This project is for educational purposes.
