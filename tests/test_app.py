# Why this test file exists:
# To provide automated quality assurance for symptom metadata, model binary integrity,
# and prediction feature vector processing.

import os
import json
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SYMPTOMS_JSON = os.path.join(MODELS_DIR, 'symptoms.json')

def test_symptoms_json_integrity():
    """
    Verifies that symptoms.json exists and contains the expected 132 feature symptoms
    and non-empty display name mappings.
    """
    assert os.path.exists(SYMPTOMS_JSON), "symptoms.json metadata file must exist in models/"
    
    with open(SYMPTOMS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    assert "symptoms" in data, "symptoms key missing from symptoms.json"
    assert "display_names" in data, "display_names key missing from symptoms.json"
    assert len(data["symptoms"]) == 132, f"Expected 132 symptoms, found {len(data['symptoms'])}"
    assert len(data["display_names"]) == 132, "Display names map count mismatch"

def test_model_loading_and_prediction():
    """
    Ensures trained model files (.joblib) exist and produce valid non-empty disease predictions
    when given a test feature vector.
    """
    model_files = [
        'decision_tree.joblib',
        'random_forest.joblib',
        'xgboost.joblib',
        'knn.joblib',
        'mlp.joblib'
    ]
    
    # Load symptoms list
    with open(SYMPTOMS_JSON, 'r', encoding='utf-8') as f:
        symptoms = json.load(f)["symptoms"]
        
    # Get reference class names from Decision Tree model
    dt_model = joblib.load(os.path.join(MODELS_DIR, 'decision_tree.joblib'))
    target_classes = getattr(dt_model, 'classes_', [])

    # Create sample vector with itching and skin_rash enabled
    input_vector = np.zeros(len(symptoms))
    if 'itching' in symptoms:
        input_vector[symptoms.index('itching')] = 1
    if 'skin_rash' in symptoms:
        input_vector[symptoms.index('skin_rash')] = 1

    for fname in model_files:
        mpath = os.path.join(MODELS_DIR, fname)
        assert os.path.exists(mpath), f"Model file missing: {fname}"
        
        model = joblib.load(mpath)
        pred = model.predict([input_vector])[0]
        
        # If integer label (e.g. XGBoost), map to reference class string
        if isinstance(pred, (int, np.integer)) and len(target_classes) > pred:
            pred = target_classes[pred]
            
        assert isinstance(pred, (str, np.str_)), f"Prediction for {fname} should be a string, got {type(pred)}"
        assert len(str(pred).strip()) > 0, f"Prediction for {fname} should not be empty"
