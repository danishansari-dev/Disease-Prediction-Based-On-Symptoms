import os
import json
import pandas as pd
import joblib

# Rule-based category mapper for 132 medical symptoms
CATEGORY_KEYWORDS = {
    "Skin & Hair": [
        "skin", "rash", "eruptions", "patches", "blisters", "blackheads", "scurring",
        "peeling", "silver", "nails", "dents", "blister", "red_sore", "yellow_crust", "itching"
    ],
    "Respiratory & ENT": [
        "sneezing", "shivering", "chills", "cough", "breathlessness", "phlegm", "throat",
        "eyes", "sinus", "runny_nose", "congestion", "chest_pain", "smell", "sputum"
    ],
    "Digestive & Abdominal": [
        "stomach", "acidity", "ulcers", "tongue", "vomiting", "indigestion", "nausea",
        "appetite", "abdominal", "diarrhoea", "constipation", "yellowish_skin", "yellowing_of_eyes",
        "liver", "bleeding", "distention", "fluid_overload"
    ],
    "Neurological & Mood": [
        "headache", "dizziness", "sensorium", "concentration", "visual", "unsteadiness",
        "balance", "spinning", "speech", "depression", "irritability", "anxiety"
    ],
    "Musculoskeletal": [
        "joint", "muscle", "back_pain", "neck", "knee", "hip", "weakness", "stiff", "walking"
    ],
    "Urinary & Endocrine": [
        "micturition", "urination", "urine", "gases", "bladder", "polyuria", "menstruation",
        "thyroid", "sugar", "weight_gain", "weight_loss", "appetite"
    ]
}

def get_symptom_category(symptom_key):
    """Assigns a symptom key to a category based on keyword matching, defaulting to General."""
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in symptom_key.lower() for kw in keywords):
            return category
    return "General / Systemic"

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'Augmented_Data.csv')
    models_dir = os.path.join(base_dir, 'models')
    json_path = os.path.join(models_dir, 'symptoms.json')

    # 1. Extract symptoms list and build categories
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        symptoms = [col for col in df.columns if col != 'prognosis']
        
        display_names = {s: s.replace('_', ' ').strip().title() for s in symptoms}
        categories = {s: get_symptom_category(s) for s in symptoms}
        
        symptom_metadata = {
            "symptoms": symptoms,
            "display_names": display_names,
            "categories": categories
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(symptom_metadata, f, indent=2)
        print(f"Successfully exported {len(symptoms)} symptoms with categories to {json_path}")

    # 2. Compress model files in models/
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        model = joblib.load(model_path)
        joblib.dump(model, model_path, compress=3)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Compressed {model_file}: {size_mb:.2f} MB")

if __name__ == '__main__':
    main()
