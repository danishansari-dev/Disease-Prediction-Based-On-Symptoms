# Why this file exists:
# This script serves as the primary user-facing Streamlit application for the Disease Prediction project.
# It loads pre-trained machine learning models and symptom metadata, accepts user symptom inputs,
# and performs differential disease diagnosis with confidence probability breakdowns and multi-model consensus.

import os
import json
import warnings

# Suppress non-fatal version mismatch warnings between scikit-learn / xgboost versions.
# These occur when deployment environments have minor patch differences.
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration for clean UI presentation
st.set_page_config(
    page_title="Disease Prediction Based on Symptoms",
    page_icon="🩺",
    layout="wide"
)

# Why @st.cache_resource is used here:
# Loading 5 machine learning models (Decision Tree, Random Forest, XGBoost, KNN, MLP) from disk
# can be expensive. Caching keeps loaded models in memory across user interactions to ensure sub-second response times.
@st.cache_resource
def load_models():
    """
    Loads and caches all pre-trained machine learning classifiers from the models directory.
    
    Returns:
        dict: A mapping of model display names to loaded joblib estimator objects.
    """
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    return {
        "Decision Tree": joblib.load(os.path.join(models_dir, 'decision_tree.joblib')),
        "Random Forest": joblib.load(os.path.join(models_dir, 'random_forest.joblib')),
        "XGBoost": joblib.load(os.path.join(models_dir, 'xgboost.joblib')),
        "K-Nearest Neighbors (KNN)": joblib.load(os.path.join(models_dir, 'knn.joblib')),
        "MLP Neural Network": joblib.load(os.path.join(models_dir, 'mlp.joblib'))
    }

# Why @st.cache_data is used here:
# Caching symptom metadata avoids reading large CSV files (11+ MB) repeatedly during UI interactions.
@st.cache_data
def load_symptom_metadata():
    """
    Loads symptom feature keys, formatted display labels, and category mappings from symptoms.json.
    Falls back to reading Augmented_Data.csv if symptoms.json is missing.
    
    Returns:
        tuple: (list of raw symptom feature keys, dict mapping raw key -> formatted display title, dict mapping raw key -> category)
    """
    json_path = os.path.join(os.path.dirname(__file__), 'models', 'symptoms.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            categories = data.get("categories", {s: "General / Systemic" for s in data["symptoms"]})
            return data["symptoms"], data["display_names"], categories
    
    # Fallback to CSV if JSON metadata is absent
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'Augmented_Data.csv')
    df = pd.read_csv(csv_path)
    symptoms = [c for c in df.columns if c != 'prognosis']
    display_names = {s: s.replace('_', ' ').strip().title() for s in symptoms}
    categories = {s: "General / Systemic" for s in symptoms}
    return symptoms, display_names, categories


# Helper function to compute top differential diagnoses with confidence scores
# Tricky logic note:
# Some estimators (like XGBoost) store targets as integer indices (0..40).
# We map class indices back to string disease names using target_classes.
def get_top_predictions(model, input_vector, fallback_classes, top_k=5):
    """
    Calculates probability distribution for a given input vector and returns top K candidate diagnoses.
    
    Tricky Logic:
        If model.classes_ contains integer indices (like in XGBClassifier), 
        we map those indices to the textual disease names from target_classes.
    """
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([input_vector])[0]
            
            # Retrieve model class labels or fall back
            raw_classes = getattr(model, "classes_", fallback_classes)
            
            # If classes are integer indices, map them to textual names in fallback_classes
            clean_classes = []
            for cls_item in raw_classes:
                if isinstance(cls_item, (int, np.integer)) and len(fallback_classes) > cls_item:
                    clean_classes.append(fallback_classes[cls_item])
                else:
                    clean_classes.append(str(cls_item))
            
            # Combine probabilities with disease labels and sort descending
            prob_list = list(zip(clean_classes, probs))
            prob_list.sort(key=lambda x: x[1], reverse=True)
            return prob_list[:top_k]
    except Exception:
        pass
    
    # Fallback if probability estimation fails
    pred = model.predict([input_vector])[0]
    if isinstance(pred, (int, np.integer)) and len(fallback_classes) > pred:
        pred = fallback_classes[pred]
    return [(str(pred), 1.0)]


# Main App Header
st.title("🩺 Medical Disease Prediction System")
st.markdown(
    "Predict potential health conditions based on reported symptoms using trained machine learning models "
    "(Decision Tree, Random Forest, XGBoost, KNN, and MLP)."
)

# Load resources
try:
    models = load_models()
    raw_symptoms, symptom_display_map, symptom_categories_map = load_symptom_metadata()
except Exception as e:
    st.error(f"Error loading model dependencies: {str(e)}")
    st.stop()

# Build inverse map (Display Name -> Raw Feature Key)
display_to_raw = {v: k for k, v in symptom_display_map.items()}

# Extract target classes from reference classifier
target_classes = getattr(models["Decision Tree"], "classes_", [])

# Sidebar for Symptom Selection & Category Filtering
st.sidebar.header("📋 Symptom Selector")

# Category Filter dropdown
available_categories = ["All Categories"] + sorted(list(set(symptom_categories_map.values())))
selected_category = st.sidebar.selectbox(
    "Filter symptoms by Category:",
    options=available_categories,
    help="Filter symptom options by body system or medical category."
)

# Filter display options based on category selection
if selected_category == "All Categories":
    filtered_raw_symptoms = raw_symptoms
else:
    filtered_raw_symptoms = [s for s in raw_symptoms if symptom_categories_map.get(s) == selected_category]

filtered_display_symptoms = sorted([symptom_display_map[s] for s in filtered_raw_symptoms])

# Symptom Multiselect with session state persistence across category switches
if 'selected_symptoms_set' not in st.session_state:
    st.session_state['selected_symptoms_set'] = set()

selected_in_widget = st.sidebar.multiselect(
    "Search or select symptoms:",
    options=filtered_display_symptoms,
    help="Type to search or select symptoms from the filtered list."
)

# Manage union of selected symptoms
# Update session set with widget selection
for item in selected_in_widget:
    st.session_state['selected_symptoms_set'].add(item)

# Remove unselected items only if they belong to current filter view
for item in filtered_display_symptoms:
    if item not in selected_in_widget and item in st.session_state['selected_symptoms_set']:
        st.session_state['selected_symptoms_set'].remove(item)

active_selected_symptoms = list(st.session_state['selected_symptoms_set'])

# Display active selection count
st.sidebar.info(f"Total Selected Symptoms: **{len(active_selected_symptoms)}**")
if active_selected_symptoms:
    with st.sidebar.expander("View Active Selection"):
        for sym in sorted(active_selected_symptoms):
            st.write(f"• {sym}")

# Construct feature vector matching model expected input
input_vector = np.zeros(len(raw_symptoms))
for disp_name in active_selected_symptoms:
    raw_key = display_to_raw[disp_name]
    symptom_index = raw_symptoms.index(raw_key)
    input_vector[symptom_index] = 1

# Tabbed Interface
tab1, tab2, tab3 = st.tabs([
    "🎯 Single Model Diagnosis",
    "📊 Multi-Model Consensus",
    "📚 Symptom Catalog by Category"
])

# TAB 1: Single Model Diagnosis
with tab1:
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("Model Configuration")
        model_choice = st.selectbox(
            "Select Classification Algorithm:",
            options=list(models.keys()),
            help="Choose which algorithm to perform the primary diagnosis."
        )
        selected_model = models[model_choice]
        
        predict_clicked = st.button("Run Diagnostic Prediction", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Diagnostic Results")
        
        # Validation guardrail: Ensure user has selected at least 1 symptom before predicting
        if len(active_selected_symptoms) == 0:
            st.warning("⚠️ Please select at least 1 symptom from the sidebar to generate a diagnostic prediction.")
        elif predict_clicked or st.session_state.get('has_predicted', False):
            st.session_state['has_predicted'] = True
            
            top_preds = get_top_predictions(selected_model, input_vector, target_classes, top_k=5)
            primary_disease, primary_prob = top_preds[0]
            
            # Display primary recommendation banner
            st.success(f"**Primary Diagnosis ({model_choice}):** {primary_disease}")
            st.metric(label="Primary Confidence Score", value=f"{primary_prob * 100:.1f}%")
            
            st.markdown("---")
            st.subheader("Top Differential Diagnoses")
            
            # Render Top 5 candidates with progress bars
            for rank, (disease, prob) in enumerate(top_preds, start=1):
                col_name, col_bar, col_pct = st.columns([3, 5, 2])
                with col_name:
                    st.write(f"**{rank}. {disease}**")
                with col_bar:
                    st.progress(float(prob))
                with col_pct:
                    st.write(f"{prob * 100:.1f}%")


# TAB 2: Multi-Model Consensus
with tab2:
    st.subheader("Multi-Model Diagnostic Comparison")
    st.markdown("Compares predicted diagnoses and top confidence scores across all 5 trained models.")
    
    if len(active_selected_symptoms) == 0:
        st.warning("⚠️ Please select symptoms from the sidebar to view multi-model consensus.")
    else:
        consensus_data = []
        for m_name, m_obj in models.items():
            preds = get_top_predictions(m_obj, input_vector, target_classes, top_k=3)
            top_dis, top_pr = preds[0]
            second_dis, second_pr = preds[1] if len(preds) > 1 else ("N/A", 0.0)
            
            consensus_data.append({
                "Algorithm": m_name,
                "Top Predicted Disease": top_dis,
                "Confidence": f"{top_pr * 100:.1f}%",
                "Secondary Diagnosis": second_dis,
                "Secondary Confidence": f"{second_pr * 100:.1f}%"
            })
        
        df_consensus = pd.DataFrame(consensus_data)
        st.dataframe(df_consensus, use_container_width=True, hide_index=True)
        
        # Consensus agreement check
        predictions_list = [d["Top Predicted Disease"] for d in consensus_data]
        most_common = max(set(predictions_list), key=predictions_list.count)
        agreement_count = predictions_list.count(most_common)
        
        if agreement_count >= 3:
            st.info(f"💡 **Consensus Recommendation:** {agreement_count} out of 5 models agree on **{most_common}**.")
        else:
            st.info("💡 **Divergent Signals:** Models show varied predictions. Consider reviewing secondary diagnoses or adding symptoms.")


# TAB 3: Symptom Catalog by Category
with tab3:
    st.subheader("Supported Symptoms by Category (132 Total)")
    st.markdown("Browse all 132 supported symptoms organized by medical category:")
    
    cat_names = sorted(list(set(symptom_categories_map.values())))
    for cat in cat_names:
        cat_symptoms = [symptom_display_map[s] for s in raw_symptoms if symptom_categories_map.get(s) == cat]
        with st.expander(f"📌 **{cat}** ({len(cat_symptoms)} symptoms)", expanded=True):
            cols = st.columns(3)
            chunk_size = int(np.ceil(len(cat_symptoms) / 3))
            for i, col in enumerate(cols):
                with col:
                    chunk = cat_symptoms[i * chunk_size : (i + 1) * chunk_size]
                    for item in chunk:
                        st.markdown(f"- {item}")

# TODO: Add clinical notes ingestion and SHAP model explainability features in future iteration.

# Medical Disclaimer Footer
st.markdown("---")
st.caption(
    "⚠️ **Medical Disclaimer**: This application is built for educational and research purposes only. "
    "It is not intended to replace professional medical advice, diagnosis, or treatment. "
    "Always seek the advice of a qualified physician or healthcare provider with any medical questions."
)
