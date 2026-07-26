import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Loading all trained models
models = {
    "Decision Tree": joblib.load('final_model1.joblib'),
    "Random Forest": joblib.load('final_model2.joblib'),
    "XGBoost": joblib.load('final_model3.joblib'),
    "K-Nearest Neighbors (KNN)": joblib.load('final_model4.joblib'),
    "MLP Neural Network": joblib.load('final_model5.joblib')
}

# Load training data to get the list of symptoms
train_data = pd.read_csv('Augmented_Data.csv')
X_train = train_data.drop(columns=['prognosis'])
symptoms = X_train.columns
# Streamlit app
st.title("Disease Prediction Based on Symptoms")

# Model selection dropdown
model_choice = st.selectbox("Select a model:", list(models.keys()))
selected_model = models[model_choice]

st.write("Select symptoms to predict the possible disease.")

# User input for symptoms
selected_symptoms = st.multiselect("Select your symptoms:", symptoms)

# Create an input vector for the model
input_vector = np.zeros(len(symptoms))
for symptom in selected_symptoms:
    symptom_index = list(symptoms).index(symptom)
    input_vector[symptom_index] = 1

# Prediction button
if st.button("Predict Disease"):
    prediction = selected_model.predict([input_vector])[0]
    st.write(f"The predicted disease using **{model_choice}** is: **{prediction}**")
