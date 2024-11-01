# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
penguins = pd.read_csv("/kaggle/input/palmer-archipelago-antarctica-penguin-data/penguins_lter.csv")

# Prepare a smaller version of the dataset for the Random Forest Model
df = pd.DataFrame({
    'Species': np.random.choice(['Species_A', 'Species_B'], 100),
    'Region': np.random.choice(['Region_1', 'Region_2'], 100),
    'Island': np.random.choice(['Island_1', 'Island_2'], 100),
    'Stage': np.random.choice(['Stage_1', 'Stage_2'], 100),
    'Culmen Length (mm)': np.random.rand(100) * 20,
    'Culmen Depth (mm)': np.random.rand(100) * 10,
    'Flipper Length (mm)': np.random.rand(100) * 30,
    'Body Mass (g)': np.random.rand(100) * 1000,
    'Sex': np.random.choice([0, 1], 100)  # Assuming 0 = Female, 1 = Male
})

# Splitting data
X = df[['Species', 'Region', 'Island', 'Stage', 'Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']]
y = df['Sex']
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit App Layout
st.title("Species Study Dashboard")
st.write(f"Model Accuracy: {accuracy:.2f}")

# Feature Importance Display
st.header("Feature Importance")
feature_importance = rf_model.feature_importances_
st.bar_chart(pd.DataFrame(feature_importance, index=X.columns, columns=["Importance"]))

# Dataset Overview
st.header("Dataset Overview")
st.write(df.head())

# User Inputs for Prediction
st.header("Make Predictions")

culmen_length = st.number_input("Enter Culmen Length (mm)")
culmen_depth = st.number_input("Enter Culmen Depth (mm)")
flipper_length = st.number_input("Enter Flipper Length (mm)")
body_mass = st.number_input("Enter Body Mass (g)")
species = st.selectbox("Enter Species", ['Species_A', 'Species_B'])
region = st.selectbox("Enter Region", ['Region_1', 'Region_2'])
island = st.selectbox("Enter Island", ['Island_1', 'Island_2'])
stage = st.selectbox("Enter Stage", ['Stage_1', 'Stage_2'])

# Button to make prediction
if st.button("Predict"):
    # Convert inputs to DataFrame
    input_data = pd.DataFrame({
        'Culmen Length (mm)': [culmen_length],
        'Culmen Depth (mm)': [culmen_depth],
        'Flipper Length (mm)': [flipper_length],
        'Body Mass (g)': [body_mass],
        'Species': [species],
        'Region': [region],
        'Island': [island],
        'Stage': [stage]
    })
    
    # Process the input data
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    
    # Make prediction
    prediction = rf_model.predict(input_data)
    result = "Male" if prediction[0] == 1 else "Female"
    st.write(f"Prediction: {result}")
