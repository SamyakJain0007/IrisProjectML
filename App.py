import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('MIPML.pkl', 'rb') as file:
    model = pickle.load(file)

# Title and description
st.title("Iris Species Prediction")
st.write("This app predicts the species of an iris flower based on its measurements.")

# Input fields for features
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Prediction
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    st.write(f"The predicted species is: {species[prediction[0]]}")
