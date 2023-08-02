import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model, scaler and label encoder
model = load('model_removed_smote.joblib')
scaler = load('scaler.joblib')
le = load('le.joblib')

# Load your dataset
df = pd.read_csv('Skyserver_SQL2_27_2018 6_51_39 PM.csv')

# Define the features in the same order as in the original dataset
features = ['u', 'g', 'r', 'i', 'z', 'specobjid', 'redshift', 'plate', 'mjd']

# Sidebar for user inputs
with st.sidebar:
    st.header("User Input Values")
    user_input = []
    for feature in features:
        value = st.number_input(f'Enter {feature}: ', step=0.01)
        user_input.append(value)
    user_input = np.array(user_input).reshape(1, -1)

    # Button for making prediction
    if st.button("Predict"):
        # Scale the input data
        user_input_scaled = scaler.transform(user_input)

        # Make prediction
        prediction = model.predict(user_input_scaled)
        prediction_label = le.inverse_transform(prediction)

        # Display the prediction on the sidebar
        st.header("Prediction")
        st.write(f'The predicted class is {prediction_label[0]}')

# Tabs
main_tab, model_tab, info_tab = st.tabs(["Main", "Model", "Info"])

# Content for the Main tab
with main_tab:
    st.write("Welcome to the celestial object classifier. Enter the required values in the sidebar and click on 'Predict' to classify the object.")

# Content for the Model tab
with model_tab:
    st.write("This model is trained on a dataset of celestial objects and can classify them into stars, galaxies, or quasars based on the input values.")
    st.write("This is the model used:")
    model_code = st.text_area("", "")

# Content for the Info tab
with info_tab:
    st.write("This project is a web application that uses machine learning to classify celestial objects. It's built using Streamlit and the model is trained on a dataset of celestial objects.")
    st.write("The below visuals and analysis is based on the dataset which the model is trained on")

    st.write("Dataset head:")
    st.dataframe(df.head())

    st.write("Dataset stats")
    st.dataframe(df.describe())

    st.write("Class distribution:")
    fig, ax = plt.subplots()
    sns.countplot(x='class', data=df, ax=ax)
    st.pyplot(fig)

    st.markdown("<h3> Angular measurement of celestial bodies.<h3>", unsafe_allow_html=True)

    # Create a 3D plot
    fig = px.scatter_3d(df, x='ra', y='dec', z='redshift', color='class')
    st.plotly_chart(fig)

    st.write("""
    1. Spatial Distribution: The three classes appear to be well distributed across the sky in terms of Right Ascension and Declination. There doesn't seems to a specific region of the sky where one class is more prevalent.

    2. Redshift distribution: It appears that stars tend to have a lower redshifts, meaning they are generally closer to us. Galaxies show a wider range of redshifts, indicating a mix of nearby and distant galaxies. Quasars on the other hand tend to have higher redshifts, suggesting that they are some of the most distant objects in the dataset.

    3. Class overlap: There are some overlaps in the redshift ranges of the different classes, particularly between galaxies and quasars. This suggests that redshift alone is not sufficient to distinguish between the different classes.
    """)
