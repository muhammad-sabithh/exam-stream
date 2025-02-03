import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the trained model
filename = 'model.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

st.title('Survival Prediction App')
st.subheader('Upload your passenger dataset to predict survival.')

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Identifying numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Handle missing values
    df.fillna(0, inplace=True)

    # Standardize numerical data
    scaler = StandardScaler()
    df_numerical = pd.DataFrame(scaler.fit_transform(df[numerical_columns]), columns=numerical_columns)

    # Encode categorical data
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    df_categorical = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))

    # Combine numerical and categorical features
    df_preprocessed = pd.concat([df_numerical, df_categorical], axis=1)

    # Make predictions
    prediction = loaded_model.predict(df_preprocessed)
    prediction_text = np.where(prediction == 1, 'Survived', 'Not Survived')

    st.subheader('Survival Prediction Results:')
    df['Survival Prediction'] = prediction_text
    st.write(df[['Survival Prediction']])

    # Download button for results
    df.to_csv("predictions.csv", index=False)
    st.download_button(label="Download Predictions", data=open("predictions.csv", "rb"), file_name="predictions.csv")



    
    