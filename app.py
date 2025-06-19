import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geography.pkl', 'rb') as file:
    one_hot_encoder_geography = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title("Bank Customer Churn Prediction")

# User Inputs
credit_score = st.slider('Credit Score', 300, 850, 600)
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', 0.0, 1000000.0, 1000.0)
estimated_salary = st.number_input('Estimated Salary', 0.0, 1000000.0, 50000.0)
tenure = st.slider('Tenure (Years)', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
geography = st.selectbox('Geography', one_hot_encoder_geography.categories_[0])

# Convert Gender using LabelEncoder
gender_encoded = label_encoder_gender.transform([gender])[0]

# One-Hot Encode Geography
geo_df = pd.DataFrame({'Geography': [geography]})
geo_encoded = one_hot_encoder_geography.transform(geo_df)
geo_encoded_df = pd.DataFrame(
    geo_encoded.toarray(),
    columns=one_hot_encoder_geography.get_feature_names_out(['Geography'])
)

# Combine all features
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

final_input = pd.concat([input_df, geo_encoded_df], axis=1)

# Ensure same column order as training
expected_columns = scaler.feature_names_in_  # This ensures perfect order match
final_input = final_input[expected_columns]

# Scale input
scaled_input = scaler.transform(final_input)

# Make prediction
prediction = model.predict(scaled_input)[0][0]

# Show result
if prediction > 0.5:
    st.error(f"Customer is likely to churn. Probability: {prediction:.2f}")
else:
    st.success(f"Customer is not likely to churn. Probability: {1 - prediction:.2f}")
