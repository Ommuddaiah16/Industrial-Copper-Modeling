import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Set the page layout to wide
st.set_page_config(layout="wide")

# Load pre-trained models and encoders/scalers
regressor = load('D:/Vscode/copper_project/RandomForestRegressor_model.joblib')
classifier = load('D:/Vscode/copper_project/ExtraTreesClassifier_model.joblib')
encoder_price = load('D:/Vscode/copper_project/encoder.joblib')
scaler_features_price = load('D:/Vscode/copper_project/scaler_features.joblib')
scaler_target_price = load('D:/Vscode/copper_project/scaler_target.joblib')
encoder_features_status = load('D:/Vscode/copper_project/encoder_features.joblib')
encoder_target_status = load('D:/Vscode/copper_project/encoder_target.joblib')
scaler_status = load('D:/Vscode/copper_project/scaler_status.joblib')

# Load dataset
df = load('D:/Vscode/copper_project/df.joblib')

# Load column names for price prediction
categorical_cols_price = load('D:/Vscode/copper_project/categorical_cols1.joblib')
numerical_cols_price = load('D:/Vscode/copper_project/numerical_cols1.joblib')

# Load column names for status prediction
categorical_cols_status = load('D:/Vscode/copper_project/categorical_cols2.joblib')
numerical_cols_status = load('D:/Vscode/copper_project/numerical_cols2.joblib')

# Extract unique values from the DataFrame for dropdowns
sample_countries = df['country'].unique().tolist()
sample_item_types = df['item type'].unique().tolist()
sample_applications = df['application'].unique().tolist()
sample_product_refs = df['product_ref'].unique().tolist()
sample_customers = df['customer'].unique().tolist()
sample_statuses = df['status'].unique().tolist()

# Main menu for navigation
menu = ["Home", "Price Prediction", "Status Prediction"]
choice = st.sidebar.selectbox("Select a page", menu)

# Input fields for Price Prediction
def get_price_prediction_input():
    country = st.selectbox("Country", sample_countries)
    item_type = st.selectbox("Item Type", sample_item_types)
    application = st.selectbox("Application", sample_applications)
    product_ref = st.selectbox("Product Reference", sample_product_refs)
    status = st.selectbox("Status", sample_statuses)
    customer = st.selectbox("Customer", sample_customers)
    width = st.number_input("Width", min_value=0.0)
    quantity_tons = st.number_input("Quantity Tons", min_value=0.0)
    thickness = st.number_input("Thickness", min_value=0.0)
    item_date_year = st.number_input("Item Date Year", min_value=2000, max_value=2024)
    item_date_month = st.number_input("Item Date Month", min_value=1, max_value=12)
    item_date_day = st.number_input("Item Date Day", min_value=1, max_value=31)
    delivery_date_year = st.number_input("Delivery Date Year", min_value=2000, max_value=2024)
    delivery_date_month = st.number_input("Delivery Date Month", min_value=1, max_value=12)
    delivery_date_day = st.number_input("Delivery Date Day", min_value=1, max_value=31)

    data = {
        "country": country,
        "item type": item_type,
        "application": application,
        "product_ref": product_ref,
        "status": status,
        "customer": customer,
        "width": width,
        "quantity tons_log": np.log(quantity_tons) if quantity_tons > 0 else 0,
        "thickness_log": np.log(thickness) if thickness > 0 else 0,
        "item_date_year": item_date_year,
        "item_date_month": item_date_month,
        "item_date_day": item_date_day,
        "delivery date_year": delivery_date_year,
        "delivery date_month": delivery_date_month,
        "delivery date_day": delivery_date_day,
    }
    return pd.DataFrame(data, index=[0])

# Input fields for Status Prediction
def get_status_prediction_input():
    country = st.selectbox("Country", sample_countries)
    item_type = st.selectbox("Item Type", sample_item_types)
    application = st.selectbox("Application", sample_applications)
    product_ref = st.selectbox("Product Reference", sample_product_refs)
    customer = st.selectbox("Customer", sample_customers)
    width = st.number_input("Width", min_value=0.0)
    quantity_tons = st.number_input("Quantity Tons", min_value=0.0)
    thickness = st.number_input("Thickness", min_value=0.0)
    item_date_year = st.number_input("Item Date Year", min_value=2000, max_value=2024)
    item_date_month = st.number_input("Item Date Month", min_value=1, max_value=12)
    item_date_day = st.number_input("Item Date Day", min_value=1, max_value=31)
    delivery_date_year = st.number_input("Delivery Date Year", min_value=2000, max_value=2024)
    delivery_date_month = st.number_input("Delivery Date Month", min_value=1, max_value=12)
    delivery_date_day = st.number_input("Delivery Date Day", min_value=1, max_value=31)
    selling_price = st.number_input("Selling Price", min_value=0.0)

    data = {
        "country": country,
        "item type": item_type,
        "application": application,
        "product_ref": product_ref,
        "customer": customer,
        "width": width,
        "quantity tons_log": np.log(quantity_tons) if quantity_tons > 0 else 0,
        "thickness_log": np.log(thickness) if thickness > 0 else 0,
        "item_date_year": item_date_year,
        "item_date_month": item_date_month,
        "item_date_day": item_date_day,
        "delivery date_year": delivery_date_year,
        "delivery date_month": delivery_date_month,
        "delivery date_day": delivery_date_day,
        "selling_price_log": np.log(selling_price) if selling_price > 0 else 0
    }
    return pd.DataFrame(data, index=[0])

# Home page
if choice == "Home":
    st.title("Industrial Copper Modeling")
    st.write("""
    Welcome to the Industrial Copper Modeling App.
    Use the sidebar to navigate to the Price Prediction and Status Prediction pages.
    """)

# Price Prediction page
elif choice == "Price Prediction":
    st.title("Price Prediction")
    user_input = get_price_prediction_input()

    # Ensure all categorical columns used in prediction are present in the encoder
    user_input_price = user_input[categorical_cols_price + numerical_cols_price]
    
    # Encoding and scaling user input
    encoded_input_price = encoder_price.transform(user_input_price[categorical_cols_price])
    scaled_input_price = scaler_features_price.transform(user_input_price[numerical_cols_price])
    final_input_price = np.concatenate([encoded_input_price, scaled_input_price], axis=1)

    if st.button("Predict Price"):
        prediction_price = regressor.predict(final_input_price)
        predicted_price = np.exp(scaler_target_price.inverse_transform(prediction_price.reshape(-1, 1)))[0][0]
        st.write(f"Predicted Selling Price: {predicted_price}")

# Status Prediction page
elif choice == "Status Prediction":
    st.title("Status Prediction")
    user_input = get_status_prediction_input()

    # Ensure all categorical columns used in prediction are present in the encoder
    user_input_status = user_input[categorical_cols_status + numerical_cols_status]

    # Encoding and scaling user input
    encoded_input_status = encoder_features_status.transform(user_input_status[categorical_cols_status])
    scaled_input_status = scaler_status.transform(user_input_status[numerical_cols_status])
    final_input_status = np.concatenate([encoded_input_status, scaled_input_status], axis=1)

    if st.button("Predict Status"):
        prediction_status = classifier.predict(final_input_status)
        status = "Won" if prediction_status[0] == 1 else "Lost"
        st.write(f"Predicted Status: {status}")
