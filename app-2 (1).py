import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
import sys
import re 

# -----------------------------------------------------------
# Configuration Constants 
# -----------------------------------------------------------
MODEL_PATH = 'model.pkl' 
SCALER_PATH = 'scaler.pkl'    
FAILURE_THRESHOLD = 0.50      

# 11 Raw input features (user input)
RAW_INPUT_FEATURES = [
    'Air_temperature_[K]', 'Process_temperature_[K]', 'Rotational_speed_[rpm]',
    'Torque_[Nm]', 'Tool_wear_[min]', 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
]

# 12 Features that the StandardScaler was fit on AND the XGBoost model expects.
MODEL_PREDICTION_FEATURES = [
    'Air_temperature_[K]', 'Process_temperature_[K]', 'Rotational_speed_[rpm]',
    'Torque_[Nm]', 'Tool_wear_[min]', 
    'Temp_Delta', 'Power_[W]', 'Wear_per_Torque', 'Speed_Torque_Ratio',
    'Type_H', 'Type_L', 'Type_M'
]

# -----------------------------------------------------------
# Load model, scaler, and data (PRE-CONFIG LOADING)
# -----------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

except FileNotFoundError as e:
    print(f"FATAL ERROR: Required file not found: {e.filename}")
    print("Please ensure your model, scaler, and data files are in the same directory.")
    sys.exit(1) 

# Define load_data function using standard Python
def load_raw_data():
    """Load and process data with a robust column cleaning mechanism."""
    try:
        df = pd.read_csv("ai4i2020.csv")
        df.rename(columns={"Machine failure": "Machine_failure"}, inplace=True)
        
        # --- ROBUST COLUMN CLEANING ---
        cleaned_columns = []
        for col in df.columns:
            col = col.strip()
            col = re.sub(r'\s+', '_', col)
            cleaned_columns.append(col)
        df.columns = cleaned_columns
        
        return df
    except FileNotFoundError:
        print("FATAL ERROR: 'ai4i2020.csv' not found. Cannot load dashboard visualizations.")
        sys.exit(1)

df = load_raw_data()


# -----------------------------------------------------------
# PAGE CONFIG AND HEADER 
# -----------------------------------------------------------
st.set_page_config(page_title="Predictive Maintenance", layout="wide") 
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("🔧 Predictive Maintenance Dashboard")
st.markdown("---")


# -----------------------------------------------------------
# Core Prediction and Feature Engineering Function
# -----------------------------------------------------------
@st.cache_data
def make_prediction_and_explain(input_df_raw):
    """
    Performs all necessary feature engineering, scaling, prediction, and SHAP calculation,
    ensuring the final input is exactly 12 features.
    """
    input_df = input_df_raw.copy()
    
    # 1. Feature Engineering 
    input_df['Temp_Delta'] = input_df['Process_temperature_[K]'] - input_df['Air_temperature_[K]']
    input_df['Power_[W]'] = input_df['Torque_[Nm]'] * (input_df['Rotational_speed_[rpm]'] * 2 * np.pi / 60)
    input_df['Wear_per_Torque'] = input_df['Tool_wear_[min]'] / (input_df['Torque_[Nm]'] + 1e-5)
    input_df['Speed_Torque_Ratio'] = input_df['Rotational_speed_[rpm]'] / (input_df['Torque_[Nm]'] + 1e-5)
    
    # 2. One-Hot Encoding for 'Type'
    df_dummies = pd.get_dummies(input_df['Type'], prefix='Type', dtype=int)
    input_df = pd.concat([input_df.drop('Type', axis=1), df_dummies], axis=1)

    # Ensure all three Type columns exist and fill missing with 0
    for col in ['Type_H', 'Type_L', 'Type_M']:
        if col not in input_df.columns:
            input_df[col] = 0

    # 3. Select the 12 features the model expects (dropping TWF, HDF, etc.)
    data_to_model = input_df[MODEL_PREDICTION_FEATURES]

    # 4. Scaling
    # Pass the NumPy array of the 12 features to the scaler
    scaled_data_np = scaler.transform(data_to_model.values)
    
    # Convert back to DataFrame for SHAP explanation purposes
    processed_data = pd.DataFrame(scaled_data_np, columns=MODEL_PREDICTION_FEATURES) 

    # 5. Prediction & Probability
    # Pass the 12 scaled features (NumPy array) to the model
    failure_proba = model.predict_proba(processed_data.values)[:, 1][0]
    
    # 6. SHAP Explainability
    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(processed_data.values)
    
    # --- CRITICAL INDEX FIX ---
    if isinstance(raw_shap_values, list):
        # Case: Standard binary output -> [class 0, class 1]. We want class 1 (index 1).
        if len(raw_shap_values) > 1:
            shap_values = raw_shap_values[1][0]
            expected_value = explainer.expected_value[1]
        else:
            # Fallback for list of size 1 (optimized for positive class)
            shap_values = raw_shap_values[0][0]
            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)): expected_value = expected_value[0]
            
    else:
        # Case: Optimized single NumPy array output (The one causing the IndexError)
        shap_values = raw_shap_values[0]
        
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[0]
    # --- END CRITICAL INDEX FIX ---
    
    return failure_proba, shap_values, expected_value, processed_data.iloc[0]


# -----------------------------------------------------------
# Sidebar Filters and Data Overview
# -----------------------------------------------------------
st.sidebar.header("Filters")

failure_filter = st.sidebar.selectbox("Machine Failure", ["All", 0, 1])
type_filter = st.sidebar.selectbox("Machine Type", ["All"] + list(df["Type"].unique()))

filtered_df = df.copy()
if failure_filter != "All":
    filtered_df = filtered_df[filtered_df["Machine_failure"] == failure_filter]

if type_filter != "All":
    filtered_df = filtered_df[filtered_df["Type"] == type_filter]

# Use tabs to organize the dashboard
tab1, tab2, tab3 = st.tabs(["Dataset Overview & Visuals", "Predictor & Alert System", "Prediction Explanation"])

with tab1:
    st.subheader("📁 Dataset Overview")
    st.dataframe(filtered_df, use_container_width=True)

    st.subheader("📊 Visualizations")
    colA, colB = st.columns(2)
    
    with colA:
        fig1 = px.histogram(
            filtered_df, x="Type", color="Machine_failure",
            title="Machine Failures by Type", barmode="group"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with colB:
        fig2 = px.scatter(
            filtered_df, x="Rotational_speed_[rpm]", y="Air_temperature_[K]",
            color="Machine_failure", title="Temperature vs Rotational Speed"
        )
        st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------------
# Prediction Input and Alert System
# -----------------------------------------------------------
with tab2:
    st.subheader("🤖 Predict Machine Failure")
    st.markdown("Enter the 11 raw operational parameters below to get a prediction.")
    
    cols = st.columns(3)
    inputs = {}
    
    # Input creation logic
    for i, feature in enumerate(RAW_INPUT_FEATURES):
        current_col = cols[i % 3] 

        with current_col:
            # Display name is cleaned for UI but the feature key MUST retain the brackets
            display_name = feature.replace('_', ' ').replace('[', '').replace(']', '')

            if feature == 'Type':
                 inputs[feature] = st.selectbox(display_name, df['Type'].unique())
            elif feature in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
                 # Even though these features are dropped later, we must collect them if the user provides them
                 inputs[feature] = st.checkbox(display_name, value=False)
            else:
                inputs[feature] = st.number_input(
                    display_name, 
                    min_value=float(df[feature].min()), 
                    max_value=float(df[feature].max()),
                    value=float(df[feature].median()),
                    step=0.1
                )
    
    # Convert inputs to a raw DataFrame
    input_data = {key: [value] if not isinstance(value, bool) else [int(value)] for key, value in inputs.items()}
    input_df_raw = pd.DataFrame(input_data)


    if st.button("Predict Failure and Generate Report"):
        
        prob, shap_values, expected_value, processed_data_for_display = make_prediction_and_explain(input_df_raw)

        # -----------------------------------------------------------
        # Alert System Implementation
        # -----------------------------------------------------------
        st.subheader("Prediction Result & Alert")
        failure_percentage = f"{prob * 100:.2f}%"

        if prob >= FAILURE_THRESHOLD:
            st.error(f"🚨 IMMEDIATE ALERT! HIGH FAILURE RISK! 🚨")
            st.warning(f"Predicted Failure Probability: **{failure_percentage}**")
            st.markdown(f"**Action Recommended:** The predicted probability exceeds the critical threshold of **{FAILURE_THRESHOLD * 100:.0f}%**.")
        else:
            st.success("✅ Machine Status: NORMAL")
            st.info(f"Predicted Failure Probability: **{failure_percentage}**")
            st.markdown(f"The predicted probability is below the critical threshold of {FAILURE_THRESHOLD * 100:.0f}%.")

        st.markdown("---")
        
        # Display the explanation plot
        st.subheader("🔍 Prediction Explanation (SHAP Waterfall Plot)")
        st.markdown(f"This plot shows the contribution of the {len(MODEL_PREDICTION_FEATURES)} features used by the final model to the current prediction.")

        fig = shap.plots.waterfall(
            shap.Explanation(
                values=shap_values,
                base_values=expected_value,
                data=processed_data_for_display.values,
                feature_names=MODEL_PREDICTION_FEATURES
            ),
            show=False,
            max_display=10 
        )
        
        # FIX: Check if the returned object is an Axes object, and get its parent Figure
        if hasattr(fig, 'figure'):
            fig = fig.figure
            
        st.pyplot(fig, use_container_width=True)

        st.toast('Prediction complete! The explanation is below.', icon='✅')


# -----------------------------------------------------------
# Prediction Explanation Tab 
# -----------------------------------------------------------
with tab3:
    st.subheader("Prediction Explanation (Summary)")
    st.info("The detailed SHAP explanation plot is generated in the **Predictor & Alert System** tab immediately after you click the 'Predict Failure and Generate Report' button.")
    st.markdown("The SHAP plot visually represents feature contribution: red bars increase the chance of failure, and blue bars decrease it.")