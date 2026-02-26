# Predictive-Maintenance-using-IoT-Sensor-Data-Explainable-Machine-Learning-
This project develops an Explainable Machine Learning system for Predictive Maintenance in Manufacturing using IoT sensor data. The goal is to predict machine failures in advance and help industries reduce unexpected downtime, maintenance costs, and safety risks.
The system uses machine learning models and explainable AI techniques (SHAP) to predict failures and provide insights into the factors contributing to machine breakdowns. An interactive Streamlit dashboard allows real-time predictions and visualization of model explanations.
🎯 Objectives
 
The main objectives of this project are:
•	Predict machine failures using IoT sensor data
•	Build accurate machine learning models
•	Handle imbalanced industrial datasets
•	Apply domain-based feature engineering
•	Provide explainable predictions using SHAP
•	Develop an interactive Streamlit dashboard
•	Support real-time maintenance decision-making
📊 Dataset
 
The dataset represents IoT sensor readings from a milling machine with 10,000 observations and 14 features.
 
Key Features
 
•	Air Temperature
•	Process Temperature
•	Rotational Speed
•	Torque
•	Tool Wear
•	Product Type
•	Engineered Features (Power, Temp Difference, Ratios)
Target Variable
•	Machine Failure (0 = No Failure, 1 = Failure)
Failure modes include:
•	Tool Wear Failure
•	Heat Dissipation Failure
•	Power Failure
•	Overstrain Failure
•	Random Failure
🔍 Methodology
 
1. Data Preprocessing
•	Data cleaning
•	One-hot encoding
•	Feature scaling
•	Train-test split
•	SMOTE for class imbalance
 
2. Exploratory Data Analysis
EDA was performed to analyze:
•	Feature distributions
•	Correlations
•	Class imbalance
•	Failure patterns
⚙️ Feature Engineering
 
Domain knowledge was used to create new features:
 
•	Power = Torque × Speed
•	Temperature Difference
•	Wear per Torque
•	Speed–Torque Ratio
These features improved prediction accuracy and interpretability.
🤖 Machine Learning Models
 
The following models were implemented:
 
1️⃣ Logistic Regression
•	Baseline model
•	Interpretable
•	Moderate performance
 
2️⃣ Random Forest
•	Nonlinear modeling
•	High accuracy
 
3️⃣ XGBoost (Best Model)
•	Highest accuracy
•	Handles imbalanced data well
•	Best overall performance
Final Accuracy: ~98.6%
📈 Model Evaluation
 
Models were evaluated using:
 
•	Accuracy
•	Precision
•	Recall
•	F1-score
•	Confusion Matrix
•	ROC-AUC
XGBoost achieved the best performance with minimal classification errors.
🔎 Explainable AI (SHAP)
 
SHAP was used to explain predictions:
 
Global Explanation
Most important features:
•	Rotational Speed
•	Tool Wear
•	Power
•	Torque
Local Explanation
SHAP waterfall plots show:
•	Why a machine will fail
•	Which features contribute most
This improves trust and interpretability for industrial users.
🖥️ Streamlit Dashboard
 
The project includes a Streamlit Web Application for real-time predictions.
Features
•	Input sensor values
•	Predict machine failure
•	Visualize SHAP explanations
•	Interactive interface
•	Real-time results
The dashboard demonstrates practical deployment of predictive maintenance systems.
🛠️ Technologies Used
•	Python
•	Pandas
•	NumPy
•	Scikit-learn
•	XGBoost
•	SHAP
•	Matplotlib
•	Seaborn
•	Streamlit
•	Imbalanced-learn (SMOTE)
 
 
📌 Results
 
•	Accuracy: 98.6%
•	High ROC-AUC
•	Low misclassification rate
•	Reliable failure prediction
•	Explainable predictions
The system demonstrates that machine learning combined with explainable AI can significantly improve predictive maintenance in manufacturing.
⚠️ Limitations
 
•	Uses simulated dataset
•	Single machine type
•	Binary classification only
•	No time-series modeling
•	Prototype dashboard
🔮 Future Work
 
•	Real industrial datasets
•	Remaining Useful Life (RUL) prediction
•	Time-series models (LSTM)
•	Real-time IoT integration
•	Cloud deployment
 
