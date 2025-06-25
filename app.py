import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page config
st.set_page_config(page_title="Crime Classification App", layout="wide")
st.title("🔍 Crime Classification using Logistic Regression")
st.markdown("This app classifies districts as **High** or **Low Crime** using 5 key crime features.")

# --- Load Data ---
try:
    df = pd.read_csv("crime_data.csv")  # Make sure this file is in the same folder
except FileNotFoundError:
    st.error("❌ 'crime_data.csv' not found in app directory.")
    st.stop()

# Strip whitespace and drop missing
df.columns = df.columns.str.strip()
df.dropna(inplace=True)

# Target: High(1) or Low(0) based on median
threshold = df['Total Cognizable IPC crimes'].median()
df['Target'] = (df['Total Cognizable IPC crimes'] > threshold).astype(int)

# --- Visualizations ---
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Target Variable Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='Target', data=df, ax=ax1)
ax1.set_title("High vs Low Crime Districts")
st.pyplot(fig1)

st.subheader("Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# --- Model Setup ---

# Drop non-feature columns
drop_cols = ['States/UTs', 'District', 'Year', 'Total Cognizable IPC crimes']
df_model = df.drop(columns=drop_cols)

# Select correct column names that exist
selected_features = ['Murder', 'Attempt to commit Murder', 'Rape', 'Kidnapping & Abduction_Total', 'Robbery']
missing_features = [col for col in selected_features if col not in df_model.columns]
if missing_features:
    st.error(f"❌ Missing expected columns: {', '.join(missing_features)}")
    st.stop()

X = df_model[selected_features]
y = df_model['Target']

# --- Model Training ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Model Evaluation ---
st.subheader("Model Performance")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
fig3, ax3 = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
st.pyplot(fig3)

# --- Prediction Section ---
st.subheader("🔮 Predict Crime Level for a New District")

input_data = []
st.markdown("Enter values for the following crime types:")
for feature in selected_features:
    val = st.number_input(f"{feature}", min_value=0.0, value=0.0, step=1.0)
    input_data.append(val)

if st.button("Predict Crime Level"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    result = "High Crime" if prediction == 1 else "Low Crime"
    st.success(f"🧠 Prediction: **{result}**")
