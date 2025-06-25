import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page configuration
st.set_page_config(page_title="Crime Classification App", layout="wide")

st.title("🔍 Crime Classification using Logistic Regression")
st.markdown("This app classifies districts as **High** or **Low Crime** based on IPC offenses using Logistic Regression.")

# Read CSV directly
try:
    df = pd.read_csv("crime_data.csv")  # Make sure 'crime_data.csv' is in the same directory
except FileNotFoundError:
    st.error("❌ 'crime_data.csv' not found. Please make sure it's in the same folder.")
    st.stop()

# Show raw data
st.subheader("Raw Data")
st.dataframe(df.head())

# Preprocessing
df.columns = df.columns.str.strip()
df.dropna(inplace=True)

# Create target
threshold = df['Total Cognizable IPC crimes'].median()
df['Target'] = (df['Total Cognizable IPC crimes'] > threshold).astype(int)

# Visualization
st.subheader("Target Variable Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='Target', data=df, ax=ax1)
ax1.set_title("High vs Low Crime Districts")
st.pyplot(fig1)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False, ax=ax2)
st.pyplot(fig2)

# Prepare data
drop_cols = ['States/UTs', 'District', 'Year', 'Total Cognizable IPC crimes']
df_model = df.drop(columns=drop_cols)

X = df_model.drop("Target", axis=1)
y = df_model["Target"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model performance
st.subheader("Model Performance")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")
fig3, ax3 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
st.pyplot(fig3)

# ----------------------------
# 🔮 Prediction Section
# ----------------------------
st.subheader("🔮 Predict Crime Level for a New District")

input_data = []
for feature in X.columns:
    val = st.number_input(f"Enter value for **{feature}**", value=0.0)
    input_data.append(val)

if st.button("Predict Crime Level"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    result = "High Crime" if prediction == 1 else "Low Crime"
    st.success(f"🧠 Prediction: **{result}**")
