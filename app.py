import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Page config & Glassmorphism CSS ---
st.set_page_config(page_title="Crime Classification App", layout="wide")

st.markdown("""
    <style>
        body {
            background: linear-gradient(145deg, #e0f7fa, #ffffff);
            background-size: cover;
        }

        .main {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 2rem;
        }

        h1, h2, h3 {
            color: #003262;
            font-weight: 700;
        }

        .stButton>button {
            background-color: #003262;
            color: white;
            border-radius: 10px;
            padding: 0.5em 1.2em;
            border: none;
            transition: 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #005fa3;
        }

        .stSelectbox>div>div {
            font-size: 16px;
        }

        .stDataFrame th {
            background-color: #e3f2fd;
        }

        .css-1cpxqw2, .css-18e3th9 {
            background: transparent !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Crime Classification using Logistic Regression")
st.markdown("This app classifies districts as **High** or **Low Crime** using 5 key crime features.")

# --- Load Data ---
try:
    df = pd.read_csv("crime_data.csv")
except FileNotFoundError:
    st.error("❌ 'crime_data.csv' not found in app directory.")
    st.stop()

df.columns = df.columns.str.strip()
df.dropna(inplace=True)

# --- Target column ---
threshold = df['Total Cognizable IPC crimes'].median()
df['Target'] = (df['Total Cognizable IPC crimes'] > threshold).astype(int)

# --- Visualizations ---
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

st.subheader("🎯 Target Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='Target', data=df, ax=ax1)
ax1.set_title("High vs Low Crime Districts")
st.pyplot(fig1)

st.subheader("🔗 Feature Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# --- Model Setup ---
drop_cols = ['States/UTs', 'District', 'Year', 'Total Cognizable IPC crimes']
df_model = df.drop(columns=drop_cols)

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

# --- Evaluation ---
st.subheader("📈 Model Performance")
st.metric(label="Accuracy", value=f"{accuracy_score(y_test, y_pred):.2f}")
st.text("Classification Report")
st.code(classification_report(y_test, y_pred), language='text')

st.subheader("📉 Confusion Matrix")
fig3, ax3 = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
st.pyplot(fig3)

# --- Predict Crime Level by District ---
st.subheader("🧠 Predict Crime Level by District")

states = sorted(df['States/UTs'].unique())
selected_state = st.selectbox("Select State/UT", states)
filtered_df = df[df['States/UTs'] == selected_state]

districts = sorted(filtered_df['District'].unique())
selected_district = st.selectbox("Select District", districts)

district_row = filtered_df[filtered_df['District'] == selected_district].sort_values("Year", ascending=False).head(1)

if district_row.empty:
    st.warning("No data found for the selected district.")
else:
    st.markdown("### 🔎 District Crime Data Used for Prediction")
    display_cols = selected_features
    st.dataframe(district_row[display_cols], use_container_width=True)

    input_data = district_row[display_cols].values
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    result = "🔴 High Crime" if prediction == 1 else "🟢 Low Crime"
    st.success(f"📍 **{selected_district}**, {selected_state}: **{result}**")
    st.info(f"📊 Model Confidence (High Crime): **{proba:.2f}**")
