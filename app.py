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
st.markdown("Upload your dataset to classify districts as High or Low Crime based on IPC offenses.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Drop NA
    df.dropna(inplace=True)

    # Create target variable
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

    # Drop non-numeric identifiers
    drop_cols = ['States/UTs', 'District', 'Year', 'Total Cognizable IPC crimes']
    df_model = df.drop(columns=drop_cols)

    X = df_model.drop("Target", axis=1)
    y = df_model["Target"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Output
    st.subheader("Model Performance")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    st.pyplot(fig3)

else:
    st.warning("👈 Upload a CSV file to begin.")
