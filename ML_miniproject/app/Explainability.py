import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px



def show_explainability():
    st.title("📊 Feature Importance (Random Forest)")

    # Feature Importance
    try:
        df = pd.read_csv('../model/feature_importance.csv')  # or wherever it's stored
        fig = px.bar(
            df.sort_values(by='Importance', ascending=True),
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis',
            title='🌾 Feature Impact on Crop Prediction',
            height=500
        )
        fig.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            title_font_color='white',
            coloraxis_colorbar=dict(title='Importance', tickfont=dict(color='white'), titlefont=dict(color='white'))
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ Unable to load feature importance: {e}")

    # Model Evaluation Section
    st.title("📈 Model Evaluation Report")
    try:
        # Load model and dataset
        model = joblib.load('../model/crop_model.pkl')
        data = pd.read_csv('../data/Crop_recommendation.csv')
        X = data.drop('label', axis=1)
        y = data['label']

        y_pred = model.predict(X)

        # 1. Classification Report
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.markdown("#### 📋 Classification Report")
        st.dataframe(report_df.style.background_gradient(cmap='viridis'))

        # 2. Confusion Matrix
        st.markdown("#### 🔁 Confusion Matrix")
        cm = confusion_matrix(y, y_pred, labels=np.unique(y))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap="YlGnBu", xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(plt)

    except Exception as e:
        st.warning(f"⚠️ Model evaluation failed: {e}")

    st.markdown("## 🥧 Crop Distribution in Dataset")

    # Load dataset
    df = pd.read_csv('../data/Crop_recommendation.csv')
    label_counts = df['label'].value_counts().reset_index()
    label_counts.columns = ['Crop', 'Count']

    # Plot Pie Chart
    pie_fig = px.pie(
        label_counts,
        names='Crop',
        values='Count',
        title='Crop Distribution',
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    st.plotly_chart(pie_fig, use_container_width=True)


    from sklearn.metrics import accuracy_score

    # Extract features and target
    X = df.drop('label', axis=1)
    y = df['label']

    # Predict and calculate accuracy
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    st.markdown("### ✅ Overall Model Accuracy")
    st.success(f"The Random Forest model achieved an accuracy of **{accuracy * 100:.2f}%** on the entire dataset.")

