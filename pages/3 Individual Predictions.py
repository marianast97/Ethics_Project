import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.dashboard_components import ImportancesComponent, ShapContributionsTableComponent, ShapContributionsGraphComponent
from sklearn.preprocessing import LabelEncoder
import streamlit.components.v1 


st.set_option('deprecation.showPyplotGlobalUse', False)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./Maternal Health Risk Data Set.csv")
    target = 'RiskLevel'
    return df, target

# Train Logistic Regression model
@st.cache_resource
def load_model():
    model = load("./random_forest_model.pkl")
    return model


def main():
    st.markdown("## Individual Predictions")
    st.title("Maternal Health Risk Predictor")
    st.markdown("""
    This page helps to understand how a particular prediction was made, highlighting the contributions of the underlying features.
    """)

    df, target = load_data()

    # Encode target variable
    label_encoder = LabelEncoder()
    df[target] = label_encoder.fit_transform(df[target])

    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=444)

    model = load_model()
    model.fit(X_train, y_train)

    # Initialize Explainer
    explainer = ClassifierExplainer(model, X_test, y_test)

    st.subheader("Feature Importances (General)")

    # Feature Importances Component
    importances_component = ImportancesComponent(explainer, title="Feature Importances")
    importances_html = importances_component.to_html()
    st.components.v1.html(importances_html, height=600, scrolling=False)

    st.subheader("Contributions (for a specific point)")

    # Index selector for SHAP contributions
    index = st.selectbox("Select an index to view contributions", options=range(len(X_test)))

    # Contributions Table Component
    contributions_table_component = ShapContributionsTableComponent(explainer, title="Contributions Table", index=index)
    contributions_table_html = contributions_table_component.to_html()
    st.components.v1.html(contributions_table_html, height=600, scrolling=False)

    # Contributions Graph Component
    contributions_graph_component = ShapContributionsGraphComponent(explainer, title="Contributions Plot", index=index)
    contributions_graph_html = contributions_graph_component.to_html()
    st.components.v1.html(contributions_graph_html, height=600, scrolling=False)

if __name__ == "__main__":
    main()
