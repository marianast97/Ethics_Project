import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.dashboard_components import FeatureInputComponent, ClassifierPredictionSummaryComponent, ClassifierRandomIndexComponent
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
    st.markdown("## Prediction Simulator")
    st.title("Maternal Health Risk Predictor")
    st.markdown("""
This page resembles 'What if... ?' It shows how the model prediction would change based on a change in the attribute values
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
    pos_label = 1  # Assuming 'high risk' or any other valid label is mapped to 1
    explainer = ClassifierExplainer(model, X_test, y_test, pos_label=pos_label)

    st.subheader("Simulator")

    # Select a sample index
    index = st.selectbox("Select an index to view and modify", options=range(len(X_test)))

    # Display the original feature values for the selected index
    sample = X_test.iloc[index].copy()
    st.write("Original Feature Values:")
    st.write(sample)

    # Create sliders for each feature to modify the values
    new_values = {}
    for col in sample.index:
        new_values[col] = st.slider(col, float(X[col].min()), float(X[col].max()), float(sample[col]))

    # Create a DataFrame with the new values
    new_sample = pd.DataFrame([new_values])

    # Display the new feature values
    st.write("Modified Feature Values:")
    st.write(new_sample)

    # Predict the risk level using the modified feature values
    new_prediction = model.predict(new_sample)
    new_prediction_proba = model.predict_proba(new_sample)

    # Display the prediction and probabilities
    st.write(f"Predicted Risk Level: {label_encoder.inverse_transform(new_prediction)[0]}")
    st.write("Prediction Probabilities:")
    st.write(new_prediction_proba)

    # SHAP explanation
    shap_values = explainer.get_shap_values_df(new_sample)
    st.write("SHAP Values:")
    st.write(shap_values)

    st.subheader("Feature Contributions to Prediction")
    shap.waterfall_plot(shap.Explanation(values=shap_values.values[0], base_values=explainer.expected_value(pos_label), data=new_sample.values, feature_names=new_sample.columns), max_display=10)
    st.pyplot()

  

if __name__ == "__main__":
    main()

