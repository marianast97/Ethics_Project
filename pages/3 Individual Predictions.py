import streamlit as st
import pandas as pd
import numpy as np
<<<<<<< HEAD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from joblib import load
import plotly.graph_objects as go
import plotly.io as pio
import time
=======
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
>>>>>>> c744faa82120502919f60aea248c92c91972fdee
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.dashboard_components import ImportancesComponent, ShapContributionsTableComponent, ShapContributionsGraphComponent
from sklearn.preprocessing import LabelEncoder
import streamlit.components.v1 
<<<<<<< HEAD
=======

>>>>>>> c744faa82120502919f60aea248c92c91972fdee

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
<<<<<<< HEAD
    
    
def main():
    df_og, target = load_data()
    label_encoder = LabelEncoder()
    df = df_og.copy()
    df[target] = label_encoder.fit_transform(df[target])
    X = df.drop(target, axis=1)
    y = df[target]
    
    if 'X_train' not in st.session_state:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
    else:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)
    
    st.write("## Indiviual Predictions")
    st.title("Maternal Health Risk Prediction")
    st.logo(
        "./love.png",
        icon_image="./heartbeat.gif",
    )
    st.write("\n\n")
    st.write("### Feature Importances")
    st.write("Using [ExplainerDashboard](https://github.com/oegedijk/explainerdashboard) for our model, we can see the feature importances.")
    st.write(f"**Model**: Random Forest (Trained on *{len(X_train)}* samples and validated on *{len(X_test)}* samples.)")
    st.write('**Note**: We do not use the model co-efficeints as feature importances because the value of each co-efficient depends on the scale of the input features. For example, if we use months as a unit for Age instead of years, the coefficient for Age will be 12 times smaller which does not make sense.')
    st.write("This means that the magnitude of a coefficient is not necessarily a good measure of a feature’s importance.")
    st.write("Hence, SHAP values are used to calculate feature importances.")
    st.write("Shapley values are a concept from game theory that provide a natural way to compute which features contribute to a prediction or contribute to the uncertainty of a prediction.")
    st.write("A prediction can be explained by assuming that each feature value of the instance is a 'player' in a game where the prediction is the payout.")
    st.info("The SHAP value of a feature is **not** the difference of the predicted value after removing the feature from the model training. It can be interpreted as - given the current set of feature values, the contribution of a feature value to the difference between the actual prediction and the mean prediction is the estimated Shapley value.", icon="ℹ️")
    
    model = load_model()
    model = model.fit(X_train, y_train)
    # importances = model.feature_importances_
    # print(importances)  
    
    with st.container():
        st.write("\n\n")
        with st.spinner(text='Loading Explainer...'):
            if 'explainer' not in st.session_state:
                explainer = ClassifierExplainer(model, X_test, y_test)
                st.session_state.explainer = explainer
                explainer.dump("./explainer.joblib")
            else:
                explainer = ClassifierExplainer.from_file("./explainer.joblib")
            
            importances_component = ImportancesComponent(explainer, hide_title=True)
            importances_html = importances_component.to_html()
            st.components.v1.html(importances_html, height=440, width=800, scrolling=False)
       
    st.toast('Explainer loaded', icon="✔️")
    st.write("From the plot above, we can see that the most prominent feature for the model in its decision making is *BS* i.e blood sugar levels")
    st.write("This gives an overview of the model's decision making process. However, if we want to see the contributions for a single sample, we proceed further.") 
     
    st.write("\n\n")
    st.write("### Contributions for a single point")
    st.write("To see the contributions for a single point, select a sample from the sidebar.")
    
    # Index selector for SHAP contributions
    index = st.sidebar.selectbox("Select an index to view contributions", options=range(len(X_test)))
    st.write(f"Selected index: {index}")
    st.write(f"Predicted class: {model.predict(X_test.iloc[[index]])[0]} ({label_encoder.classes_[model.predict(X_test.iloc[[index]])[0]]})")
    
    sample_df = df_og.loc[[X_test.index[index]]]
    sample_df = sample_df.to_frame().T if isinstance(sample_df, pd.Series) else sample_df
=======


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
>>>>>>> c744faa82120502919f60aea248c92c91972fdee

    col1, col2 = st.columns(2)
    
    with col1:
        # Contributions Table Component
        st.write("\n\n")
        st.dataframe(sample_df, hide_index=True)
        st.write("\n\n")
        contributions_table_component = ShapContributionsTableComponent(explainer, title="Contributions Table", index=index)
        contributions_table_html = contributions_table_component.to_html()
        st.components.v1.html(contributions_table_html, height=600, scrolling=False)
        
    with col2:
        # Contributions Graph Component
        st.write("\n\n")
        contributions_graph_component = ShapContributionsGraphComponent(explainer, title="Contributions Plot", index=index)
        contributions_graph_html = contributions_graph_component.to_html()
        st.components.v1.html(contributions_graph_html, height=730, width=500, scrolling=False)
    
    st.write("The table shows the contributions of each feature to the prediction for the selected sample.")
    st.write("And the plot shows how the model makes a prediction for a sample. The base value is the average prediction of the model. The SHAP values show how each feature contributes to the prediction.")
    st.write("The SHAP values are added one at a time, starting from the left, until the current model prediction is reached.")
    st.info("One of the fundamental properties of Shapley values is that they always sum up to the difference between the game outcome when all players are present and the game outcome when no players are present. In our case, it means that SHAP values of all the input features will always sum up to the difference between baseline (expected) model output and the current model output for the prediction being explained.", icon="ℹ️")
    st.balloons()
        
    
if __name__ == "__main__":
    main()
