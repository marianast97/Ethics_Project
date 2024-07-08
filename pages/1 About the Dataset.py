import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import shap

st.set_option('deprecation.showPyplotGlobalUse', False)

# Set page configuration
st.set_page_config(
    #page_title="Maternal Health Risk Predictor",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./Maternal Health Risk Data Set.csv")
    target = 'RiskLevel'
    return df, target

def main():
    # Page Title
    st.markdown("## About the Dataset")
    st.title("Maternal Health Risk Predictor")
    st.write("\n\n\n")

    st.write("Data was collected from five hospitals and one maternity clinic in Dhaka, Bangladesh. The collection process was done via a questionnaire, made from previous medical studies and by discussions with people in the field of medicine. The patient’s health data was also collected through wearable sensing devices, and the final dataset was made by merging the two.")

    # Add DOI link to sidebar
    st.write(
        """
        **DOI**: [10.24432/C5DP5D](https://archive.ics.uci.edu/dataset/863/maternal+health+risk)
        """
    )
    
    st.write("**Author**: Marzia Ahmed")
    st.write("**Dataset donated on**: 14/8/2023")

    # Load Dataset
    df, target = load_data()
    
    # Display dataset
    st.markdown("### Maternal Health Risk Dataset")

    st.dataframe(df)

    categorical = ["Age"]
    attributes = df.columns.tolist()

    st.sidebar.header("Selection Options")
    selected_attribute = st.sidebar.selectbox('Select an attribute', attributes)

    st.sidebar.header("Filter Options")

    # Initialize filtered_df based on the data type of selected_attribute
    if df[selected_attribute].dtype == 'object':
        unique_values = df[selected_attribute].unique().tolist()
        selected_values = st.sidebar.multiselect(f'Select {selected_attribute} values', unique_values, default=unique_values)
        filtered_df = df[df[selected_attribute].isin(selected_values)]
    else:
        min_value, max_value = st.sidebar.slider(
            f'Filter {selected_attribute} values',
            int(df[selected_attribute].min()),
            int(df[selected_attribute].max()),
            (int(df[selected_attribute].min()), int(df[selected_attribute].max()))
        )
        filtered_df = df[(df[selected_attribute] >= min_value) & (df[selected_attribute] <= max_value)]


    st.write("\n\n\n")
    st.write("### Distribution of the attributes")
    st.write("In the sidebar, select the attribute for which you want to see the distribution.")
    st.write("You can also filter the data based on the attribute values.")
    st.write("The graph shows the distribution of the selected filtered attribute collored by the risk classification level.")

    # Plotting
    custom_palette = {'low risk': '#00B38A', 'high risk': '#EA324C', 'mid risk': '#F2AC42'}

    # Map risk levels to colors
    color_discrete_map = {'low risk': custom_palette['low risk'], 'mid risk': custom_palette['mid risk'], 'high risk': custom_palette['high risk']}

    # Plotting with Plotly
    fig = px.histogram(
        filtered_df, 
        x=selected_attribute, 
        color='RiskLevel', 
        title=f"{selected_attribute} by Risk Classification",
        category_orders={"RiskLevel": ['low risk', 'mid risk', 'high risk']}, 
        barmode='stack',
        height=300, 
        width=600,
        color_discrete_map=color_discrete_map
    )   

    fig.update_traces(marker_line=dict(width=1, color='white'))

    st.plotly_chart(fig)
    

if __name__ == "__main__":
    main()
