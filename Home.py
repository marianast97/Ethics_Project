import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set page configuration
st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    layout="centered", #centered #wide
    initial_sidebar_state="auto", #collapsed
)

# Load the Data set
df = pd.read_csv("./Maternal Health Risk Data Set.csv")

st.markdown("## Home")
st.title("Maternal Health Risk Predictor")

st.markdown("""
This app allows you to explore the predictions of maternal health, the dataset used to train the prediction model and it aims to explain how the model performed its prediction.
""")

# Display image on Home
st.image("./MomArt.jpeg", use_column_width=True)

# Sidebar Content
st.sidebar.markdown("---")
st.sidebar.markdown("### App Developers:")
st.sidebar.markdown("##### Aditya Panchal")
st.sidebar.markdown("##### Mariana Steffens")
st.sidebar.markdown("##### Navya Reddy Tiyyagura")
st.sidebar.markdown("##### Se Yeon Kim")
st.sidebar.markdown("---")
st.sidebar.markdown("##### App developed as part of the Human Centered Data Science course at the Freie Universität Berlin")
st.sidebar.markdown("\n\n\n")
st.sidebar.markdown("##### 02.07.2024")

