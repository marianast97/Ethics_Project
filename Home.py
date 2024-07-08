import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Data set
df = pd.read_csv("./Maternal Health Risk Data Set.csv")

st.title("CDC Diabetes Health Indicators Data Explorer")
st.write("This app allows you to explore the [CDC Diabetes Health Indicators Data set](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators).")
st.write("It is a subset of the Behavioral Risk Factor Surveillance System (BRFSS) data.")
st.write("It aims to better identify the relationship between lifestyle and diabetes in the US.")

# Display the dataset
st.write("\n\n\n")
st.write("### ➾ CDC Diabetes Health Indicators")
st.dataframe(df)

categorical = ["Age"]
attributes = df.columns.tolist()

st.sidebar.header("☢ Selection Options")
selected_attribute = st.sidebar.selectbox('Select an attribute', attributes)

st.sidebar.header("☢ Filter Options")
min_value, max_value = st.sidebar.slider(
    f'Filter {selected_attribute} values',
    int(df[selected_attribute].min()),
    int(df[selected_attribute].max()),
    (int(df[selected_attribute].min()), int(df[selected_attribute].max()))
)

# Filter data based on slider
filtered_df = df[(df[selected_attribute] >= min_value) & (df[selected_attribute] <= max_value)]

st.write("\n\n\n")
st.write("### ➾ Distribution of the attributes")
st.write("Select the attribute, in the sidebar, that you want to see the distribution of.")
st.write("You can also filter the data based on the attribute values.")
st.write("The plot will show the distribution of the selected filtered attribute along with the proprtion of the distribution that has diabetes and not.")

# Plotting
custom_palette = {'low risk': '#9bc1bc', 'high risk': '#ed6a5a', 'mid risk': '#f4f1bb'}
fig, ax = plt.subplots()
sns.histplot(
    data=filtered_df,
    x=selected_attribute,
    hue='RiskLevel',
    multiple='stack',
    ax=ax,
    stat='probability',
    palette=custom_palette
)
ax.set_title(f'Distribution of {selected_attribute} with RiskLevel')
ax.set_xlabel(selected_attribute)
ax.set_ylabel('Proportion')

st.pyplot(fig)
st.write("The y-axis is normalized to show the proportion of the distribution instead of showing value counts.")
st.write("The blue area represents the proportion without diabetes and the orange area represents the proportion with diabetes.")

