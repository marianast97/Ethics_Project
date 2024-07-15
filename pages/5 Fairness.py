import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import fairness_functions as ff


st.set_option('deprecation.showPyplotGlobalUse', False)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./Maternal Health Risk Data Set.csv")
    target = 'RiskLevel'
    return df, target

def main():
    st.title("Fairness Metrics")

    df, target = load_data()

    # create a list of the conditions
    conditions = [
        (df['Age'] >= 10) & (df['Age'] <= 19),
        (df['Age'] >= 20) & (df['Age'] <= 34),
        (df['Age'] > 34)
        ]
    # create a list of the values we want to assign for each condition
    values = ['teenager', 'adult', 'advanced maternal age']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df['AgeGroup'] = np.select(conditions, values, default='unknown')

    # Encode the target variable
    # high risk = 0, low risk = 1, mid risk = 2, adult = 0, advanced maternal age = 1, teenager = 2
    le = LabelEncoder()
    df['AgeGroupEncoded'] = le.fit_transform(df['AgeGroup'])
    df['RiskLevelEncoded'] = le.fit_transform(df['RiskLevel'])

    # Split the data
    X = df.drop(['AgeGroup','RiskLevel', 'RiskLevelEncoded'], axis=1)
    y = df['RiskLevelEncoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a One-vs-Rest classifier with RandomForest
    rf_classifier = RandomForestClassifier(random_state=42)
    ovr = OneVsRestClassifier(rf_classifier)
    ovr.fit(X_train, y_train)

    # Make predictions
    y_pred = ovr.predict(X_test)

    # Add predictions to the DataFrame for fairness functions
    df_test = X_test.copy()
    df_test['TrueLabel'] = y_test
    df_test['Prediction'] = y_pred

    # Group Fairness
    high_risk_adult = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 0, 'Prediction', 0)
    high_risk_adv_age = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 1, 'Prediction', 0)
    high_risk_teen = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 2, 'Prediction', 0)
    mid_risk_adult = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 0, 'Prediction', 2)
    mid_risk_adv_age = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 1, 'Prediction', 2)
    mid_risk_teen = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 2, 'Prediction', 2)
    low_risk_adult = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 0, 'Prediction', 1)
    low_risk_adv_age = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 1, 'Prediction', 1)
    low_risk_teen = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 2, 'Prediction', 1)

    st.write("**Group Fairness:**")
    group_fairness_df = pd.DataFrame({
        "Age Group": ["Teenager", "Adult", "Advanced Maternal Age"],
        "High Risk": [f"{high_risk_teen*100:.2f}%", f"{high_risk_adult*100:.2f}%", f"{high_risk_adv_age*100:.2f}%"],
        "Mid Risk": [f"{mid_risk_teen*100:.2f}%", f"{mid_risk_adult*100:.2f}%", f"{mid_risk_adv_age*100:.2f}%"],
        "Low Risk": [f"{low_risk_teen*100:.2f}%", f"{low_risk_adult*100:.2f}%", f"{low_risk_adv_age*100:.2f}%"]
    })
    group_fairness_df = group_fairness_df.style.set_properties(**{'text-align': 'left'})
    group_fairness_df.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
    st.dataframe(group_fairness_df, width=500, hide_index=True)

    st.write("Group fairness aims to ensure that certain desirable properties or outcomes are evenly distributed among groups defined by sensitive attributes, such as age, gender, race, or socioeconomic status. For the sake of this project, we assumed age to be the sensitive attribute. Each group need to have the same probability of being assigned to the predicted class.")
    st.write("For example, if we investigate the 'Age Group' then all groups, protected and unprotected should ideally have the same probability to receive a high risk, mid risk, and low risk prediction. Mathematically this is stated as followed:")
    st.write(r"$P(RiskPrediction = high \vert Age Group = Advanced Maternal Age) == P(RiskPrediction = high \vert Age Group = Adult)$")

    # Predictive Parity
    ppv_adult = ff.predictive_parity(df_test, "AgeGroupEncoded", 0, "Prediction", "TrueLabel")
    ppv_adv_age = ff.predictive_parity(df_test, "AgeGroupEncoded", 1, "Prediction", "TrueLabel")
    ppv_teenager = ff.predictive_parity(df_test, "AgeGroupEncoded", 2, "Prediction", "TrueLabel")

    st.write("**Predictive Parity:**")
    ppv_df = pd.DataFrame({
        "Age Group": ["Teenager", "Adult", "Advanced Maternal Age"],
        "Positive Predictive Value (PPV)": [f"{ppv_teenager*100:.2f}%", f"{ppv_adult*100:.2f}%", f"{ppv_adv_age*100:.2f}%"]
    })
    ppv_df = ppv_df.style.set_properties(**{'text-align': 'left'})
    ppv_df.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
    st.dataframe(ppv_df, width=500, hide_index=True)

    st.write("Predictive Parity measures the proportion of positive predictions that are actually correct. A high PPV indicates that we can be sure that a positive prediction is true.")
    st.write("In our example, we would like to analyze whether different age groups are less likely to truly belong to the prediction and whether there is a significant difference among these three groups.")

    # False Positive Error Rate Balance
    fpr_adult = ff.fp_error_rate_balance(df_test, "AgeGroupEncoded", 0, "Prediction", "TrueLabel")
    fpr_adv_age = ff.fp_error_rate_balance(df_test, "AgeGroupEncoded", 1, "Prediction", "TrueLabel")
    fpr_teenager = ff.fp_error_rate_balance(df_test, "AgeGroupEncoded", 2, "Prediction", "TrueLabel")

    st.write("**False Positive Error Rate Balance:**")
    fpr_df = pd.DataFrame({
        "Age Group": ["Teenager", "Adult", "Advanced Maternal Age"],
        "False Positive Rate (FPR)": [f"{fpr_teenager*100:.2f}%", f"{fpr_adult*100:.2f}%", f"{fpr_adv_age*100:.2f}%"]
    })
    fpr_df = fpr_df.style.set_properties(**{'text-align': 'left'})
    fpr_df.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
    st.dataframe(fpr_df, width=500, hide_index=True)

    st.write("False Positive Error Rate measures the proportion of negative cases that are incorrectly classified as positive. In other words, it tells you how often a model incorrectly predicts the positive class for cases that should be in the negative class.")
    st.write("We would like to analyze if any age group is favored by having a higher FPR than the other, thus predicting it more often to be prone to other types risks even though they are not.")

if __name__ == "__main__":
    main()
