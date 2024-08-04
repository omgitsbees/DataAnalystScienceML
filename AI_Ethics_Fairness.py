import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample data
data = {
    'gender': ['male', 'female', 'male', 'female', 'male', 'female'],
    'race': ['white', 'black', 'asian', 'white', 'black', 'asian'],
    'actual': [1, 0, 1, 0, 1, 0],
    'predicted': [1, 0, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

# Function to calculate fairness metrics
def calculate_fairness(df, sensitive_attribute):
    groups = df[sensitive_attribute].unique()
    fairness_metrics = {}

    for group in groups:
        group_df = df[df[sensitive_attribute] == group]
        accuracy = accuracy_score(group_df['actual'], group_df['predicted'])
        tn, fp, fn, tp = confusion_matrix(group_df['actual'], group_df['predicted']).ravel()
        false_positive_rate = fp / (fp + tn)
        false_negative_rate = fn / (fn + tp)

        fairness_metrics[group] = {
            'accuracy': accuracy,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate
        }

    return fairness_metrics

# Calculate fairness metrics for gender
gender_fairness = calculate_fairness(df, 'gender')
print("Fairness metrics by gender:")
print(gender_fairness)

# Calculate fairness metrics for race
race_fairness = calculate_fairness(df, 'race')
print("\nFairness metrics by race:")
print(race_fairness)