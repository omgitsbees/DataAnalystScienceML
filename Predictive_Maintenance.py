import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import requests
import zipfile
import io

# Download the dataset
url = 'https://datahub.io/machine-learning/turbofan/r/turbofan.csv'
response = requests.get(url)
response.raise_for_status()  # Ensure the request was successful

# Load the data into a DataFrame
columns = ["unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2",
           "operational_setting_3"] + [f"sensor_measurement_{i}" for i in range(1, 22)]
data = pd.read_csv(io.StringIO(response.text))

# Display the first few rows of the dataset
print("Data Preview:")
print(data.head())

# Feature Engineering
data['temp_rolling_mean'] = data['sensor_measurement_2'].rolling(window=5).mean()
data['vibration_diff'] = data['sensor_measurement_3'].diff()

# Fill NaN values resulting from rolling or differencing
data.fillna(method='bfill', inplace=True)

# Define the target variable: RUL (Remaining Useful Life)
max_cycle = data.groupby('unit_number')['time_in_cycles'].max().reset_index()
max_cycle.columns = ['unit_number', 'max_cycle']
data = data.merge(max_cycle, on=['unit_number'], how='left')
data['RUL'] = data['max_cycle'] - data['time_in_cycles']

# Binarize the target variable: 1 if RUL < threshold, else 0
threshold = 30
data['failure'] = (data['RUL'] <= threshold).astype(int)

# Drop unnecessary columns
data = data.drop(columns=['unit_number', 'time_in_cycles', 'max_cycle', 'RUL'])

# Features and target variable
X = data.drop(columns=['failure'])
y = data['failure']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'predictive_maintenance_model.pkl')

# Load the model (for deployment)
loaded_model = joblib.load('predictive_maintenance_model.pkl')

# Verify the loaded model works
y_pred_loaded = loaded_model.predict(X_test)
print("\nLoaded Model Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_loaded))
print("\nLoaded Model Classification Report:")
print(classification_report(y_test, y_pred_loaded))
