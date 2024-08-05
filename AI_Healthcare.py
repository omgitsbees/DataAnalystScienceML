import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import PySimpleGUI as sg 

# Sample Data
data = {
    'age': [25, 34, 45, 52, 23, 40, 60, 48, 33, 55],
    'bmi': [22.4, 27.8, 30.1, 25.6, 24.5, 28.9, 31.2, 29.4, 26.7, 32.1],
    'blood_pressure': [120, 130, 140, 135, 125, 138, 145, 132, 128, 142],
    'condition': [0, 1, 1, 0, 0, 1, 1, 0, 0, 1]  # 0: No condition, 1: Has condition
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['age', 'bmi', 'blood_pressure']]
y = df['condition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# GUI layout
layout = [
    [sg.Text('AI for healthcare')],
    [sg.Text('Age'), sg.InputText(key='age')],
    [sg.Text('BMI'), sg.InputText(key='BMI')],
    [sg.Text('Predict'), sg.Button('Exit')],
    [sg.Text('Prediction:'), sg.Text('', key='prediction')],
    [sg.Text('Accuracy:'), sg.Text(f'{accuracy:.2f}')],
    [sg.Text('Confusion Matrix:'), sg.Text(f'{conf_matrix}')],
    [sg.Text('Classification Report:'), sg.Text(f'{class_report}')]
]

# Create the window
window = sg.Window('AI for Healthcare', layout)

# Event loop
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == 'Predict':
        age = float(values['age'])
        bmi = float(values['bmi'])
        blood_pressure = flat(values['blood_pressure'])
        input_data = np.array([[age, bmi, blood_pressure]])
        prediction = model.predict(input_data)[0]
        window['prediction'].update('Has condition' if prediction == 1 else 'No condition')

window.close()