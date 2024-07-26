import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button, Text, Scrollbar

# Function to load data
def load_data():
    global df, pivot_table, train_data, test_data, train_data_matrix, test_data_matrix
    file_path = filedialog.askopenfilename()
    if file_path:
        columns = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(file_path, sep='\t', names=columns)
        df.drop('timestamp', axis=1, inplace=True)
        pivot_table = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        train_data, test_data = train_test_split(pivot_table, test_size=0.2, random_state=42)
        train_data_matrix = train_data.values
        test_data_matrix = test_data.values
        messagebox.showinfo("Information", "Data loaded successfully!")
    else:
        messagebox.showwarning("Warning", "No file selected!")

# Function to perform collaborative filtering
def collaborative_filtering():
    global user_similarity, user_prediction
    user_similarity = cosine_similarity(train_data_matrix)
    user_prediction = predict(train_data_matrix, user_similarity, type='user')
    messagebox.showinfo("Information", "Collaborative filtering completed!")

# Function to make predictions
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    return pred

# Function to compute RMSE
def compute_rmse():
    rmse_value = rmse(user_prediction, test_data_matrix)
    result_text.insert(tk.END, f'User-based CF RMSE: {rmse_value}\n')

# Function to compute RMSE
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

# Create the main window
root = tk.Tk()
root.title("Collaborative Filtering Recommendation System")

# Add buttons and text box to the GUI
load_button = Button(root, text="Load Data", command=load_data)
load_button.pack(pady=10)

filter_button = Button(root, text="Collaborative Filtering", command=collaborative_filtering)
filter_button.pack(pady=10)

rmse_button = Button(root, text="Compute RMSE", command=compute_rmse)
rmse_button.pack(pady=10)

result_label = Label(root, text="Results:")
result_label.pack(pady=10)

result_text = Text(root, height=10, width=50)
result_text.pack(pady=10)

scrollbar = Scrollbar(root)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
result_text.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=result_text.yview)

# Run the GUI main loop
root.mainloop()
