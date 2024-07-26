import pandas as pd
import os
import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext

# File path (relative to the current script)
source_file = 'data.csv'

# Step 1: Extract data from a source (e.g., CSV file, database)
def extract_data(source_file):
    data = pd.read_csv(source_file)  # Example for CSV extraction
    return data

# Step 2: Transform the data (cleaning, filtering, adding columns, etc.)
def transform_data(data):
    print("Columns in the DataFrame:", data.columns)  # Print columns to debug

    # Example transformation: convert date strings to datetime objects
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
    else:
        print("Error: 'datetime' column not found")
    
    # Example transformation: filter data for specific conditions
    if 'money' in data.columns:
        data_filtered = data[data['money'] > 30]  # Filter for money > 30
    else:
        print("Error: 'money' column not found")
        data_filtered = data  # Return unfiltered data for now
    
    return data_filtered

# Step 3: Save the transformed data locally as a CSV file
def save_data(data, target_file):
    try:
        data.to_csv(target_file, index=False)
        print("Data saved successfully to:", target_file)
        messagebox.showinfo("Success", f"Data saved successfully to {target_file}")
    except Exception as e:
        print("Error:", e)
        messagebox.showerror("Error", f"Error occurred: {e}")

# Function to display original and transformed data in a GUI
def display_data_gui(original_data, transformed_data):
    # Create a GUI window
    window = tk.Tk()
    window.title("ETL Pipeline Results")

    # Create labels for original and transformed data
    original_label = tk.Label(window, text="Original Data", font=("Arial", 12, "bold"))
    original_label.pack(pady=10)
    
    # Create scrolled text widget for original data
    original_text_area = scrolledtext.ScrolledText(window, width=60, height=20)
    original_text_area.pack(padx=10, pady=5)
    original_text_area.insert(tk.INSERT, original_data.to_string(index=False))

    # Create labels for changes
    changes_label = tk.Label(window, text="Changes", font=("Arial", 12, "bold"))
    changes_label.pack(pady=10)

    # Create scrolled text widget for transformed data
    transformed_text_area = scrolledtext.ScrolledText(window, width=60, height=20)
    transformed_text_area.pack(padx=10, pady=5)
    transformed_text_area.insert(tk.INSERT, transformed_data.to_string(index=False))

    # Run the GUI
    window.mainloop()

# Example usage of the ETL pipeline with GUI
if __name__ == "__main__":
    try:
        # Step 1: Extract data from CSV file
        extracted_data = extract_data(source_file)
        
        # Step 2: Transform the extracted data
        transformed_data = transform_data(extracted_data)
        
        # Step 3: Save the transformed data locally as a CSV file
        target_file = 'transformed_data.csv'
        save_data(transformed_data, target_file)
        
        # Display original and transformed data in GUI
        display_data_gui(extracted_data, transformed_data)
    
    except Exception as e:
        print("Error:", e)
        messagebox.showerror("Error", f"Error occurred: {e}")
