import pandas as pd
import tkinter as tk
from tkinter import ttk
import folium
from tkinter import filedialog
from tkinter import messagebox
import webbrowser

# Load dataset
file_path = r'C:\Users\kyleh\OneDrive\Desktop\PowerBI Dashboards\credit_card_transactions.csv'
data = pd.read_csv(file_path)

# Initialize the main window
root = tk.Tk()
root.title("Fraud Detection Spreadsheet Viewer")

# Create a frame for the table
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Function to display the table
def display_table():
    for widget in frame.winfo_children():
        widget.destroy()
    
    # Select columns to display
    columns = ['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'merch_zipcode']
    df = data[columns].head(10)
    
    tree = ttk.Treeview(frame, columns=columns, show='headings')
    
    # Define column headings
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor='w')
    
    # Insert data rows
    for index, row in df.iterrows():
        tree.insert("", "end", values=list(row))
    
    tree.pack(fill=tk.BOTH, expand=True)

# Function to create and display the map
def display_map():
    # Create a map centered around a default location (can be adjusted)
    map_center = [37.0902, -95.7129]  # Default center (USA)
    fraud_map = folium.Map(location=map_center, zoom_start=4)

    # Select rows with non-null `merch_zipcode` and filter fraud data
    df_fraud = data[['merch_lat', 'merch_long', 'is_fraud']].dropna()
    fraud_data = df_fraud[df_fraud['is_fraud'] == 1].head(10)  # Limit to first 10 fraud entries

    # Add markers for fraud locations
    for _, row in fraud_data.iterrows():
        folium.Marker(
            location=[row['merch_lat'], row['merch_long']],
            popup=f"Location: {row['merch_lat']}, {row['merch_long']}",
            icon=folium.Icon(color='red')
        ).add_to(fraud_map)
    
    # Save map to HTML file
    map_file = 'fraud_map.html'
    fraud_map.save(map_file)
    
    # Open the map in the default web browser
    webbrowser.open(map_file)

# Create UI elements
def create_buttons():
    button_frame = ttk.Frame(root)
    button_frame.pack(pady=10)

    # Create buttons for displaying table and map
    button_table = ttk.Button(button_frame, text="Show Data Table", command=display_table)
    button_table.pack(padx=10, pady=5)
    
    button_map = ttk.Button(button_frame, text="Show Fraud Map", command=display_map)
    button_map.pack(padx=10, pady=5)

create_buttons()

# Run the application
root.mainloop()
