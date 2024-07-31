import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import h2o
from h2o.automl import H2OAutoML
from h2o import H2OFrame
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score

# Initialize H2O
h2o.init()

# Function to perform AutoML
def run_automl():
    try:
        # Load dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to H2OFrame
        train = H2OFrame(pd.DataFrame(X_train, columns=iris.feature_names).assign(y=y_train))
        test = H2OFrame(pd.DataFrame(X_test, columns=iris.feature_names).assign(y=y_test))
        
        # Set predictors and response
        predictors = iris.feature_names
        response = 'y'
        
        # Train AutoML model
        aml = H2OAutoML(max_runtime_secs=300)
        aml.fit(x=predictors, y=response, training_frame=train)
        
        # Predict and evaluate
        preds = aml.predict(test)
        y_pred = preds.as_data_frame()['predict'].astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Prepare results
        leaderboard_df = aml.leaderboard.as_data_frame()
        result_text = f"Best Model: {aml.leaderboard[0, 'model_id']}\nAccuracy: {accuracy:.2f}\n"
        result_text += "\nLeaderboard:\n" + leaderboard_df.to_string(index=False)
        
        # Update text area with results
        result_text_area.config(state=tk.NORMAL)
        result_text_area.delete(1.0, tk.END)
        result_text_area.insert(tk.END, result_text)
        result_text_area.config(state=tk.DISABLED)
    
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create GUI
def create_gui():
    global result_text_area
    
    root = tk.Tk()
    root.title("Automated Machine Learning with H2O")
    
    # Create a frame for the button and result area
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Add a label
    label = ttk.Label(frame, text="Click the button to run AutoML:")
    label.grid(row=0, column=0, padx=5, pady=5)
    
    # Add a button to run AutoML
    run_button = ttk.Button(frame, text="Run AutoML", command=run_automl)
    run_button.grid(row=1, column=0, padx=5, pady=5)
    
    # Add a text area to display results
    result_text_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=80, height=20, state=tk.DISABLED)
    result_text_area.grid(row=2, column=0, padx=5, pady=5)
    
    # Start the GUI loop
    root.mainloop()

if __name__ == "__main__":
    create_gui()
