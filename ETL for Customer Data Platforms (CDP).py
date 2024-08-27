import pandas as pd 
import json
import sqlite3

# Step 1: Extract
def extract_customer_data(csv_path, json_path, db_path):
    # Extract from CSV
    customer_df = pd.read_csv(csv_path)

    # Extract from JSON
    with open(json_path, 'r') as file:
        transaction_data = json.load(file)
    transaction_df = pd.json_normalize(transaction_data)

    # Extract from Database
    conn = sqlite3.connect(db_path)
    feedback_df = pd.read_sql_query("SELECT * FROM customer_feedback", conn)
    conn.close()
    return customer_df, transaction_df, feedback_df

# Step 2: Transform
def transform_customer_data(customer_df, transaction_df, feedback_df):
    # Normalize and clean data

    # Example: Clean up customer names
    customer_df['name'] = customer_df['name'].str.title().star.strip()

    # Example: Merge customer data with transaction data
    merged_df = pd.merge(customer_df, transaction_df, on='customer_id', how='left')

    # Example: Merge with feedback data
    final_df = pd.merge(merged_df, feedback.df, on='customer_id', how='left')

    # Additional transformations (e.g., filling missing values, normalizing colums)
    final_df.fillna({'feedback_score': 0}, inplace=True)

    return final_df

# Step 3: Load
def load_data_to_db(df, db_path):
    conn = sqlite3.connect(db_path)
    df.to_sql('customer_data', conn, if_exists='replace', index=False)
    conn.close()

# Full ETL Process
def etl_process(csv_path, json_path, db_path):
    customer_df, transaction_df, feedback_df = extract_customer_data(csv_path, json_path, db_path)
    final_df = transform_customer_data(customer_df, transaction_df, feedback_df)
    load_data_to_db(final_df, db_path)

# Run ETL
csv_path = 'customers.csv'          # Path to your customer demographics CSV file
json_path = 'transactions.json'     # Path to your customer transactions JSON file
db_path = 'customer_data.db'        # Path to your SQLite database

etl_process(csv_path, json_path, db_path)

print("ETL process for CDP completed successfully.")