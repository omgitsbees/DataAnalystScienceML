import re
import pandas as pd
import sqlite3

# Step 1: Extract
def extract_log_data(file_path):
    with open(file_path, 'r') as file:
        log_data = file.readlines()
    return log_data

# Step 2: Transform
def transform_log_data(log_data):
    log_entries = []
    for line in log_data:
        match = re.match(r'(\w+)\s+([\d-]+\s[\d:,]+)\s(\w+):\s(.+)', line)
        if match:
            log_entries.append(match.groups())
    
    # Convert to DataFrame
    df = pd.DataFrame(log_entries, columns=['LogLevel', 'Timestamp', 'Module', 'Message'])
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    return df

# Step 3: Load
def load_data_to_db(df, db_path):
    conn = sqlite3.connect(db_path)
    df.to_sql('logs', conn, if_exists='replace', index=False)
    conn.close()

# Full ETL Process
def etl_process(log_file_path, db_path):
    log_data = extract_log_data(log_file_path)
    df = transform_log_data(log_data)
    load_data_to_db(df, db_path)

# Run ETL
log_file_path = 'logs.txt'  # Path to your log file
db_path = 'logs.db'         # Path to your SQLite database

etl_process(log_file_path, db_path)

print("ETL process completed successfully.")
