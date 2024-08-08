import re
import pandas as pd 
import sqlite3 
from datetime import datetime 

def extract_log_data(log_file):
    with open(log_file, 'r') as file:
        log_lines = file.readlines()

    log_data = []
    for line in log_lines:
        match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), (\w+), (.+)", line)
        if match:
            log_data.append({
                "timestamp": match.group(1),
                "log_level": match.group(2),
                "message": match.group(3)
            })

    return pd.DataFrame(log_data)

def transform_log_data(log_df):
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
    error_logs = log_df[log_df['log_level'] == 'ERROR']
    log_level_counts = log_df['log_level'].value_counts()
    return error_logs, log_level_counts

def load_data_to_db(log_df, db_nme='logs_analysis.db'):
    conn = sqlite3.connect(db_name)
    log_df.to_sql('logs', conn, if_exists='replace', index=False)
    conn.close()

# Usage
log_file = "path_to_log_file.log"
extracted_data = extract_log_data(log_file)
error_logs, log_level_counts = transform_log_data(extracted_data)
load_data_to_db(extracted_data)