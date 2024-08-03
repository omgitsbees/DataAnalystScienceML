import pandas as pd
from sqlalchemy import create_engine

# Extract
def extract(file_path):
    data = pd.read_csv(file_path)  # Corrected method name
    return data

# Transform
def transform(data):
    # Example transformation: Convert all column names to lowercase
    data.columns = [col.lower() for col in data.columns]
    return data

# Load
def load(data, db_url, table_name):
    engine = create_engine(db_url)
    data.to_sql(table_name, con=engine, if_exists='replace', index=False)

# Main ETL function
def etl_pipeline(file_path, db_url, table_name):
    data = extract(file_path)
    data = transform(data)
    load(data, db_url, table_name)

# Example usage
file_path = 'path/to/your/csvfile.csv'
db_url = 'sqlite:///your_database.db'  # Example using SQLite
table_name = 'your_table_name'

etl_pipeline(file_path, db_url, table_name)
