import pandas as pd 
from sqlalchemy import create_engine 

# Extract 
def extract_from_csv(file_path):
    return pd.read_csv(file_path)

def extract_from_db(connection_string, query):
    engine = create_engine(connection_string)
    return pd.read_sql(query, engine)

# Transform
def transform(data):
    # Example transformation: Clean and aggregate data
    data['date'] = pd.to_datetime(data['date'])
    data = data.dropna(subset=['value'])
    data = data.groupby('date').sum().reset_index()
    return data

# Load
def load_to_csv(data, file_path):
    data.to_csv(file_path, index=False)

def load_to_db(data, connection_string, table_name):
    engine = create_engine(connection_string)
    data.to_sql(table_name, engine, if_exists='replace', index=False)

# Main ETL Function
def etl_process():
    # Extract data
    csv_data = extract_from_csv('source_data.csv')
    db_data = extract_from_db('sqlite:///example.db', 'SELECT * FROM source_table')

# Transform data
transformed_csv_data = transform(csv_data)
transformed_db_data = transform(db_data)

# Load data
load_to_csv(transformed_csv_data, 'transformed_data.csv')
load_to_csv(transformed_db_data, 'sqlite:///example.db', 'transformed_table')

if __name__ == "__main__":
    etl_process()