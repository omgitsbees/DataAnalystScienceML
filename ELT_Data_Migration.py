import pandas as pd
from sqlalchemy import create_engine


# Extract
def extract_data(csv_file_path):
    """
    Extract data from a CSV file.
    
    :param csv_file_path: Path to the CSV file.
    :return: DataFrame containing the extracted data.
    """
    data = pd.read_csv(csv_file_path)
    return data


# Transform
def transform_data(data):
    """
    Transform the data.
    
    :param data: DataFrame containing the extracted data.
    :return: DataFrame containing the transformed data.
    """
    # Example transformation: remove rows with missing values
    data_cleaned = data.dropna()
    
    # Additional transformations can be added here
    
    return data_cleaned


# Load
def load_data(data, database_url, table_name):
    """
    Load the data into a SQL database.
    
    :param data: DataFrame containing the transformed data.
    :param database_url: SQLAlchemy database URL.
    :param table_name: Name of the table where data will be loaded.
    """
    engine = create_engine(database_url)
    
    # Load data into the specified table
    data.to_sql(table_name, engine, if_exists='replace', index=False)


# Main ETL function
def etl(csv_file_path, database_url, table_name):
    # Step 1: Extract
    data = extract_data(csv_file_path)
    print("Data extracted successfully.")

    # Step 2: Transform
    transformed_data = transform_data(data)
    print("Data transformed successfully.")

    # Step 3: Load
    load_data(transformed_data, database_url, table_name)
    print("Data loaded successfully.")


# Example usage
csv_file_path = 'path/to/your/source_data.csv'
database_url = 'sqlite:///path/to/your/destination_database.db'  # Example: SQLite database URL
table_name = 'your_table_name'

etl(csv_file_path, database_url, table_name)
