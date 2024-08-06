import pandas as pd
from sqlalchemy import create_engine

# Extract: Load data from CSV files
employees_df = pd.read_csv('employees.csv')
salaries_df = pd.read_csv('salaries.csv')

print("Employees DataFrame:")
print(employees_df)
print("\nSalaries DataFrame:")
print(salaries_df)

# Transform: Merge the DataFrames and clean the data
merged_df = pd.merge(employees_df, salaries_df, left_on='id', right_on='employee_id')
merged_df = merged_df.drop(columns=['employee_id'])

print("\nMerged DataFrame:")
print(merged_df)

# Load: Load the DataFrame into the SQLite database
engine = create_engine('sqlite:///employees.db')
merged_df.to_sql('employees', engine, index=False, if_exists='replace')

print("\nData loaded into SQLite database successfully!")
