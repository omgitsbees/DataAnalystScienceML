import pandas as pd

def load_dataset(file_path):
    """Load dataset from CSV file."""
    return pd.read_csv(file_path)

def display_basic_info(df):
    """Display basic statistics and information about the dataset."""
    print("Basic Information:")
    print(df.info())
    print("\nStatistics:")
    print(df.describe())
    
def handle_missing_values(df, strategy='mean'):
    """Handle missing values in the dataset."""
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError("Strategy not recognized. Use 'mean' or 'drop'.")
    
def remove_duplicates(df):
    """Remove duplicate rows from the dataset."""
    return df.drop_duplicates()

def normalize_data_format(df, column, date_format='%Y-%m-%d'):
    """Normalize date format in the specified column."""
    df[column] = pd.to_dateime(df[column]).dt.strftime(date_format)
    return df

def export_dataset(df, file_path):
    """Export the cleaned dataset to a new CSV file."""
    df.to_csv(file_path, index=False)
    print(f"dataset exported to {file_path}")
    
def main():
    file_path = input("Enter the path to your CSV file: ")
    df = load_dataset(file_path)
    
    display_basic_info(df)
    
    while True:
        print("\nSelect an option:")
        print("1. Handle missing values")
        print("2. Remove duplicates")
        print("3. Normalize date format")
        print("4. Export dataset")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            strategy = input("Enter strategy for handling missing values ('mean' or 'drop'): ")
            df = handle_missing_values(df, strategy=strategy)
        elif choice == '2':
            df = remove_duplicates(df)
        elif choice == '3':
            column = input("Enter the name of the date column: ")
            date_format = input("Enter the desired formate (e.g., '%Y-%m-%d'): ")
            df = normalize_data_format(df, column=column, date_format=date_format)
        elif choice == '4':
            output_path = input("Enter the path to save the cleaned CSV file: ")
            export_dataset(df, output_path)
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")
            
if __name__ == "__main__":
    main()