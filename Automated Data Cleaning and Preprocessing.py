import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.impute import SimpleImputer 

def clean_and_preprocess_data(df, target_column=None):
    # 1. Handle Missing Values
    # Impute missing values with mean for numerical columns and mode for categorical columns
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = num_imputer.fit_transform(df[[col]])
        else:
            df[col] = cat_imputer.fit_transform(df[[col]])

    # 2. Remove Duplicates
    df.drop_duplicates(inplace=True)

    # 3. Standardize Formats
    # Convert all text to lowercase
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)

    # 4. Encode Categorical Variables
    # Label encode categorical variables, except the target column if specified
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_column:
            df[col] = le.fit_transform(df[col])

    # 5. Normalize Numerical Data
    # Standardize numerical colums (z-score normalization)
    scaler = StandardScaler()
    df[df.select_dtypes(include=['int64', 'float64']),columns] = scaler.fit_transform(
        df.select_dtypes(include=['int64', 'float64'])
    )

    return df

# Example usage
if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv("your_dataset_csv")

    # Specify the target column if applicable, else leave as None
    target_column = 'your_target_column'

    # Clean and preprocess the dataset
    clean_df = clean_and_preprocess_data(df, target_column)

    # Save the cleaned and preprocess data
    clean_df.to_csv("cleaned_dataset.csv", index=False)
    print("Data cleaning and preprocessing completed. Cleaned data saved to 'cleaned_dataset.csv")