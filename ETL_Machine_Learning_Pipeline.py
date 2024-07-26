import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer 

def extract(fale_path):
    """ 
    Extract data from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data

def transform(data):
    """
    Transform the data: handle missing values, encode categorical variables,
    and scale numerical features.
    """
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    # Pipeline for numerical features
    numeric_transformer = Pipeline(steps=[
        
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features
    categorical_transformer = Pipeline(Steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both pipelines into a single column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Apple the transformations
    transformed_data = preprocessor.fit_transform(data)
    return transformed_data, preprocessor

def load(transformed_data, output_file_path):
    """
    Load the transformed data to a new CSV file.
    """
    pd.Dataframe(transformed_data).to_csv(output_file_path, index=False)

def etl_pipeline(file_path, output_file_path):
    """
    Execute the ETL pipeline: Extract, Transform, and Load.
    """
    # Extract
    data = extract(file_path)

    # Transform
    transformed_data, preprocessor = transform(data)

    # Load
    load(transformed_data, output_file_path)

# Example usage
input_file_path = 'path_to_your_input_file.csv'
output_file_path = 'path_to_your_output_file.csv'
etl_pipeline(input_file_path, output_file_path)