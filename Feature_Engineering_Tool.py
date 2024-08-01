import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2

class FeatureEngineeringTool:
    def __init__(self, df):
        self.df = df

    def handle_missing_values(self, strategy='mean'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mode().iloc[0])
        return self.df

    def encode_categorical(self, columns, method='onehot'):
        if method == 'onehot':
            self.df = pd.get_dummies(self.df, columns=columns)
        elif method == 'label':
            le = LabelEncoder()
            for col in columns:
                self.df[col] = le.fit_transform(self.df[col])
        return self.df

    def scale_features(self, method='minmax'):
        scaler = MinMaxScaler()
        self.df = pd.DataFrame(scaler.fit_transform(self.df), columns=self.df.columns)
        return self.df

    def apply_pca(self, n_components):
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(self.df)
        self.df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(n_components)])
        return self.df

    def select_k_best(self, target, k=10):
        y = self.df[target]
        X = self.df.drop(columns=[target])
        selector = SelectKBest(score_func=chi2, k=k)
        X_new = selector.fit_transform(X, y)
        self.df = pd.DataFrame(X_new, columns=[X.columns[i] for i in selector.get_support(indices=True)])
        self.df[target] = y.values
        return self.df

    def add_custom_feature(self, func, new_column_name):
        self.df[new_column_name] = self.df.apply(func, axis=1)
        return self.df

# Sample data
data = {
    'A': [1, 2, np.nan, 4],
    'B': [4, np.nan, 6, 8],
    'C': ['cat', 'dog', 'cat', 'bird'],
    'D': [1, 1, 0, 0]
}
df = pd.DataFrame(data)

# Initialize the tool
fe_tool = FeatureEngineeringTool(df)

# Handle missing values
df = fe_tool.handle_missing_values(strategy='median')

# Encode categorical features
df = fe_tool.encode_categorical(columns=['C'], method='onehot')

# Add custom feature
def custom_feature(row):
    return row['A'] * row['B']

df = fe_tool.add_custom_feature(custom_feature, 'CustomFeature')

# Scale features
df = fe_tool.scale_features(method='minmax')

# Select K Best features
df['D'] = data['D']  # Adding target back
df = fe_tool.select_k_best(target='D', k=2)

# Apply PCA
df = fe_tool.apply_pca(n_components=2)

print(df)
