import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.fftpack import fft

# Example DataFrame
data = {
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'feature3': np.random.rand(100),
    'categorical_feature': np.random.choice(['A', 'B', 'C'], 100),
    'text_column': ['text data' for _ in range(100)],
    'date_column': pd.date_range('2021-01-01', periods=100),
    'target': np.random.randint(0, 2, 100)
}

df = pd.DataFrame(data)

# Define the feature matrix X and target y
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Interaction Features
interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interaction = interaction.fit_transform(X)

# Target Encoding
def target_encode(df, feature, target, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    encoded_feature = pd.Series(index=df.index, dtype=float)
    for train_idx, val_idx in kf.split(df):
        train_mean = df.iloc[train_idx].groupby(feature)[target].mean()
        encoded_feature.iloc[val_idx] = df.iloc[val_idx][feature].map(train_mean)
    return encoded_feature

df['encoded_feature'] = target_encode(df, 'categorical_feature', 'target')

# Feature Hashing
# Wrap each categorical value in a list to make it iterable
df['categorical_feature_list'] = df['categorical_feature'].apply(lambda x: [x])
hasher = FeatureHasher(n_features=10, input_type='string')
hashed_features = hasher.transform(df['categorical_feature_list'])

# Log Transformation
df['log_feature'] = np.log1p(df['feature1'])

# Bin Features
df['binned_feature'] = pd.cut(df['feature1'], bins=5, labels=False)

# Datetime Features
df['year'] = df['date_column'].dt.year
df['month'] = df['date_column'].dt.month
df['day_of_week'] = df['date_column'].dt.dayofweek

# Lag Features
df['lag_1'] = df['feature1'].shift(1)
df['lag_2'] = df['feature1'].shift(2)

# Rolling/Aggregate Features
df['rolling_mean'] = df['feature1'].rolling(window=3).mean()
df['rolling_sum'] = df['feature1'].rolling(window=3).sum()

# Dimensionality Reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Clustering-Based Features
kmeans = KMeans(n_clusters=3)
df['cluster'] = kmeans.fit_predict(X)

# Fourier Transform Features
df['fft_feature'] = np.abs(fft(df['feature1']))

# Text Features: TF-IDF
tfidf = TfidfVectorizer(max_features=10)
X_tfidf = tfidf.fit_transform(df['text_column'])

# Feature Selection
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)

# Your transformed dataset is now ready for modeling!
