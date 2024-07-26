import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate synthetic data
np.random.seed(0)
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]] # diagonal coveriance
x = np.random.multivariate_normal(mean, cov, 100)

# Standarize the data
X_mean = np.mean(x, axis=0)
X_std = np.std(x, axis=0)
X_standardized = (x - X_mean) / X_std

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standardized)

# Plot original data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(x[:, 0], x[:, 1], alpha=0.5)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot PCA-transformed data
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, color='r')
plt.title('PCA Transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()