import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import datashader as ds
import datashader.transfer_functions as tf
from colorcet import fire
import matplotlib.pyplot as plt
from PIL import Image

# Generate some sample data
np.random.seed(42)
n_samples = 300
n_outliers = 20

# Generate normal data
X = 0.3 * np.random.randn(n_samples, 2)
# Generate outlier data
X[:n_outliers] = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))

# Convert to DataFrame for easier manipulation
data = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

# Initialize the Isolation Forest model
model = IsolationForest(contamination=float(n_outliers) / n_samples, random_state=42)
model.fit(data)

# Predict anomalies
data['Anomaly'] = model.predict(data)

# Separate normal data and anomalies
normal_data = data[data['Anomaly'] == 1]
anomaly_data = data[data['Anomaly'] == -1]

# Create a datashader Canvas
canvas = ds.Canvas(plot_width=800, plot_height=800)

# Aggregate normal data into hex bins
agg_normal = canvas.points(normal_data, 'Feature1', 'Feature2', agg=ds.count())

# Aggregate anomaly data into hex bins
agg_anomaly = canvas.points(anomaly_data, 'Feature1', 'Feature2', agg=ds.count())

# Create color maps
normal_img = tf.shade(agg_normal, cmap=fire, min_alpha=40)
anomaly_img = tf.shade(agg_anomaly, cmap=["red"], min_alpha=40)

# Convert the datashader images to PIL images
normal_pil = tf.set_background(normal_img, 'white').to_pil()
anomaly_pil = tf.set_background(anomaly_img, None).to_pil()

# Combine the images using PIL
combined_img = Image.alpha_composite(normal_pil.convert('RGBA'), anomaly_pil.convert('RGBA'))

# Plot using matplotlib
plt.imshow(combined_img)
plt.axis('off')
plt.show()

# Print the anomalies
print("Anomalies found:")
print(anomaly_data)
