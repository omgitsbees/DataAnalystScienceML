# Import necessary libraries
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example data: [Age, Income]
X = np.array([[25, 50000], [35, 65000], [45, 80000], [20, 25000], [35, 70000], [52, 95000], [23, 46000], [40, 78000]])
y = np.array([0, 0, 1, 0, 0, 1, 0, 1])  # 0: Will not buy, 1: Will buy

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the Decision Tree classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Visualize the Decision Tree (optional)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=['Age', 'Income'], class_names=['Not Buy', 'Buy'], filled=True)
plt.show()