import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalAnalysisTool:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
    
    def data_overview(self):
        print("Data Overview:")
        print(self.data.info())
        print("\nFirst 5 rows of the dataset:")
        print(self.data.head())
    
    def basic_statistics(self):
        print("Basic Statistics:")
        print(self.data.describe())
    
    def data_visualization(self):
        # Histograms
        self.data.hist(figsize=(10, 10))
        plt.show()
        
        # Box plots
        self.data.plot(kind='box', subplots=True, layout=(int(np.ceil(len(self.data.columns)/3)), 3), figsize=(12, 8))
        plt.show()
        
        # Scatter plot matrix
        sns.pairplot(self.data)
        plt.show()
    
    def correlation_analysis(self):
        print("Correlation Matrix:")
        corr_matrix = self.data.corr()
        print(corr_matrix)
        
        # Heatmap of correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.show()

# Usage
file_path = 'path_to_your_dataset.csv'  # Replace with the path to your dataset
tool = StatisticalAnalysisTool(file_path)
tool.data_overview()
tool.basic_statistics()
tool.data_visualization()
tool.correlation_analysis()
