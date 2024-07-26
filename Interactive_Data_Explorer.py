import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import ipywidgets as widgets 
from IPython.display import display 

# Load dataset 
df = sns.load_dataset('titanic')

# Create widgets
column_selector = widgets.Dropdown( 
    options=df.columns,
    description='column:'
)

age_slider = widgets.IntRangeSlider(
    value=[0, 80],
    min=0,
    max=80,
    step=1,
    description='Age range:'
)

# Update function
def update_plot(column, age_range):
    filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
    plt.figure(figsize=(10, 6))
    sns.countplot(data=filtered_df, x=column)
    plt.title(f'Distribution of {column}')
    plt.show()

# Link widgets to function
interactive_plot = widgets.interactive(update_plot, column=column_selector, age_range=age_slider)

# Display Widgets
display(column_selector, age_slider)
display(interactive_plot)