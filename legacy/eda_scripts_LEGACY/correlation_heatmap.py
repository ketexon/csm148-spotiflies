# %pip install pandas numpy matplotlib seaborn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'cleaned_spotify.csv')

# Select only numeric columns + create correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
df_corr = numeric_df.corr()

# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))  # Optional: Adjust the figure size
sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)

# Show the plot
plt.show()
