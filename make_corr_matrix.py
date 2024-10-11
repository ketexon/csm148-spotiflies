import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate some data
df = pd.read_csv(r'dataset.csv')

df_corr = df.corr()
print(df_corr)


df_corr.to_csv('correlation matrix.csv')
