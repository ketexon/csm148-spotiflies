import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate some data
df = pd.read_csv('../csv_outputs/cleaned_spotify.csv')

df_corr = df.corr()
print(df_corr)

df_corr.to_csv('../csv_outputs/correlation_matrix.csv')
