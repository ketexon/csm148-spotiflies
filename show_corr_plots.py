import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate some data
df = pd.read_csv(r'dataset.csv')

df_corr = df.corr()
print(df_corr)

columns = df.columns

for i in columns:
    for j in columns:
        if '64' in str(df[i].dtype) and '64' in str(df[j].dtype) and i != j and 'Unnamed' not in i and 'Unnamed' not in j:
            plt.plot(np.unique(df[i]), np.poly1d(np.polyfit(df[i], df[j], 1))(np.unique(df[i])), color='red')
            plt.scatter(df[i], df[j], s=5)
            plt.xlabel(i)
            plt.ylabel(j)
            plt.show()


#df_corr.to_csv('correlation matrix.csv')