import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("wine.data")
print(df)


x = df.values.astype(float)

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df_normalized = pd.DataFrame(x_scaled)
print(df_normalized)
