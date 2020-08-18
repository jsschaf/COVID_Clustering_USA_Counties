import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# import data
df = pd.read_csv("counties_by_date.csv", index_col=0)

# Normalize by row sums
# new_df = df.div(df.sum(axis=1), axis=0)#.fillna(0)
# new_df.to_csv("sum_norm_counties_by_date.csv", na_rep=0)

# Normalize by row max
new_df = df.div(df.max(axis=1), axis=0)#.fillna(0)
new_df.to_csv("max_counties_by_date.csv", na_rep=0)


# Remove leading zeros
# no_zeros = pd.DataFrame([np.trim_zeros(i) for i in df.values], index=df.index).fillna(0)
# no_zeros.columns = df.columns[:len(no_zeros.columns)]
# no_zeros.to_csv("no_zeros_counties_by_date.csv", na_rep=0)


# Do Formal Normalization and Standardizqtion
# Make all values between 0 and 1, with unit variance
df = pd.read_csv("max_counties_by_date.csv", index_col=0)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit_transform(df)
norm_df = pd.DataFrame(scaler)
print(norm_df.head())
norm_df.to_csv("max_norm_counties_by_date.csv", na_rep=0)