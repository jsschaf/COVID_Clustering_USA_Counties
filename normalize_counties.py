import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
# import data
df = pd.read_csv("pop_norm_counties_by_date.csv", index_col=0)

# Normalize by row sums
# new_df = df.div(df.sum(axis=1), axis=0)#.fillna(0)
# new_df.to_csv("sum_norm_counties_by_date.csv", na_rep=0)

# Normalize by row max
new_df = df.div(df.max(axis=1), axis=0)#.fillna(0)
new_df.to_csv("pop_maxnorm_counties_by_date.csv", na_rep=0)

# Remove leading zeros
# no_zeros = pd.DataFrame([np.trim_zeros(i) for i in df.values], index=df.index).fillna(0)
# no_zeros.columns = df.columns[:len(no_zeros.columns)]
# no_zeros.to_csv("no_zeros_counties_by_date.csv", na_rep=0)
