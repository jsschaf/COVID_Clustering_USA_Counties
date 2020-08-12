import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import data
df = pd.read_csv("us-counties.csv")

df.drop(df.loc[df['county']=='Unknown'].index, inplace=True)
df["counties"] = df["county"] + ", " + df["state"]
df.drop(columns=['county', 'state'], axis=0, inplace=True)
test = df.groupby(["counties", "date"])["deaths"].sum().apply(lambda x: x).to_dict()
dates = df.date.unique()
counties = df.counties.unique()

new_df = pd.DataFrame(index=counties,columns=dates)
i = 0
print("Num Total Iterations:")
print(len(test.items()))

for _, rows in df.iterrows():
    new_df.loc[rows['counties'], rows['date']] = rows['cases']
    i += 1
    if (i % 1000 == 0):
        print("Done: " + str(i) + " iterations")

new_df = new_df.diff(axis=1).fillna(new_df)
new_df.to_csv("counties_by_datev2.csv", na_rep=0)
