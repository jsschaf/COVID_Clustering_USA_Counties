import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import data
df = pd.read_csv("us-counties.csv")

df.drop(df.loc[df['county']=='Unknown'].index, inplace=True)
df["counties"] = df["county"] + ", " + df["state"]
df.drop(columns=['county', 'state'], axis=0, inplace=True)
test = df.groupby(["counties", "date"])["cases"].count().apply(lambda x: x).to_dict()
dates = df.date.unique()
counties = df.counties.unique()

new_df = pd.DataFrame(index=counties,columns=dates)


for elt in test.items():
    new_df.loc[elt[0][0],elt[0][1]] = elt[1]

new_df.to_csv("counties_by_date.csv", na_rep=0)
