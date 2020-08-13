import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
# import data
df = pd.read_csv("us-counties.csv")

df.drop(df.loc[df['county']=='Unknown'].index, inplace=True)
df["counties"] = df["county"] + ", " + df["state"]
df.drop(columns=['county', 'state'], axis=0, inplace=True)
dates = df.date.unique()
counties = df.counties.unique()

fips_dict = {}
for _, elt in df.iterrows():
    if math.isnan(elt['fips']):
        # some counties are not valid 
        # ie NYC population is 8,336,817
        fips_dict[elt['counties']] = 000000
    else: 
        fips_dict[elt['counties']] = int(elt['fips'])

# import populatio  data
pops = pd.read_csv("co-est2019-alldata.csv", usecols=['STATE', 'COUNTY', 'POPESTIMATE2019'])
pops['STATE'] = pops['STATE'].astype(str)
pops['COUNTY'] = pops['COUNTY'].astype(str).str.zfill(3)
pops["fips"] = pops["STATE"] + pops["COUNTY"]
pops.drop(columns=['COUNTY', 'STATE'], axis=0, inplace=True)
pops.set_index('fips')

unknown_counties = set()

pop_dict = {}
for _, elt in pops.iterrows():
    pop_dict[elt['fips']] = elt['POPESTIMATE2019']

new_df = pd.DataFrame(index=counties,columns=dates)
i = 0
# create dataframe to normalize by population

pop_df = pd.DataFrame(index=counties, columns=dates)
for _, rows in df.iterrows():
    if str(fips_dict[rows['counties']]) in pop_dict:
        fips = fips_dict[rows['counties']]
        pop = pop_dict[str(fips)]
        pop_df.loc[rows['counties'], rows['date']] = (float(rows['cases'])/float(pop))
    else:
        unknown_counties.add(rows['counties'])
    new_df.loc[rows['counties'], rows['date']] = rows['cases']

    i += 1
    if (i % 1000 == 0):
        print("Done: " + str(i) + " iterations")

new_df = new_df.diff(axis=1).fillna(new_df)
new_df.to_csv("counties_by_date.csv", na_rep=0)

for county in unknown_counties:
   pop_df.drop([county], inplace=True)

pop_df.to_csv("pop_norm_counties_by_date.csv", na_rep=0)

print(unknown_counties)
