import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# import data
df = pd.read_csv("us-counties.csv")
masks = pd.read_csv("mask-use-by-county.csv")

# create single masks score for each county to multiply case counts
mask_score = {}
for index, elt in masks.iterrows():
    mask_score[index] = elt['RARELY'] + 2*elt['SOMETIMES'] + 3*elt['FREQUENTLY'] + 4*elt['ALWAYS']


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
'''
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

pop_df = pd.DataFrame(index=counties,columns=dates)
'''
new_df = pd.DataFrame(index=counties,columns=dates)
mask_df = pd.DataFrame(index=counties,columns=dates)
unknown_masks = set()
i = 0
for _, rows in df.iterrows():
    fips = fips_dict[rows['counties']]
    ''' for pop df
    if str(fips) in pop_dict:
        pop = pop_dict[str(fips)]
        pop_df.loc[rows['counties'], rows['date']] = (float(rows['cases'])/float(pop))
    else:
        unknown_counties.add(rows['counties'])
        
    '''
    ''' for mask score '''
    if fips in mask_score:
        score = mask_score[fips]
        mask_df.loc[rows['counties'], rows['date']] = float(rows['cases']*score)
    else:
        unknown_masks.add(rows['counties'])

    new_df.loc[rows['counties'], rows['date']] = rows['cases']
    i += 1

    if (i % 1000 == 0):
        print("Done: " + str(i) + " iterations")

new_df = new_df.diff(axis=1).fillna(new_df)
new_df.to_csv("counties_by_date.csv", na_rep=0)

# remove counties with no mask data 
for county in unknown_masks:
    mask_df.drop([county], inplace=True)

mask_df = mask_df.diff(axis=1).fillna(mask_df)
mask_df.to_csv("mask_counties_by_date.csv", na_rep=0)
'''
for county in unknown_counties:
    pop_df.drop([county], inplace=True)
pop_df = pop_df.diff(axis=1).fillna(pop_df)
pop_df.to_csv("pop_norm_counties_by_date.csv", na_rep=0)

print(unknown_counties)
'''