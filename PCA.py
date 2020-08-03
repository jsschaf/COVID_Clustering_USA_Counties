from sklearn.decomposition import PCA
import pandas as pd

def do_PCA(covid_df, n):
    
    pca = PCA(n_components=n)
    pca.fit(covid_df)
    covid_nd = pca.transform(covid_df)
    covid_df_nd = pd.DataFrame(covid_nd)
    covid_df_nd.index = covid_df.index
    pca_columns = []
    
    for i in range(1, n+1):
        pca_columns.append('PC' + str(i))
    
    covid_df_nd.columns = pca_columns
    
    # Allows us to find contribution of each feature to PCA, if desired
    # loadings = pd.DataFrame(pca.components_.T, index=covid_df.columns)

    return covid_df_nd