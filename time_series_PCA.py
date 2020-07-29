import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
import sys
import itertools

def find_similar(setA, list_of_sets):
    current_max = 0
    best_elt = set()
    for elt in list_of_sets:
        common_no = len(set.intersection(setA, elt))
        total_no = len(set.union(setA, elt))
        proportion_same = float(common_no)/float(total_no)
        if proportion_same >= current_max:
            current_max = proportion_same
            best_elt = elt
    list_of_sets.remove(best_elt)
    return best_elt, current_max

def find_matches(listA, listB):
    bestmatchA = []
    bestmatchB = []
    best_sim = 0
    for perm in itertools.permutations(listA):
        total_similarity = 0
        matchA = []
        matchB = []
        listA.sort(key=len, reverse=True)
        comparisons = list(listB)
        for elt in perm:
            match, similarity = find_similar(elt, comparisons)
            total_similarity += similarity
            if(match in matchB):
                print("Error: duplicate clusters have been compared.")
                sys.exit()
            matchA.append(elt)
            matchB.append(match)

        if total_similarity > best_sim:
            best_sim = total_similarity
            bestmatchA = matchA
            bestmatchB = matchB
    return bestmatchA, bestmatchB

def cluster(min_clusters, max_clusters, df):
    
    ''' For K-means '''
    performance = []

    for k in range(min_clusters, max_clusters):
        # making random_state an int will make kmeans determinstic
        kmeans = KMeans(n_clusters=k, random_state=5)
        clusters = kmeans.fit(df)
        performance.append(kmeans.inertia_)
    
    # add min_clusters to offset 0-indexing
    optimal_k = min_clusters + performance.index(max(performance))

    best = KMeans(n_clusters=optimal_k, random_state=5)

    # re-create optimal kmeans clustering and plot
    ''' For Spectral Clustering 
    optimal_k = min_clusters
    best = SpectralClustering(n_clusters=optimal_k, random_state=5)
    '''
    clusters = best.fit(df)

    return clusters, optimal_k

def compare_clusters(old, covid_df_nd, dimensionality, k, clusters_nd):

    clustered_nd_data = covid_df_nd.copy()
    clustered_nd_data['cluster'] = pd.Series(clusters_nd.labels_, index=covid_df_nd.index)

    # Compare These Clusters to original higher dimensional kmeans clustering
    clustered_data[str(dimensionality) + 'D_cluster'] = pd.Series(clusters_nd.labels_, index=covid_df_nd.index)

    # See if grouping of countries is different before and after PCA
    # Create two unordered sets and compare grouping of countries
    cluster_grouping_nD = []
    cluster_grouping_HD = []
    
    for cluster_number in range(0, k):
        cluster_grouping_nD.append(set(clustered_data[clustered_data[str(dimensionality) + 'D_cluster'] == cluster_number].index))
        cluster_grouping_HD.append(set(clustered_data[clustered_data['HD_cluster'] == cluster_number].index))
    
    cluster_A, cluster_B = find_matches(cluster_grouping_nD, cluster_grouping_HD)
    
    if(cluster_A == cluster_B):
        results.write("Clusters after PCA to " + str(dimensionality) + " dimensions is the same\n")
    else:
        unchanged_counties = set()
        for i in range(k):
            common = set.intersection(cluster_A[i], cluster_B[i])
            unchanged_counties = set.union(unchanged_counties,common)

        results.write("PCA to " + str(dimensionality) + " has changed cluster membership\n")
        results.write(str(len(unchanged_counties)) + "/" + str(len(old.index)) + " Counties have remained in same Clusters.\n")

    # Map clusters to colors instead of grayscale
    color_map = clustered_nd_data.cluster.map({0:'r', 1: 'g', 2: 'b', 3:'k', 4:'m', 5:'c', 6:'y', 7:'w'})
    
    if dimensionality == 1:
        clustered_nd_data['PC2'] = 0
        cluster_plot = clustered_nd_data.plot(kind='scatter',x='PC1',y='PC2', c=color_map, figsize=(16,2))
        cluster_plot.set_title(u"1D Scatter Plot")
        plt.savefig('County_PCA_kmeans_1D.png')
        plt.close()
    elif dimensionality == 2:
        cluster_plot = clustered_nd_data.plot(kind='scatter',x='PC2',y='PC1', c=color_map, figsize=(16,8))
        cluster_plot.set_title(u"2D Scatter Plot")
        #for i, county in enumerate(covid_df_nd.index):
        #    cluster_plot.annotate(county, (covid_df_nd.iloc[i].PC2, covid_df_nd.iloc[i].PC1))
        plt.savefig('County_PCA_kmeans_2D.png')
        plt.close()
    elif(dimensionality) == 3:  
        # 3D plotting
        fig = plt.figure(figsize=(24,12))
        ax3 = fig.gca(projection='3d')
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        ax3.set_zlabel('PC3')
        ax3.set_title(u"3D Scatter Plot")
        colors = {0:'r', 1: 'g', 2: 'b', 3:'k', 4:'m', 5:'c', 6:'y', 7:'w'}

        for i, county in enumerate(df.index):
            col = colors[int(clustered_data.iloc[i]['3D_cluster'])]
            x, y, z = df_3d.iloc[i]['PC1'], df_3d.iloc[i]['PC2'], df_3d.iloc[i]['PC3']
            ax3.scatter(x, y, z, c=col)
            # to label each point
            # ax3.text(x, y, z, '{0}'.format(df_3d.index[i]))
        plt.savefig('County_PCA_kmeans_3D.png')
        plt.close()

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
    loadings = pd.DataFrame(pca.components_.T, index=covid_df.columns)

    return covid_df_nd


def print_results(df, k, labels):
    ''' Output Findings '''
    for i in range(k):
        # select only data observations with cluster label == i
        ds = df.iloc[np.where(labels==i)]
        HD_first_dates = []
        HD_max_cases = 0
        HD_total_cases = 0
        # save the data observations
        HD_number_members = len(ds)
        for county, value in ds.iterrows():
            county_cases = sum(value)
            if county_cases > HD_max_cases:
                HD_max_cases = county_cases
            HD_total_cases += county_cases
            for date, cases in value.items():
                if cases != 0:
                    date = datetime.datetime.strptime(date,'%Y-%m-%d')
                    HD_first_dates.append(date)
                    break
        
        # Find Average Date of first case
        total_offset = sum((d - HD_first_dates[0] for d in HD_first_dates), datetime.timedelta(0)) / len(HD_first_dates)
        ave_date = HD_first_dates[0] + total_offset
        
        # Output Metrics + Summary
        results.write("\nCluster: " + str(i) + ": \n" + str(HD_number_members) + " Counties\n")
        results.write("Max Cases for single County: " + str(HD_max_cases)+"\nTotal Cases: " + str(HD_total_cases))
        results.write("\nAverage Date of First Case: " + str(ave_date.strftime('%Y-%m-%d')) + "\n\n")

results = open("Results.txt", "w")

df = pd.read_csv("counties_by_date.csv", index_col=0)
# df = pd.read_csv("test_case.csv", index_col=0)
# Randomize rows - makes PCA non-deterministic
# df = shuffle(df)

# Perform clustering before PCA, for comparison
kmeans_performance = []
min_clusters = 4
max_clusters = 8

HD_clusters, optimal_k = cluster(min_clusters, max_clusters, df)

clustered_data = pd.DataFrame(index=df.index)

clustered_data['HD_cluster'] = pd.Series(HD_clusters.labels_, index=df.index)

# Do PCA for 1, 2 and 3 dimensions
df_1d = do_PCA(df, 1)
df_2d = do_PCA(df, 2)
df_3d = do_PCA(df, 3)
        
clusters_2d, k_2 = cluster(min_clusters, max_clusters, df_2d)
clusters_3d, k_3 = cluster(min_clusters, max_clusters, df_3d)
clusters_1d, k_1 = cluster(min_clusters, max_clusters, df_1d)

if(k_2 != k_3 != k_1 != optimal_k):
    results.write("Different no. of clusters formed before/after DR\nk before: " + str(optimal_k) + " k after: " + str(k_2))
else:
    # Compare CLustering before and after PCA for data with dimensions: 1, 2, 3 
    compare_clusters(clustered_data['HD_cluster'], df_1d, 1, k_1, clusters_1d)
    compare_clusters(clustered_data['HD_cluster'], df_2d, 2, k_2, clusters_2d)
    compare_clusters(clustered_data['HD_cluster'], df_3d, 3, k_3, clusters_3d)

# Plot and cluster total cases for each county
county_counts = pd.DataFrame(index=df.index)
county_counts["cases"] = df.sum(axis=1)
total_clusters, k = cluster(min_clusters, max_clusters, county_counts)
county_counts['y'] = 0
clustered_total_data = county_counts.copy()
clustered_total_data['cluster'] = pd.Series(total_clusters.labels_, index=county_counts.index)
color_map = clustered_total_data.cluster.map({0:'r', 1: 'g', 2: 'b', 3:'k', 4:'m', 5:'c', 6:'y', 7:'w'})
total_plot = county_counts.plot(kind='scatter',x='cases',y='y', c=color_map, figsize=(16,2))
total_plot.set_title(u"1D Scatter Plot of Total Cases")
total_plot.set_xlabel('Total Cases')
plt.savefig('Total_Cases_Scatter.png')
plt.close()


# Save Findings
results.write("Data by County total Counts")
results.write("\nClustering Data before PCA:\n")
print_results(df, optimal_k, HD_clusters.labels_)
results.write("\nClustering Data after PCA to 1D:\n")
print_results(df, optimal_k, clusters_1d.labels_)
results.write("\nClustering Data after PCA to 2D:\n")
print_results(df, optimal_k, clusters_2d.labels_)
results.write("\nClustering Data after PCA to 3D:\n")
print_results(df, optimal_k, clusters_3d.labels_)
