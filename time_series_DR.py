import numpy as np
import pandas as pd
import sys
import statistics
from sklearn.utils import shuffle
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
from helper import find_similar, find_matches, pca_reconstruct, choose_k, plot_total
import seaborn as sns
import umap
import r_PCA as robust
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, MeanShift, Birch, MiniBatchKMeans

def compare_clusters(old, new, k):

    # See if grouping of countries is different before and after PCA
    # Create two unordered sets and compare grouping of countries
    cluster_grouping_nD = []
    cluster_grouping_HD = []
    
    for cluster_number in range(0, k):
        cluster_grouping_HD.append(set(old[old == cluster_number].index))
        cluster_grouping_nD.append(set(new[new == cluster_number].index))

    cluster_A, cluster_B = find_matches(cluster_grouping_nD, cluster_grouping_HD)

    if(cluster_A == cluster_B):
        results.write("Clusters after PCA to 2 dimensions is the same\n")
    else:
        num_changed = 0
        num_same = 0
        for i in range(k):
            common = set.intersection(cluster_A[i], cluster_B[i])
            changed_countes = set.symmetric_difference(cluster_A[i], cluster_B[i])
            num_changed += len(changed_countes)

            num_same += len(common)

        num_changed /= 2
        results.write("\nCluster membership has changed:\n")
        results.write(str(int(num_changed)) + "/" + str(len(old.index)) + " Counties have changed Clusters.\n\n")

def print_results(df, k, labels, og_data):
    ''' Output Findings '''
    for i in range(k):
        # select only data observations with cluster label == i
        ds = og_df.iloc[np.where(labels==i)]
        first_dates = []
        all_county_cases = []
        total_cases = 0
        max_case = 0
        min_case = 99999999
        # save the data observations
        number_members = len(ds)
        for _, value in ds.iterrows():
            county_cases = sum(value)
            if county_cases > max_case:
                max_case = county_cases
            if county_cases < min_case:
                min_case = county_cases
            all_county_cases.append(county_cases)
            total_cases += county_cases
            for date, cases in value.items():
                if cases != 0:
                    date = datetime.datetime.strptime(date,'%Y-%m-%d')
                    first_dates.append(date)
                    break
        
        # Find Average Date of first case
        total_offset = sum((d - first_dates[0] for d in first_dates), datetime.timedelta(0)) / len(first_dates)
        ave_date = first_dates[0] + total_offset
        
        # Output Metrics + Summary
        results.write("\nCluster: " + str(i) + ": \n" + str(number_members) + " Counties\n")
        results.write("Median Cases by County: " + str(statistics.median(all_county_cases))+"\nMean Cases by County: " + str(statistics.mean(all_county_cases)))
        results.write("\nMinimum total Case Count:" + str(min_case))
        results.write("\nMax total Case Count: " + str(max_case))
        results.write("\nAverage Date of First Case: " + str(ave_date.strftime('%Y-%m-%d')) + "\n\n")

def do_TSNE(df, clustered_data):
    X_2d = TSNE(n_components=2, random_state=5).fit_transform(df)
    covid_df_nd = pd.DataFrame(X_2d)
    covid_df_nd.index = df.index
    tsne_columns = []
    for i in range(1, 3):
        tsne_columns.append('T' + str(i))
    covid_df_nd.columns = tsne_columns
    tsne_2d = covid_df_nd

    color_map = clustered_data['HD_cluster'].map({0:'b', 1: 'r', 2: 'k', 3:'g', 4:'m', 5:'c', 6:'y', 7:'w'})
    
    tsne_plot = tsne_2d.plot(kind='scatter',x='T1',y='T2', c=color_map, figsize=(12,8))
    tsne_plot.set_title(u"TSNE Colored by Original Clusters")
    tsne_plot.tick_params(labelsize=12)
    tsne_plot.set_ylabel('T2', fontsize='12')
    tsne_plot.set_xlabel('T1', fontsize='12')
    plt.savefig('County_TSNE_kmeans_2D_oldclusters.png')
    plt.close()

    # Do kmeans
    clustering = KMeans(n_clusters=4, random_state=5).fit(X_2d)

    # do mini batch k means
    # clustering = MiniBatchKMeans(n_clusters=4, random_state=5).fit(X_2d)

    # spectral cluster
    # clustering = SpectralClustering(n_clusters=4).fit(X_2d)

    # agglomerative clustering
    # clustering = AgglomerativeClustering(n_clusters=4).fit(X_2d)

    # meanshift cluster
    # clustering = MeanShift(bandwidth=4).fit(X_2d)

    # birch
    # clustering = Birch(n_clusters=4).fit(X_2d).predict(X_2d)

    clustered_data['TSNE'] = pd.Series(clustering.labels_, index=df.index)

    color_map = clustered_data['TSNE'].map({0:'b', 1: 'k', 2: 'g', 3:'r', 4:'m', 5:'c', 6:'y', 7:'w'})
    
    tsne_plot = tsne_2d.plot(kind='scatter',x='T1',y='T2', c=color_map, figsize=(12,8))
    tsne_plot.set_title(u"TSNE Colored by New Clusters")
    tsne_plot.tick_params(labelsize=12)
    tsne_plot.set_ylabel('T2', fontsize='12')
    tsne_plot.set_xlabel('T1', fontsize='12')
    plt.savefig('County_TSNE_kmeans_2D_newclusters.png')
    plt.close()

def do_LDA(df, clustered_data):
    # Do LDA for 1, 2, and 3 dimensions 
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(df, clustered_data['HD_cluster'])
    transformed = lda.transform(df)
    covid_df_nd = pd.DataFrame(transformed)
    covid_df_nd.index = df.index
    lda_columns = []
    for i in range(1, 3):
        lda_columns.append('LD' + str(i))
    covid_df_nd.columns = lda_columns
    lda_2d = covid_df_nd
    color_map = clustered_data['HD_cluster'].map({0:'r', 1: 'g', 2: 'b', 3:'k', 4:'m', 5:'c', 6:'y', 7:'w'})
    lda_plot = lda_2d.plot(kind='scatter',x='LD2',y='LD1', c=color_map, figsize=(12,8))
    lda_plot.set_title(u"LDA Colored by Old Clusters")
    plt.savefig('County_LDA_kmeans_2D_oldclusters.png')
    plt.close()

    # new cluster color
    kmeans = KMeans(n_clusters=4, random_state=5)
    clusters = kmeans.fit(lda_2d)
    clustered_data['LDA'] = pd.Series(clusters.labels_, index=df.index)
    color_map = clustered_data['LDA'].map({0:'b', 1: 'k', 2: 'g', 3:'r', 4:'m', 5:'c', 6:'y', 7:'w'})
    lda_plot = lda_2d.plot(kind='scatter',x='LD2',y='LD1', c=color_map, figsize=(12,8))
    lda_plot.set_title(u"LDA Colored by New Clusters")
    plt.savefig('County_LDA_kmeans_2D_newclusters.png')
    plt.close()

def do_UMAP(df, clustered_data):
    # Umap algorithms
    reducer = umap.UMAP(random_state=5)
    embedding = reducer.fit_transform(df)

    color_map = clustered_data['HD_cluster'].map({0:'k', 1: 'b', 2: 'g', 3:'r', 4:'m', 5:'c', 6:'y', 7:'w'})
    plt.figure(figsize=(12, 8))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=color_map)
    #plt.gca().set_aspect('equal', 'datalim')
    plt.tick_params(labelsize=12)
    plt.ylabel('UMAP2', fontsize='12')
    plt.xlabel('UMAP1', fontsize='12')
    plt.title('UMAP Colored by Original Clusters', fontsize=18)
    plt.savefig('County_UMAP_kmeans_2D_oldclusters.png')
    plt.close()

    # new cluster color
    kmeans = KMeans(n_clusters=4, random_state=5)
    clusters = kmeans.fit(embedding)
    clustered_data['UMAP'] = pd.Series(clusters.labels_, index=df.index)
    color_map = clustered_data['UMAP'].map({0:'k', 1: 'r', 2: 'b', 3:'g', 4:'m', 5:'c', 6:'y', 7:'w'})
    plt.figure(figsize=(12, 8))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=color_map)
    plt.tick_params(labelsize=12)
    plt.ylabel('UMAP2', fontsize='12')
    plt.xlabel('UMAP1', fontsize='12')
    #plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP Colored by New Clusters', fontsize=18)
    plt.savefig('County_UMAP_kmeans_2D_newclusters.png')
    plt.close()

def do_PCA(df, clustered_data):
    pca = PCA(n_components=2, random_state=5)
    pca.fit(df)
    covid_nd = pca.transform(df)
    choose_k(covid_nd, "pca2d_kmeans_perf.png")
    cluster = KMeans(n_clusters=4, random_state=5)
    clusters = cluster.fit(covid_nd)
    clustered_data['PCA'] = pd.Series(clusters.labels_, index=df.index)

    covid_df_nd = pd.DataFrame(covid_nd)
    covid_df_nd.index = df.index
    covid_df_nd.columns = ['PC1', 'PC2']

    color_map = clustered_data['PCA'].map({0:'g', 1: 'b', 2: 'k', 3:'r', 4:'m', 5:'c', 6:'y', 7:'w'})
    
    cluster_plot = covid_df_nd.plot(kind='scatter', x='PC1', y='PC2', c=color_map, figsize=(12,8))
    cluster_plot.set_title(u"PCA Colored by New Clusters")
    cluster_plot.tick_params(labelsize=12)
    cluster_plot.set_ylabel('PC2', fontsize='12')
    cluster_plot.set_xlabel('PC1', fontsize='12')
    plt.savefig('County_PCA_kmeans_2D_newclusters.png')
    plt.close()
    color_map = clustered_data['HD_cluster'].map({0:'r', 1: 'g', 2: 'b', 3:'k', 4:'m', 5:'c', 6:'y', 7:'w'})

    original_cluster_plot = covid_df_nd.plot(kind='scatter', x='PC1', y='PC2', c=color_map, figsize=(12,8))
    original_cluster_plot.set_title(u"PCA Colored by Original Clusters")
    cluster_plot.tick_params(labelsize=12)
    cluster_plot.set_xlabel('PC1', fontsize='12')
    cluster_plot.set_ylabel('PC2', fontsize='12')
    plt.savefig('County_PCA_kmeans_2D_oldclusters.png')
    plt.close()

    return

def do_kPCA(df, clustered_data):

    #   do kernel PCA
    kpca = KernelPCA(n_components=2, kernel="poly")
    covid_nd = kpca.fit_transform(df) 

    cluster = KMeans(n_clusters=4, random_state=5)
    clusters = cluster.fit(covid_nd)
    clustered_data['kPCA'] = pd.Series(clusters.labels_, index=df.index)

    covid_df_nd = pd.DataFrame(covid_nd)
    covid_df_nd.index = df.index
    covid_df_nd.columns = ['PC1', 'PC2']

    color_map = clustered_data['kPCA'].map({0:'b', 1: 'k', 2: 'g', 3:'r', 4:'m', 5:'c', 6:'y', 7:'w'})
    
    cluster_plot = covid_df_nd.plot(kind='scatter', x='PC2', y='PC1', c=color_map, figsize=(12,8))
    cluster_plot.set_title(u"kPCA Colored by New Clusters")

    plt.savefig('County_kPCA_kmeans_2D_newclusters.png')
    plt.close()
    color_map = clustered_data['HD_cluster'].map({0:'r', 1: 'g', 2: 'b', 3:'k', 4:'m', 5:'c', 6:'y', 7:'w'})

    original_cluster_plot = covid_df_nd.plot(kind='scatter', x='PC2', y='PC1', c=color_map, figsize=(12,8))
    original_cluster_plot.set_title(u"kPCA Colored by Original Clusters")
    plt.savefig('County_kPCA_kmeans_2D_oldclusters.png')
    plt.close()

    return

def do_MDS(df, clustered_data):
    mds = MDS(n_components=2)
    covid_nd = mds.fit_transform(df)

    choose_k(covid_nd, "mds2d_kmeans_perf.png")
    cluster = KMeans(n_clusters=4, random_state=5)
    clusters = cluster.fit(covid_nd)
    clustered_data['MDS'] = pd.Series(clusters.labels_, index=df.index)

    covid_df_nd = pd.DataFrame(covid_nd)
    covid_df_nd.index = df.index
    covid_df_nd.columns = ['MD1', 'MD2']

    color_map = clustered_data['MDS'].map({0:'b', 1: 'k', 2: 'g', 3:'r', 4:'m', 5:'c', 6:'y', 7:'w'})
    
    cluster_plot = covid_df_nd.plot(kind='scatter', x='MD2', y='MD1', c=color_map, figsize=(12,8))
    cluster_plot.set_title(u"MDS Colored by New Clusters")

    plt.savefig('County_MDS_kmeans_2D_newclusters.png')
    plt.close()
    color_map = clustered_data['HD_cluster'].map({0:'r', 1: 'g', 2: 'b', 3:'k', 4:'m', 5:'c', 6:'y', 7:'w'})

    original_cluster_plot = covid_df_nd.plot(kind='scatter', x='MD2', y='MD1', c=color_map, figsize=(12,8))
    original_cluster_plot.set_title(u"MDS Colored by Original Clusters")
    plt.savefig('County_MDS_kmeans_2D_oldclusters.png')
    plt.close()

    return

results = open("results.txt", "w")

df = pd.read_csv("max_counties_by_date.csv", index_col=0)
og_df = pd.read_csv("counties_by_date.csv", index_col=0)

# Perform clustering before PCA, for comparison
kmeans_performance = []
optimal_k = 4
max_clusters = 8
plt.rc('font', size=12)
HD_clusters = KMeans(n_clusters=4, random_state=5).fit(df)

# print out graph to show kmeans performance for each k in higher dimension
# choose_k(df, "HD_kmeans_perf.png")

clustered_data = pd.DataFrame(index=df.index)
# clustered_data['HD_cluster'] = y = target labels = cluster assignment
clustered_data['HD_cluster'] = pd.Series(HD_clusters.labels_, index=df.index)

# reverse is construct from PCA reduced data
# reverse = pca_reconstruct(df)
# Save Findings
#results.write("Data by County total Counts")
#results.write("\nClustering Data before PCA:\n")
print_results(df, optimal_k, clustered_data['HD_cluster'], og_df)

# Do PCA
print("PCA")
do_PCA(df, clustered_data)
results.write("Comparing PCA Clusters")
compare_clusters(clustered_data['HD_cluster'], clustered_data['PCA'], optimal_k)
print_results(df, 4, clustered_data['PCA'], og_df)
# Do MDS
# do_MDS(df, clustered_data)
# results.write("Comparing MDS Clusters")
# compare_clusters(clustered_data['HD_cluster'], clustered_data['MDS'], optimal_k)

# Do kPCA
# do_kPCA(df, clustered_data)
# results.write("Comparing Kernel PCA Clusters")
# compare_clusters(clustered_data['HD_cluster'], clustered_data['kPCA'], optimal_k)

# Do TSNE for 2 DIMS
print("TSNE")
results.write("Comparing TSNE Clusters")
do_TSNE(df, clustered_data)
print_results(df, 4, clustered_data['TSNE'], og_df)
compare_clusters(clustered_data['HD_cluster'], clustered_data['TSNE'], optimal_k)

# Do LDA for 2 Dimensions
# results.write("Comparing LDA Clusters")
# do_LDA(df, clustered_data)
# compare_clusters(clustered_data['HD_cluster'], clustered_data['LDA'], optimal_k)

# Do UMAP
print("UMAP")
results.write("Comparing UMAP Clusters")
do_UMAP(df, clustered_data)
print_results(df, 4, clustered_data['UMAP'], og_df)
compare_clusters(clustered_data['HD_cluster'], clustered_data['UMAP'], optimal_k)


'''
print(clustered_data)

arr = df.to_numpy()
# Do ROBUST PCA
rpca = robust.R_pca(arr)
L, S = rpca.fit(max_iter=1000, iter_print=100)

# visually inspect results
rpca.plot_fit()
plt.savefig("RobustPCA.png")

compare_clusters(clustered_data['HD_cluster'], df_2d, 2, k_2, clusters_2d)
plot_total(df, min_clusters, max_clusters)
'''
