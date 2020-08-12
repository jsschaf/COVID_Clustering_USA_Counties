from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
import sys
import itertools
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def choose_k(df, filename):
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=5)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.savefig(filename)
    plt.close()

def plot_total(df, min_clusters, max_clusters):
    # Plot and cluster total cases for each county
    county_counts = pd.DataFrame(index=df.index)
    county_counts["cases"] = df.sum(axis=1)
    total_clusters = KMeans(n_cluster=4, random_state=5).fit(county_counts)
    county_counts['y'] = 0
    clustered_total_data = county_counts.copy()
    clustered_total_data['cluster'] = pd.Series(total_clusters.labels_, index=county_counts.index)
    color_map = clustered_total_data.cluster.map({0:'r', 1: 'g', 2: 'b', 3:'k', 4:'m', 5:'c', 6:'y', 7:'w'})
    total_plot = county_counts.plot(kind='scatter',x='cases',y='y', c=color_map, figsize=(16,2))
    total_plot.set_title(u"1D Scatter Plot of Total Cases")
    total_plot.set_xlabel('Total Cases')
    plt.savefig('Total_Cases_Scatter.png')
    plt.close()

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

def pca_reconstruct(df):
    print("Original Shape")
    print(df.shape)
    pca = PCA(n_components=2)
    X_proj = pca.fit_transform(df)
    # find out hwo much variance is explained by these components
    print("Explained Variance")
    print(np.sum(pca.explained_variance_ratio_))
    
    print("New Reduced Shape")
    print(X_proj.shape)

    reverse = pd.DataFrame(pca.inverse_transform(X_proj), index=df.index)
    print("Reverse Shape")
    print(reverse.shape)

    print("Reversed Head")
    print(reverse.head())

    return reverse