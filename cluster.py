from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
import sys
import itertools
import pandas as pd

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
        # todo: use elbow method with inertia
        # todo: investigate adjusted rand index
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