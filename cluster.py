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