from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN

from src.svara import asc_desc

def duration_clustering(sd, eps=0.05, min_samples=1):        
        
    durations = np.array([sd['duration'] for sd in sd])

    # dbscan
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(durations.reshape(-1, 1))
    labels = clustering.labels_
    clusters = []

    clusters = {l:[] for l in set(labels)}
    for i,s in enumerate(sd):
        l = labels[i]
        if l != -1:
            clusters[l].append((i,s))

    clusters = [v for k,v in clusters.items()]

    return clusters


def cadence_clustering(sd, svara):
    clusters = {}
    for i,s in sd:
        start = s['start']
        end = s['end']
        prec = s['preceeding_svara'] if s['preceeding_svara'] else 'silence'
        prec  = prec.replace('(','').replace(')','')
        suc = s['succeeding_svara'] if s['succeeding_svara'] else 'silence'
        suc = suc.replace(')','').replace('(','')

        ad = asc_desc(prec, svara, suc)
        
        if ad not in clusters:
            clusters[ad] = [(i,s)]
        else:
            clusters[ad].append((i,s))

    clusters = [(k,v) for k,v in clusters.items()]
    clusters = sorted(clusters, key=lambda y: ['asc', 'desc', 'cp'].index(y[0]))
    clusters = [x[1] for x in clusters]

    return clusters


def get_inter_intra(df, dcol):
    inter = df[df['cluster1']==df['cluster2']][dcol].mean()
    intra = df[df['cluster1']!=df['cluster2']][dcol].mean()
    return inter/intra


def hier_clustering(sd, distances, t=1, min_in_group=1, linkage_criteria='ward', criterion='inconsistent'):
    
    ix = [i for i,s in sd]
    
    if len(ix)==1:
        if min_in_group > 1:
            return []
        return [sd]
    
    distances = distances[
        (distances['index1'].isin(ix)) & \
        (distances['index2'].isin(ix))]

    dist_piv = distances.pivot("index1", "index2", 'distance').fillna(0)
    indices = dist_piv.columns
    piv_arr = dist_piv.values
    
    X = piv_arr + np.transpose(piv_arr)

    Z = linkage(squareform(X), linkage_criteria)
    
    clustering = fcluster(Z, t=t, criterion=criterion, R=None, monocrit=None)

    count_clust = Counter(clustering)
    good_clust = [k for k,v in count_clust.items() if v>=min_in_group]

    clusters = {g:[] for g in good_clust}
    for i, ix in enumerate(indices):
        cluster = clustering[i]
        if cluster in good_clust:
            s = [(x,y) for x,y in sd if x==ix]
            clusters[cluster] += s

    clusters = [v for k,v in clusters.items()]

    return clusters
