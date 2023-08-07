%load_ext autoreload
%autoreload 2

from collections import Counter
import compiam
import os 
import librosa
import mirdata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from numpy.linalg import norm
from numpy import dot

from src.dtw import dtw
from src.tools import compute_novelty_spectrum, get_loudness, interpolate_below_length, get_derivative
from src.utils import load_audacity_annotations, cpath, write_pkl, write_pitch_track, load_pitch_track, read_txt, load_pkl
from src.visualisation import get_plot_kwargs, plot_subsequence
from src.pitch import pitch_seq_to_cents
from src.svara import get_svara_dict, get_unique_svaras, pairwise_distances_to_file
from scipy.signal import savgol_filter

track = '191_Mati_Matiki'
raga = 'mohanam'

data_home = '/Volumes/MyPassport/mir_datasets2/saraga1.5_carnatic/'
saraga = mirdata.initialize('saraga_carnatic', data_home=data_home)

#carnatic = compiam.load_dataset('saraga_carnatic')
tracks = saraga.load_tracks()


###########
# Get paths
###########
mridangam_left_path = tracks[track].audio_mridangam_left_path
mridangam_right_path = tracks[track].audio_mridangam_right_path
vocal_path = tracks[track].audio_vocal_path
tonic_path = tracks[track].ctonic_path
sections_path = tracks[track].sections_path
tempo_path = tracks[track].tempo_path
annotations_path = os.path.join('data', 'annotation', f'{track}.txt')



###########
# Load Data
###########
sr = 44100
mridangam_left, _ = librosa.load(mridangam_left_path, sr=sr)
mridangam_right, _ = librosa.load(mridangam_right_path, sr=sr)
vocal, _ = librosa.load(vocal_path, sr=sr)

sections = pd.read_csv(sections_path, sep='\t', header=None)
sections.columns = ['start', 'group', 'duration', 'section']
del sections['group']

tempo = read_txt(tempo_path) #tempo, matra_interval, sama_interval, matras_per_cycle, start_time, end_time
tonic = read_txt(tonic_path)
tonic = float(tonic[0].replace('\n',''))
plot_kwargs = get_plot_kwargs(raga, tonic)
yticks_dict = plot_kwargs['yticks_dict']

# Load annotations
annotations = load_audacity_annotations(annotations_path)

##################
# Extract Features
##################
# Pitch track -> change points
ftanet_carnatic = compiam.load_model("melody:ftanet-carnatic")
pitch_track_path = cpath('data', 'pitch_tracks', f'{track}.tsv')
#ftanet_pitch_track = ftanet_carnatic.predict(vocal_path,hop_size=30)
#write_pitch_track(ftanet_pitch_track, pitch_track_path, sep='\t')
ftanet_pitch_track = load_pitch_track(pitch_track_path)


pitch = ftanet_pitch_track[:,1]
time = ftanet_pitch_track[:,0]
timestep = time[3]-time[2]

pitch = interpolate_below_length(pitch, 0, (250*0.001/timestep))
null_ind = pitch==0

pitch[pitch<50]=0
pitch[null_ind]=0

pitch_cents = pitch_seq_to_cents(pitch, tonic)


#####################
# Get Svaras Features
#####################
svara_dict_path = cpath('data', 'svara_dict', f'{track}.csv')
unique_svaras = get_unique_svaras(annotations)
svara_dict = get_svara_dict(annotations, pitch_cents, timestep, track, path=None)

#####
# DTW
#####
distance_paths = {}
for svara in unique_svaras:
    print(f'Computing distances for {svara}')
    all_svaras = svara_dict[svara]
    all_ix = list(range(len(all_svaras)))
    dtw_distances_path = cpath('data', 'dtw_distances', f'{track}', f'{svara}.csv')
    pairwise_distances_to_file(all_ix, all_svaras, dtw_distances_path)
    distance_paths[svara] = dtw_distances_path

#####
# DTW
#####
#       dtw_diff_distances_path = cpath('data', 'dtw_diff_distances', 'svara', f'{track}', f'{svara}.csv')
#       try:
#           print('Removing previous distances file')
#           os.remove(dtw_diff_distances_path)
#       except OSError:
#           pass

#       # i=23
#       # j=1
#       # s1 = all_svaras[i]['pitch']
#       # s2 = all_svaras[j]['pitch']

#       # pi = len(s1)
#       # pj = len(s2)
#       # l_longest = max([pi, pj])
#       # radius = round(l_longest*0.1)

#       # from src.dtw import *

#       ##text=List of strings to be written to file
#       header = 'index1,index2,dtw'
#       with open(dtw_diff_distances_path,'a') as file:
#           file.write(header)
#           file.write('\n')
#           for i in tqdm.tqdm(all_ix):
#               for j in all_ix:
#                   print(f'i={i}, j={j}')
#                   if i <= j:
#                       continue
#                   pat1 = all_svaras[i]['pitch']
#                   pat2 = all_svaras[j]['pitch']
#                   
#                   pat1,_ = get_derivative(pat1, pat1)
#                   pat2,_ = get_derivative(pat2, pat2)

#                   pi = len(pat1)
#                   pj = len(pat2)
#                   l_longest = max([pi, pj])

#                   path, dtw_val = dtw(pat1, pat2, radius=round(l_longest*0.05))
#                   l = len(path)
#                   dtw_norm = dtw_val/l

#                   row = f'{i},{j},{dtw_norm}'
#                   
#                   file.write(row)
#                   file.write('\n')

#       ########
#       # COSINE
#       ########
#       cosine_distances_path = cpath('data', 'cosine_distances', 'svara', f'{track}', f'{svara}.csv')
#       try:
#           print('Removing previous distances file')
#           os.remove(cosine_distances_path)
#       except OSError:
#           pass

#       # i=23
#       # j=1
#       # s1 = all_svaras[i]['pitch']
#       # s2 = all_svaras[j]['pitch']

#       # pi = len(s1)
#       # pj = len(s2)
#       # l_longest = max([pi, pj])
#       # radius = round(l_longest*0.1)

#       # from src.dtw import *

#       def get_features(pitch, timestep, window=0.05):
#           
#           window = round(window*len(pitch))

#           if window < 1:
#               window = 1

#           features = {
#               'max_freq': np.max(pitch),
#               'min_freq': np.min(pitch),
#               'std_freq': np.std(pitch),
#               'mean_freq': np.mean(pitch),
#               'start_freq': np.mean(pitch[:window]),
#               'end_freq': np.mean(pitch[-window:])
#           }

#           return features

#       cosine_similarity = lambda a,b: dot(a, b)/(norm(a)*norm(b))

#       ##text=List of strings to be written to file
#       header = 'index1,index2,cosine'
#       with open(cosine_distances_path,'a') as file:
#           file.write(header)
#           file.write('\n')
#           for i in tqdm.tqdm(all_ix):
#               for j in all_ix:
#                   print(f'i={i}, j={j}')
#                   if i <= j:
#                       continue
#                   pat1 = all_svaras[i]['pitch']
#                   pat2 = all_svaras[j]['pitch']
#                   
#                   pat1_feat = np.array(list(get_features(pat1, timestep).values()))
#                   pat2_feat = np.array(list(get_features(pat2, timestep).values()))

#                   cos = 1-cosine_similarity(pat1_feat, pat2_feat)

#                   row = f'{i},{j},{cos}'
#                   
#                   file.write(row)
#                   file.write('\n')


#############
## Clustering
#############
t = 1
min_in_group = 3
cluster_dict = {}

def get_inter_intra(df, dcol):
    inter = df[df['cluster1']==df['cluster2']][dcol].mean()
    intra = df[df['cluster1']!=df['cluster2']][dcol].mean()
    return inter/intra

for svara, distance_path in distance_paths.items():
    print(f'Clustering {svara}')
    all_svaras = svara_dict[svara]
    distances = pd.read_csv(distance_path) # cosine_distances_path, dtw_diff_distances_path, dtw_distances_path

    distances_flip = distances.copy()
    distances_flip.columns = ['index2', 'index1', 'distance']
    distances = pd.concat([distances, distances_flip])

    dist_piv = distances.pivot("index1", "index2", 'distance').fillna(0)
    indices = dist_piv.columns

    piv_arr = dist_piv.values
    X = piv_arr + np.transpose(piv_arr)

    Z = linkage(squareform(X), 'ward')

    #metrics = []
    #all_ts = list(np.arange(0, 2, 0.01))
    #for t in tqdm.tqdm(all_ts):
    #    clustering = fcluster(Z, t=t, criterion='inconsistent')

    #    clusterer = {x:y for x,y in zip(indices, clustering)}

    #    distances['cluster1'] = distances['index1'].apply(lambda y: clusterer[y])
    #    distances['cluster2'] = distances['index2'].apply(lambda y: clusterer[y])

    #    metrics.append(get_inter_intra(distances, 'distance'))

    #plot_path = cpath('plots', f'{track}', 'clustering', f'deriv_hierarchical_svara_clustering_{svara}.png')

    #plt.plot(all_ts, metrics)
    #plt.title('Varying t')
    #plt.xlabel('t')
    #plt.ylabel('Ratio of inter:intra cluster distance')
    #plt.savefig(plot_path)
    #plt.close('all')

    clustering = fcluster(Z, t=t, criterion='inconsistent', R=None, monocrit=None)

    clus_dir = cpath('plots', str(track), 'clustering', svara)

    try:
        import shutil
        shutil.rmtree(clus_dir)
    except:
        pass

    count_clust = Counter(clustering)
    good_clust = [k for k,v in count_clust.items() if v>min_in_group]

    star_clusters = {g:[] for g in good_clust}
    for i,ix in tqdm.tqdm(list(enumerate(indices))):
        pt = all_svaras[ix]['pitch']
        times = [x*timestep for x in range(len(pt))]
        cluster = clustering[i]
        if cluster in good_clust:
            star_clusters[cluster].append((times, pt))
            path = cpath(clus_dir, str(cluster), f'{i}.png')
            #plt.plot(times, pt)
            #plt.savefig(path)
            #plt.close('all')

    star_clusters_reduced = {}

    for c,tracks in star_clusters.items():
        all_pitch = [p for t,p in tracks]
        star_clusters_reduced[c] = all_pitch[0]

    cluster_dict[svara] = star_clusters_reduced

########################
## Get Distance profiles
########################
wl = round(145*0.001/timestep)
wl = wl if not wl%2 == 0 else wl+1
distance_profiles = {}

for svara, star_clusters_reduced in cluster_dict.items():
    print(f'Svara: {svara}')
    distance_profiles[svara] = {c:[] for c,_ in star_clusters_reduced.items()}
    for c, unit in list(star_clusters_reduced.items()):
        print(f'Cluster {c}')
        for i in tqdm.tqdm(list(range(len(pitch_cents)))):
            target = pitch_cents[i:i+len(unit)]
            if len(target) <= wl:
                break
            target = savgol_filter(target, polyorder=2, window_length=wl, mode='interp')

            if np.isnan(target).any():
                distance_profiles[svara][c].append(np.Inf)
                continue

            pi = len(target)
            pj = len(unit)
            l_longest = max([pi, pj])

            path, dtw_val = dtw(target, unit, radius=round(l_longest*0.05))
            l = len(path)
            dtw_norm = dtw_val/l
            distance_profiles[svara][c].append(dtw_norm)

dp_path = cpath('data', 'distance_profiles', f'{track}.pkl')
write_pkl(distance_profiles, dp_path)

distance_profiles = load_pkl(dp_path)

###########
## Annotate
###########
annot_dist_thresh = 15

occurences = []
distances = []
lengths = []
labels = []

for s,ap in distance_profiles.items():
    print(f'Svara: {s}')
    for i in ap:
        dp = np.array(ap[i]).copy()

        max_dist = 0
        while max_dist < annot_dist_thresh:
            
            l = len(cluster_dict[s][i])
            
            ix = dp.argmin()
            dist = dp[ix]
            
            dp[max(0,int(ix-l)):min(len(dp),int(ix+l))] = np.Inf

            distances.append(dist)
            lengths.append(l)
            occurences.append(ix)
            labels.append(s)

            max_dist = dist



#   for i,o in tqdm.tqdm(list(enumerate(occurences))):
#       p = pitch_cents[o:o+len(unit)]
#       p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')
#       t = [(o+x)*timestep for x in range(len(p))]
#       plt.plot(t,p)
#       plt.xlabel('Time (s)')
#       plt.ylabel('Pitch (cents)')
#       path = cpath('plots', f'{track}', 'annotations', 'dha', f'{i}.png')
#       plt.savefig(path)
#       plt.close('all')

#distance_profile = stumpy.mass(unit, pitch_cents, normalize=False)


#########
## Export
#########
starts = [o*timestep for o in occurences]
ends   = [starts[i]+(lengths[i]*timestep) for i in range(len(starts))]
transcription = pd.DataFrame({
        'start':starts,
        'end':ends,
        'label':labels,
    }).sort_values(by='start')

trans_path = cpath('data', 'transcription', f'{track}.txt')
transcription.to_csv(trans_path, index=False, header=False, sep='\t')

#  - Clean up
#      - dtw distances between them all to trim
#      - remove silences
#  - Characterize svaras
#      - cluster into profiles based on features
#      - what are the distinguishing characteristics
#      - Markov chains
#  - Query track with MP style approach to identify other occurrences
#  - Clean up annotations with raga grammar
#  - Clean annotations with onsets from spectral flux?
#  - Which characteristics are due to the pronunciation and which due to the melody?
#  - Evaluate somehow?
