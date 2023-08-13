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
import random
import tqdm

from numpy.linalg import norm
from numpy import dot

from src.dtw import dtw
from src.tools import compute_novelty_spectrum, get_loudness, interpolate_below_length, get_derivative
from src.utils import load_audacity_annotations, cpath,  write_pkl, write_pitch_track, load_pitch_track, read_txt, load_pkl
from src.visualisation import get_plot_kwargs, plot_subsequence
from src.pitch import pitch_seq_to_cents
from src.svara import get_svara_dict, get_unique_svaras, pairwise_distances_to_file
from scipy.signal import savgol_filter
from src.clustering import duration_clustering, cadence_clustering, hier_clustering

out_dir = cpath('data', 'short_test')

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
svara_dict_path = cpath(out_dir, 'data', 'svara_dict', f'{track}.csv')
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
    dtw_distances_path = cpath(out_dir, 'data', 'dtw_distances', f'{track}', f'{svara}.csv')
    pairwise_distances_to_file(all_ix, all_svaras, dtw_distances_path, mean_norm=True)
    distance_paths[svara] = dtw_distances_path


#############
## Clustering
#############
wl = round(145*0.001/timestep)
wl = wl if not wl%2 == 0 else wl+1

min_samples = 1 # duration min samp
eps = 0.05 # duration epsilon

t = 1 # hierarchical clustering t
min_in_group = 1 # min in group for hierarchical

plot = False # plot final clusters?

cluster_dict = {}
for svara, sd in svara_dict.items():
    print(f'Duration clustering, {svara}')
    cluster_dict[svara] = duration_clustering(sd, eps=0.05)
    print(f'    {len(cluster_dict[svara])} clusters')


for svara, clusters in cluster_dict.items():
    print(f'Cadence clustering, {svara}')
    new_clusters = []
    for cluster in clusters:
        cadclust = cadence_clustering(cluster, svara)
        new_clusters += cadclust
    cluster_dict[svara] = new_clusters
    print(f'    {len(cluster_dict[svara])} clusters')


for svara, clusters in cluster_dict.items():
    print(f'Hierarchical clustering, {svara}')
    distance_path = distance_paths[svara]
    distances = pd.read_csv(distance_path)
    distances_flip = distances.copy()
    distances_flip.columns = ['index2', 'index1', 'distance']
    distances = pd.concat([distances, distances_flip])
    new_clusters = []
    for cluster in clusters:
        hier = hier_clustering(cluster, distances, t=t, min_in_group=min_in_group)
        if hier: # hier can return empty array if min_in_group > 1
            new_clusters += hier
    cluster_dict[svara] = new_clusters
    print(f'    {len(cluster_dict[svara])} clusters')

if plot:
    for svara, clusters in cluster_dict.items():
        print(f'Plotting {svara} cluster...')
        for i,cluster in tqdm.tqdm(list(enumerate(clusters))):
            for j,sd in cluster:
                p = sd['pitch']
                gamaka = sd['gamaka']
                t = [x*timestep for x in range(len(p))]
                
                plt.plot(t,p)
                plt.xlabel('Time (s)')
                plt.ylabel('Pitch (cents)')
                path = cpath(out_dir, 'plots', f'{track}', 'clustering', f'{svara}', f'cluster_{i}', f'{gamaka}_{j}.png')
                plt.savefig(path)
                plt.close('all')


#########################
## Pick cluster candidate
#########################
final_clusters = {}
for svara, clusters in cluster_dict.items():
    print(f'Reducing {svara} clusters to 1 candidate')
    clusts = []
    for c in clusters:
        clusts.append(random.choice(c)[1])
    final_clusters[svara] = clusts


########################
## Get Distance profiles
########################
sample = 30

if sample:
    sample_pitch_cents = pitch_cents[:round(sample/timestep)]
else:
    sample_pitch_cents = pitch_cents

distance_profiles = {}
for svara, clusters in final_clusters.items():
    print(f'Computing {svara} distance profile')
    distance_profiles[svara] = {i:[] for i in range(len(clusters))}
    for c, s in enumerate(clusters):
        print(f'Cluster {c}')
        unit = s['pitch']
        for i in tqdm.tqdm(list(range(len(sample_pitch_cents)))):
            target = sample_pitch_cents[i:i+len(unit)]
            if len(target) < wl:
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

dp_path = cpath(out_dir, 'data', 'distance_profiles', f'{track}.pkl')
write_pkl(distance_profiles, dp_path)

distance_profiles = load_pkl(dp_path)

###########
## Annotate
###########
annot_dist_thresh = 0.005

occurences = []
distances = []
lengths = []
labels = []
gamakas = []
for s,ap in distance_profiles.items():
    print(f'Annotating svara, {s}')
    for ix,i in enumerate(ap):
        gamaka = svara_dict[s][ix]['gamaka']
        dp = np.array(ap[i]).copy()

        max_dist = 0
        while max_dist < annot_dist_thresh:
            
            l = len(final_clusters[s][i]['pitch'])
            
            ix = dp.argmin()
            dist = dp[ix]
            
            dp[max(0,int(ix-l)):min(len(dp),int(ix+l))] = np.Inf

            distances.append(dist)
            lengths.append(l)
            occurences.append(ix)
            labels.append(s)
            gamakas.append(gamaka)

            max_dist = dist

occurences = np.array(occurences)
distances = np.array(distances)
lengths = np.array(lengths)
labels = np.array(labels)
gamakas = np.array(gamakas)

###########
## Clean Up
###########
def get_overlap(x0, x1, y0, y1):
    
    ix0 = set(range(x0, x1+1))
    ix1 = set(range(y0, y1+1))
    
    inters = ix1.intersection(ix0)

    o1 = len(inters)/len(ix0)
    o2 = len(inters)/len(ix1)
    
    return o1, o2


def do_patterns_overlap(x0, x1, y0, y1, perc_overlap=None):
    
    o1, o2 = get_overlap(x0, x1, y0, y1)

    if perc_overlap:
        return o1>perc_overlap and o2>perc_overlap
    else:
        return o1 > 0 and o2 > 0


def reduce_labels(occurences, labels, lengths, distances, gamakas):
    ex_svaras = set(labels)

    reduced_occ = []
    reduced_len = []
    reduced_gam = []
    reduced_dist = []
    reduced_labs = []
    for s in ex_svaras:
        ix = np.where(labels==s)[0]
        ix = sorted(ix, key=lambda y: occurences[y])
        occs = occurences[ix]
        lens = lengths[ix]
        dist = distances[ix]
        gama = gamakas[ix]

        batches = [[0]] # first occurence in batch automatically
        for i in range(len(occs))[1:]:
            o0 = occs[i-1]
            o1 = occs[i]
            l0 = lens[i-1]
            l1 = lens[i]
            overlap = do_patterns_overlap(o0, o0+l0, o1, o1+l1)
            if overlap:
                # append to existing batch
                batches[-1].append(i)
            else:
                # create new batch
                batches.append([i])

        
        # take longest of each batch
        for b in batches:
            min_t = min(occs[b])
            max_t = max(occs[b]+lens[b])
            reduced_occ.append(min_t) 
            reduced_len.append(max_t-min_t)
            reduced_labs.append(s)

    return reduced_occ, reduced_len, reduced_gam, reduced_dist, reduced_labs

occs, lens, gams, dists, labs = reduce_labels(occurences, labels, lengths, distances, gamakas)
# join identical svaras
    # record gamaka
    # record distances

# sort out borders
    # if overlap is small adjust based on distance
    # if overlap is large chose most likely based on neighbours
    # if overlap is large and no neighbour chose most lowest distance

# pass through and highlight errors as per avro aro


#########
## Export
#########
starts = [o*timestep for o in occs]
ends   = [starts[i]+(lens[i]*timestep) for i in range(len(starts))]
transcription = pd.DataFrame({
        'start':starts,
        'end':ends,
        'label':labs,
    }).sort_values(by='start')

trans_path = cpath(out_dir, 'data', 'transcription', f'{track}.txt')
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
