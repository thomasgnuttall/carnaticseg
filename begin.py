%load_ext autoreload
%autoreload 2

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
from src.utils import load_audacity_annotations, cpath, write_pkl, write_pitch_track, load_pitch_track, read_txt
from src.visualisation import get_plot_kwargs, plot_subsequence
from src.pitch import pitch_seq_to_cents

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
ftanet_pitch_track = ftanet_carnatic.predict(vocal_path,hop_size=30)
write_pitch_track(ftanet_pitch_track, pitch_track_path, sep='\t')
ftanet_pitch_track = load_pitch_track(pitch_track_path)


pitch = ftanet_pitch_track[:,1]
time = ftanet_pitch_track[:,0]
timestep = time[3]-time[2]

pitch = interpolate_below_length(pitch, 0, (250*0.001/timestep))
null_ind = pitch==0

#pitch = savgol_filter(pitch, polyorder=2, window_length=13, mode='interp')
pitch[pitch<50]=0
pitch[null_ind]=0

pitch_cents = pitch_seq_to_cents(pitch, tonic)



# Loudness
loudness = get_loudness(vocal)
loudness_step = len(vocal)/(sr*len(loudness))

# Spectral Flux
spectral_flux, Fs_feature = compute_novelty_spectrum(vocal)
sf_step = len(vocal)/(sr*len(spectral_flux))

#####################
# Get Svaras Features
#####################
def chop_time_series(ts, s, e, timestep):
    s = round(s/timestep)
    e = round(e/timestep)
    return ts[s:e]

def trim_zeros(pitch, time):
    m = pitch!=0
    i1,i2 = m.argmax(), m.size - m[::-1].argmax()
    return pitch[i1:i2], time[i1:i2]

svara_dict = {}
for i,row in annotations.iterrows():
    start = row['start']
    end = row['end']
    label = row['label'].strip().lower()
    duration = end-start

    if label not in ['sa', 'ri', 'ga', 'ma', 'pa', 'dha', 'ni']:
        continue

    pitch_curve = chop_time_series(pitch_cents, start, end, timestep)
    pitch_curve = pitch_curve[pitch_curve!=None]

    loudness_curve = chop_time_series(loudness, start, end, loudness_step)
    sf_curve = chop_time_series(spectral_flux, start, end, sf_step)

    d = {
            'pitch': pitch_curve,
            'loudness': loudness_curve,
            'spectral_flux': sf_curve,
            'track': track,
            'start': start,
            'end': end,
            'duration': duration,
            'annotation_index': row.index
        }

    if label in svara_dict:
        svara_dict[label].append(d)
    else:
        svara_dict[label] = [d]

svara_dict_path = cpath('data', 'svara_dict', f'{track}.pkl')
write_pkl(svara_dict, svara_dict_path)

for k,v in svara_dict.items():
    print(f'{len(v)} occurrences of {k}')

# save all svaras
for svara in svara_dict.keys():
    print(svara)
    all_svaras = svara_dict[svara]
    for i in range(len(all_svaras)):
        pt = all_svaras[i]['pitch']
        times = [x*timestep for x in range(len(pt))]
        path = cpath('plots', str(track), 'all_svaras', svara, f'{i}.png')
        plt.plot(times, pt)
        plt.savefig(path)
        plt.close('all')



svara = 'dha'

all_ix = list(range(len(all_svaras)))

#####
# DTW
#####
dtw_distances_path = cpath('data', 'dtw_distances', 'svara', f'{track}', f'{svara}.csv')
try:
    print('Removing previous distances file')
    os.remove(dtw_distances_path)
except OSError:
    pass

# i=23
# j=1
# s1 = all_svaras[i]['pitch']
# s2 = all_svaras[j]['pitch']

# pi = len(s1)
# pj = len(s2)
# l_longest = max([pi, pj])
# radius = round(l_longest*0.1)

# from src.dtw import *

##text=List of strings to be written to file
bad_indices = []
header = 'index1,index2,dtw'
with open(dtw_distances_path,'a') as file:
    file.write(header)
    file.write('\n')
    for i in tqdm.tqdm(all_ix):
        for j in all_ix:
            print(f'i={i}, j={j}')
            if i <= j:
                continue
            
            pat1 = all_svaras[i]['pitch']
            pat2 = all_svaras[j]['pitch']
            
            if len(pat1)==0 or len(pat2)==0:
                if len(pat1)==0:
                    bad_indices.append(i)
                if len(pat2)==0:
                    bad_indices.append(j)
                continue
            pat1,_ = get_derivative(pat1, pat1)
            pat2,_ = get_derivative(pat2, pat2)

            pi = len(pat1)
            pj = len(pat2)
            l_longest = max([pi, pj])

            path, dtw_val = dtw(pat1, pat2, radius=round(l_longest*0.01))
            l = len(path)
            dtw_norm = dtw_val/l

            row = f'{i},{j},{dtw_norm}'
            
            file.write(row)
            file.write('\n')

bad_indices = list(set(bad_indices))

########
# COSINE
########
cosine_distances_path = cpath('data', 'cosine_distances', 'svara', f'{track}', f'{svara}.csv')
try:
    print('Removing previous distances file')
    os.remove(cosine_distances_path)
except OSError:
    pass

# i=23
# j=1
# s1 = all_svaras[i]['pitch']
# s2 = all_svaras[j]['pitch']

# pi = len(s1)
# pj = len(s2)
# l_longest = max([pi, pj])
# radius = round(l_longest*0.1)

# from src.dtw import *

def get_features(pitch, timestep, window=0.05):
    
    window = round(window*len(pitch))

    if window < 1:
        window = 1

    features = {
        'max_freq': np.max(pitch),
        'min_freq': np.min(pitch),
        'std_freq': np.std(pitch),
        'mean_freq': np.mean(pitch),
        'start_freq': np.mean(pitch[:window]),
        'end_freq': np.mean(pitch[-window:])
    }

    return features

cosine_similarity = lambda a,b: dot(a, b)/(norm(a)*norm(b))

##text=List of strings to be written to file
header = 'index1,index2,cosine'
with open(cosine_distances_path,'a') as file:
    file.write(header)
    file.write('\n')
    for i in tqdm.tqdm(all_ix):
        for j in all_ix:
            print(f'i={i}, j={j}')
            if i <= j:
                continue
            pat1 = all_svaras[i]['pitch']
            pat2 = all_svaras[j]['pitch']
            
            if len(pat1)==0 or len(pat2)==0:
                continue

            pat1_feat = np.array(list(get_features(pat1, timestep).values()))
            pat2_feat = np.array(list(get_features(pat2, timestep).values()))

            cos = 1-cosine_similarity(pat1_feat, pat2_feat)

            row = f'{i},{j},{cos}'
            
            file.write(row)
            file.write('\n')


#############
## Clustering
#############
distances = pd.read_csv(cosine_distances_path)

def get_inter_intra(df, dcol):
    inter = df[df['cluster1']==df['cluster2']][dcol].mean()
    intra = df[df['cluster1']!=df['cluster2']][dcol].mean()
    return inter/intra

distances_flip = distances.copy()
distances_flip.columns = ['index2', 'index1', 'dtw']
distances = pd.concat([distances, distances_flip])

dist_piv = distances.pivot("index1", "index2", 'dtw').fillna(0)
indices = dist_piv.columns

piv_arr = dist_piv.values
X = piv_arr + np.transpose(piv_arr)

Z = linkage(squareform(X), 'ward')

metrics = []
all_ts = list(np.arange(0, 1, 0.02))
for t in tqdm.tqdm(all_ts):
    clustering = fcluster(Z, t=t, criterion='distance')

    clusterer = {x:y for x,y in zip(indices, clustering)}

    distances['cluster1'] = distances['index1'].apply(lambda y: clusterer[y])
    distances['cluster2'] = distances['index2'].apply(lambda y: clusterer[y])

    metrics.append(get_inter_intra(distances, 'dtw'))

plot_path = cpath('plots', f'{track}', f'deriv_hierarchical_svara_clustering_{svara}.png')

plt.plot(all_ts, metrics)
plt.title('Varying t')
plt.xlabel('t')
plt.ylabel('Ratio of inter:intra cluster distance')
plt.savefig(plot_path)
plt.close('all')


t = 5
clustering = fcluster(Z, t=t, criterion='distance', R=None, monocrit=None)

for i,ix in enumerate(indices):
    pt = all_svaras[ix]['pitch']
    times = [x*timestep for x in range(len(pt))]
    cluster = clustering[i]
    path = cpath('plots', str(track), 'clustering', svara, str(cluster), f'{i}.png')
    plt.plot(times, pt)
    plt.savefig(path)
    plt.close('all')

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
