%load_ext autoreload
%autoreload 2

import compiam
import os 
import librosa
import mirdata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import tqdm
import seaborn as sns

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


###############
# Get sequences
###############
full_seq = list(annotations['label'].values)
full_seq = ['BREAK' if '(' in x else x.lower().strip() for x in full_seq]
full_seq = [x.replace('?','BREAK') for x in full_seq]
full_seq = ' '.join(full_seq)
sequences = [x.split(' ') for x in full_seq.split('BREAK') if x]
sequences = [s for s in sequences if not all([not x for x in s])]
sequences = [[x for x in s if x] for s in sequences]


##############
# Markov Chain
##############
def transition_matrix(arr, states, n=1):
    """"
    Computes the transition matrix from Markov chain sequence of order `n`.

    :param arr: Discrete Markov chain state sequence in discrete time with states in 0, ..., N
    :param n: Transition order
    """

    num_arr = [[states.index(y) for y in x] for x in arr]

    n_states = max([x for y in num_arr for x in y])
    M = np.zeros(shape=(n_states + 1, n_states + 1))
    for a in num_arr:
        for (i, j) in zip(a, a[1:]):
            M[i, j] += 1

    T = (M.T / M.sum(axis=1)).T

    return np.linalg.matrix_power(T, n)

SVARAS = ['sa', 'ri', 'ga', 'ma', 'pa', 'dha', 'ni']
states = set([x for y in sequences for x in y])
states = [s for s in SVARAS if s in states]

M1 = transition_matrix(sequences, states, 1)
M2 = transition_matrix(sequences, states, 2)
M3 = transition_matrix(sequences, states, 3)

######
# Plot
######
M_plot_path = cpath('plots', f'{track}', 'transition_matrix', 'first_order.png')

plt.figure(figsize = (6,6))
sns.heatmap(M, annot=True, xticklabels=states, yticklabels=states)
plt.title(f'First order svara transition matrix for {track}')
plt.savefig(M_plot_path)
plt.close('all')


