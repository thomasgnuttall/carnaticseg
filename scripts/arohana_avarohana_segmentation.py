import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from src.utils import load_pitch_track, cpath
from src.visualisation import plot_and_annotate, get_plot_kwargs

pitch_track_path = 'data/pitch_tracks/191_Mati_Matiki.tsv'
tonic = 191
raga = 'mohanam'

window = 10

# Load pitch track
ftanet_pitch_track = load_pitch_track(pitch_track_path)

pitch_ = ftanet_pitch_track[:,1]
time_ = ftanet_pitch_track[:,0]
timestep = time_[3]-time_[2]

# Sample
i1 = 5000
dur = 1000

pitch = pitch_[i1:i1+dur]
time = time_[i1:i1+dur]

# Create envelope - pass window across track, get average value
def get_env(track, window=2048):
    windows = np.array_split(track, round(len(track)/window))
    return [np.nanmean(w) for w in windows]

envelope = get_env(pitch, window=window)
envelope_time = np.array([time[int(x*window)] for x in range(len(envelope))])

# Smooth envelope?
envelope_smooth = savgol_filter(envelope, polyorder=2, window_length=7, mode='interp')

# Segment based on change points define arohana/avorahana
#diff = np.array([envelope_smooth[i]-envelope_smooth[i-1] for i in range(1, len(envelope_smooth)+1)])

diff = np.diff(envelope_smooth, 1)
asign = np.sign(diff)
extrema = np.where(((np.roll(asign, 1) - asign) != 0).astype(int)==1)[0]
extrema_times = envelope_time[extrema]
extrema_annot = ['arohana' if envelope_smooth[e]>envelope_smooth[e-1] else 'avarohana' for e in extrema]

annotations = list(zip(extrema_annot, extrema_times))


path =cpath('plots', 'scratch', 'env_smooth_test.png')

from scipy import signal

b, a = signal.butter(8, 0.05)
filtered_signal = signal.filtfilt(b, a, pitch, method="gust")
plt.plot(time, pitch)
plt.plot(time,filtered_signal)
plt.savefig(path)
plt.close('all')


# annotate plot to verifys
plot_path = cpath('plots', 'scratch', 'arohana_seg.png')
yticks_dict = get_plot_kwargs(raga, tonic)['yticks_dict']

plot_and_annotate(pitch, time, annotations, plot_path, yticks_dict=yticks_dict, title=None, figsize=(10,4))


