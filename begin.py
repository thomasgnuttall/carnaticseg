%load_ext autoreload
%autoreload 2

import compiam
import librosa
from librosa.onset import onset_detect
import soundfile as sf
import pandas as pd

from utils import load_json, read_txt
from tools import get_loudness, interpolate_below_length
from pitch import pitch_seq_to_cents
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, hilbert
from scipy.ndimage import gaussian_filter1d

from sklearn.cluster import DBSCAN
from visualisation import plot_kwargs, plot_subsequence

# Load saraga dataset mohanam
import mirdata

raga = 'mohanam'

data_home = '/Volumes/MyPassport/mir_datasets2/saraga1.5_carnatic/'
saraga = mirdata.initialize('saraga_carnatic', data_home=data_home)

#carnatic = compiam.load_dataset('saraga_carnatic')
tracks = saraga.load_tracks()

mohanam_tracks = {}
for t, v in tracks.items():
    metadata = load_json(v.metadata_path)
    r = metadata['raaga']
    if len(r) > 0:
        if 'common_name' in r[0]:
            name = r[0]['common_name']
            if name  == raga:
                mohanam_tracks[t] = metadata


track = '191_Mati_Matiki'

# Get data
mridangam_left_path = tracks[track].audio_mridangam_left_path
mridangam_right_path = tracks[track].audio_mridangam_right_path
vocal_path = tracks[track].audio_vocal_path
tonic_path = tracks[track].ctonic_path
sections_path = tracks[track].sections_path
tempo_path = tracks[track].tempo_path

# Load Data
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

# get sample
start = sections[sections['section']=='Kalpanā svara']['start'].iloc[0]
duration = sections[sections['section']=='Kalpanā svara']['duration'].iloc[0]

mridangam_left = mridangam_left[round(start*sr):round((start+duration)*sr)]
mridangam_right = mridangam_right[round(start*sr):round((start+duration)*sr)]
vocal = vocal[round(start*sr):round((start+duration)*sr)]

kalpana_path = f'{track}_kalpanasvara.wav'

sf.write(kalpana_path, vocal, samplerate=sr)

# Extract Features

# Pitch track -> change points
ftanet_carnatic = compiam.load_model("melody:ftanet-carnatic")
ftanet_pitch_track = ftanet_carnatic.predict(kalpana_path,)

pitch = ftanet_pitch_track[:,1]
time = ftanet_pitch_track[:,0]
time = time+start
timestep = time[3]-time[2]


pitch = interpolate_below_length(pitch, 0, (250*0.001/timestep))
null_ind = pitch==0
#pitch = gaussian_filter1d(pitch, 2.5)
#pitch = savgol_filter(pitch, polyorder=2, window_length=13, mode='interp')
pitch[pitch<50]=0
pitch[null_ind]=0

pitch_cents = pitch_seq_to_cents(pitch, tonic)

# Loudness
loudness = get_loudness(vocal)
loudness_step = len(vocal)/(sr*len(loudness))


# Get sample
start_point = 1828.6 -start# seconds
duration = 18 # seconds
end_point = start_point+duration
min_t = round(start_point/timestep)
max_t = round(end_point/timestep)

time_samp_pitch = time[min_t:max_t]
pitch_samp = pitch[round(start_point/timestep):round(end_point/timestep)]

loudness_samp = loudness[round(start_point/loudness_step):round(end_point/loudness_step)]
time_samp_loudness = np.arange(round(start_point), round(end_point), (round(end_point)-round(start_point))/len(loudness_samp))

mridangam_left_samp = mridangam_left[round(start_point*sr):round(end_point*sr)]
mridangam_right_samp = mridangam_right[round(start_point*sr):round(end_point*sr)]
vocal_samp = vocal[round(start_point*sr):round(end_point*sr)]
loudness_samp = loudness[round(start_point/loudness_step):round(end_point/loudness_step)]
time_samp_audio = np.arange(round(start_point), round(end_point), (round(end_point)-round(start_point))/len(vocal_samp))

sample_path = f'{track}_sample.wav'

sf.write(sample_path, vocal_samp, samplerate=sr)

annotations = [(0.139, "Dha"), (1.091, "Pa"), (1.440, "Ga"), (1.904, "Pa"), (2.287, "Dha"), (3.158, "Pa"), (3.541, "Ga"), (4.087, "Dha"), (4.447, "Pa"), (4.828, "Dha"), (5.271, "Ga"), (6.269, "Pa"), (6.571, "Dha"), (7.001, "Sa"), (7.372, "Pa"), (8.290, "Dha"), (8.676, "Ga"), (9.604, "Pa"), (10.000," Dha"), (10.457," Sa"), (10.867," Re"), (11.313," Dha"), (12.359," Sa"), (12.770," Pa"), (13.523," Ga"), (13.930," Ma"), (14.573," Ga"), (15.194," Dha"), (16.088," Sa")]




















# Peaks and Onsets
loudness_peaks, _ = find_peaks(loudness_samp, height=-40)
mridangam_left_onset = onset_detect(mridangam_left_samp, sr=sr, units='samples')
mridangam_right_onset = onset_detect(mridangam_right_samp, sr=sr, units='samples')


def get_env(vocal_samp):
    windows =np.array_split(vocal_samp, round(len(vocal_samp)/4096))
    return [np.max(w) for w in windows]

env = get_env(vocal_samp)
vocal_onset, _ = find_peaks(env, height=0.01, prominence=0.005)
vocal_onset = [x*4096 for x in vocal_onset]
#vocal_onset = onset_detect(vocal_samp, sr=sr, units='samples')

# convert to time
loudness_peaks_s = [l*loudness_step for l in loudness_peaks]
mridangam_left_peaks_s = [l/sr for l in mridangam_left_onset]
mridangam_right_peaks_s = [l/sr for l in mridangam_right_onset]
vocal_peaks_s = [l/sr for l in vocal_onset]

# Plot peaks
plt.figure(figsize=(50,1))
plt.plot(loudness_peaks_s, [0 for o in loudness_peaks_s],'.')
#plt.plot(mridangam_left_peaks_s, [0 for o in mridangam_left_peaks_s],'.')
#plt.plot(mridangam_right_peaks_s, [0 for o in mridangam_right_peaks_s],'.')
plt.plot(vocal_peaks_s, [0 for o in vocal_peaks_s],'.')
plt.tight_layout()
plt.grid()
plt.savefig('peaks.png')
plt.close('all')

# Join all segmentation points
min_in_group = 1

all_peaks = np.array(
    loudness_peaks_s + 
    mridangam_right_peaks_s + 
    #mridangam_left_peaks_s + 
    vocal_peaks_s
)
clustering = DBSCAN(eps=0.1, min_samples=1).fit(all_peaks.reshape(-1, 1))
labels = clustering.labels_
labels_peaks = list(zip(labels, all_peaks))
clusters = {p:[] for p in range(max(labels)+1)}
for c,p in labels_peaks:
    clusters[c].append(p)

final_peaks = []
for c, pks in clusters.items():
    if len(pks) >= min_in_group:
        final_peaks.append(np.mean(pks))
final_peaks = sorted(final_peaks)
 
# Plot
fig, axs = plt.subplots(nrows = 5, figsize=(14, 30))
plt.subplots_adjust(hspace=0.3)
plt.suptitle('Mati Matiki - Sumithra Vasudev\nKalpanasvara Sample')
plt.xlabel('Time (s)')
axs[0].plot(time_samp_loudness, loudness_samp)
axs[0].plot(time_samp_loudness[loudness_peaks], loudness_samp[loudness_peaks],'x')
axs[1].plot(time_samp_audio, vocal_samp)
axs[1].plot(time_samp_audio[vocal_onset], vocal_samp[vocal_onset], 'x')
axs[2].plot(time_samp_audio, mridangam_left_samp)
axs[2].plot(time_samp_audio[mridangam_left_onset], [0 for x in mridangam_left_onset], 'x')
axs[3].plot(time_samp_audio, mridangam_right_samp)
axs[3].plot(time_samp_audio[mridangam_right_onset], [0 for x in mridangam_right_onset], 'x')
axs[4].plot([x-start for x in time_samp_pitch], pitch_samp)
for f in final_peaks:
    axs[4].axvline(f+start_point, linestyle='--', color='forestgreen')

axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[3].grid()
axs[4].grid()

axs[0].set_ylabel('dB')
axs[1].set_ylabel('Amplitude')
axs[2].set_ylabel('Amplitude')
axs[3].set_ylabel('Amplitude')
axs[4].set_ylabel('Frequency (Hz)')

axs[0].set_title('Loudness', fontsize=10)
axs[1].set_title('Vocal', fontsize=10)
axs[2].set_title('Mridangam Left', fontsize=10)
axs[3].set_title('Mridangam Right', fontsize=10)
axs[4].set_title('Predominant Pitch Track', fontsize=10)

plt.savefig('test.png')
plt.close('all')
## axs[4].plot(loudness_samp, time_samp)


# Visualise with annotations
fig, ax = plt.subplots()
plt.figure(figsize=(20,4))
plt.plot(time_samp_pitch, pitch_samp)

ytick = {k:v for k,v in yticks_dict.items() if v<=max(pitch_samp) and v>190}
tick_names = list(ytick.keys())
tick_loc = [p for p in ytick.values()]
plt.yticks(ticks=tick_loc, labels=tick_names)
plt.ylim((190,500))

for t, a in annotations:
    t = t+start+start_point
    plt.axvline(t, linestyle='--', color='red', linewidth=1)
    plt.annotate(a, (t+0.1,470), rotation=90)

plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Manual annotation of Mati Matiki - Sumithra Vasudev (ragam mohanam)')
plt.savefig('with_annotations.png')
plt.close('all')



# Visualise automatic
fig, ax = plt.subplots()
plt.figure(figsize=(20,4))
plt.plot(time_samp_pitch, pitch_samp)

ytick = {k:v for k,v in yticks_dict.items() if v<=max(pitch_samp) and v>190}
tick_names = list(ytick.keys())
tick_loc = [p for p in ytick.values()]
plt.yticks(ticks=tick_loc, labels=tick_names)
plt.ylim((190,500))

for f in final_peaks:
    f = f+start+start_point
    plt.axvline(f, linestyle='--', color='red', linewidth=1)

plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Automatic Segmentation of Mati Matiki - Sumithra Vasudev (ragam mohanam)')
plt.savefig('with_segmentation.png')
plt.close('all')

#p = plot_subsequence(start_point, duration_samples, pitch, time, timestep, path='test.png', plot_kwargs=plot_kwargs, margin=0.1)







# Grammar analysis


