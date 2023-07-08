import compiam
import librosa
import pandas as pd

from src.utils import interpolate_below_length, write_pitch_track

# Get data
vocal_path = 'path here'
tonic_path = 'tonic path here'

# Load Data
sr = 44100
vocal, _ = librosa.load(vocal_path, sr=sr)

tonic = read_txt(tonic_path)
tonic = float(tonic[0].replace('\n',''))

# Pitch track -> change points
ftanet_carnatic = compiam.load_model("melody:ftanet-carnatic")
ftanet_pitch_track = ftanet_carnatic.predict(vocal_path,)

pitch = ftanet_pitch_track[:,1]
time = ftanet_pitch_track[:,0]
timestep = time[3]-time[2]


pitch = interpolate_below_length(pitch, 0, (250*0.001/timestep))
null_ind = pitch==0

# Smoothing
#pitch = gaussian_filter1d(pitch, 2.5)
#pitch = savgol_filter(pitch, polyorder=2, window_length=13, mode='interp')
pitch[pitch<50]=0
pitch[null_ind]=0

pitch_cents = pitch_seq_to_cents(pitch, tonic)
