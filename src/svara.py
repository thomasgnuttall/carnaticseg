import os
import tqdm

from scipy.signal import savgol_filter

from src.dtw import dtw
from src.utils import cpath, write_pkl

SVARAS = ['sa', 'ri', 'ga', 'ma', 'pa', 'dha', 'ni']

def chop_time_series(ts, s, e, timestep):
    s = round(s/timestep)
    e = round(e/timestep)
    return ts[s:e]


def get_unique_svaras(annotations):
    svaras = list(set([s.strip().lower() for s in annotations['label']]))
    return [s for s in SVARAS if s in  svaras]


def get_svara_dict(annotations, pitch_cents, timestep, track, min_length=0.145, smooth_window=0.145, path=None, plot_dir=None, verbose=True):
    
    if min_length < smooth_window:
        raise Exception(f'<min_length> cannot be smaller than <smooth_window>')

    svara_dict = {}
    for i,row in annotations.iterrows():
        start = row['start']
        end = row['end']
        label = row['label'].strip().lower()
        duration = end-start
        if i != 0:
            prev = annotations.iloc[i-1]
            if start - prev['end'] < 2:
                prev_svara = prev['label'].strip().lower()
            else:
                prev_svara = 'silence'
        else:
            prev_svara = None

        if i != len(annotations)-1:
            nex = annotations.iloc[i+1]
            if nex['start'] - end < 2:
                next_svara = nex['label'].strip().lower()
            else:
                next_svara = 'silence'
        else:
            suc_svara = None

        if label not in SVARAS:
            continue

        pitch_curve = chop_time_series(pitch_cents, start, end, timestep)
        pitch_curve = pitch_curve[pitch_curve!=None]
        
        if len(pitch_curve)*timestep < min_length:
            print(f'Index {i} discarded, below minimum length')
            continue
        
        wl = round(smooth_window/timestep)
        wl = wl if not wl%2 == 0 else wl+1
        pitch_curve = savgol_filter(pitch_curve, polyorder=2, window_length=wl, mode='interp')

        d = {
                'pitch': pitch_curve,
                'track': track,
                'start': start,
                'end': end,
                'duration': duration,
                'annotation_index': i,
                'preceeding_svara': prev_svara,
                'succeeding_svara': next_svara
            }

        if label in svara_dict:
            svara_dict[label].append(d)
        else:
            svara_dict[label] = [d]
    
    if verbose:
        for k,v in svara_dict.items():
            print(f'{len(v)} occurrences of {k}')

    if path:
        svara_dict_path = cpath(path)
        write_pkl(svara_dict, svara_dict_path)

    if plot_dir:
        for svara in svara_dict.keys():
            all_svaras = svara_dict[svara]
            for i in range(len(all_svaras)):
                pt = all_svaras[i]['pitch']
                times = [x*timestep for x in range(len(pt))]
                path = cpath(plot_dir, svara, f'{i}.png')
                plt.plot(times, pt)
                plt.savefig(path)
                plt.close('all')

    return svara_dict


def pairwise_distances_to_file(ix, all_svaras, path, r=0.05, mean_norm=False):
    try:
        print('Removing previous distances file')
        os.remove(path)
    except OSError:
        pass

    header = 'index1,index2,distance'
    with open(path,'a') as file:
        file.write(header)
        file.write('\n')
        for i in tqdm.tqdm(ix):
            for j in ix:
                if i <= j:
                    continue
                pat1 = all_svaras[i]['pitch']
                pat2 = all_svaras[j]['pitch']

                pi = len(pat1)
                pj = len(pat2)
                l_longest = max([pi, pj])

                path, dtw_val = dtw(pat1, pat2, radius=round(l_longest*r), mean_norm=mean_norm)
                l = len(path)
                dtw_norm = dtw_val/l

                row = f'{i},{j},{dtw_norm}'
                
                file.write(row)
                file.write('\n')


def get_centered_svaras(svara):
    MASSVARAS = SVARAS + SVARAS + SVARAS
    occs = [i for i,x in enumerate(MASSVARAS) if x==svara]
    secocc = occs[1]
    svaras = MASSVARAS[secocc-3: secocc+4]
    return svaras



def asc_desc(n0, n, n2):
    cent_svara = get_centered_svaras(n)
    ni = cent_svara.index(n)
    
    if n0 not in cent_svara: # i.e. silence or unknown
        n2i = cent_svara.index(n2)
        if ni < n2i:
            return 'asc'
        elif ni > n2i:
            return 'desc'
        else:
            return 'cp'  

    elif n2 not in cent_svara:
        n0i = cent_svara.index(n0)
        if n0i < ni:
            return 'asc'
        elif n0i > ni:
            return 'desc'
        else:
            return 'cp'

    elif n0 not in cent_svara and n2 not in cent_svara:
        return np.nan
    
    n0i = cent_svara.index(n0)
    n2i = cent_svara.index(n2)

    if n0i < ni < n2i:
        return 'asc'
    elif n0i > ni > n2i:
        return 'desc'
    else:
        return 'cp'