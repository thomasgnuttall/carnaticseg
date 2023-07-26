import os
import pickle
import json 
import yaml
import csv
import numpy as np
import pandas as pd 

def load_json(path):
    """
    Load json at <path> to dict
    
    :param path: path of json
    :type path: str

    :return: dict of json information
    :rtype: dict
    """ 
    # Opening JSON file 
    with open(path) as f: 
        data = json.load(f) 
    return data


def write_json(j, path):
    """
    Write json, <j>, to <path>

    :param j: json
    :type path: json
    :param path: path to write to, 
        if the directory doesn't exist, one will be created
    :type path: str
    """ 
    create_if_not_exists(path)
    # Opening JSON file 
    with open(path, 'w') as f:
        json.dump(j, f)


def read_txt(path):
    with open(path) as f:
        lines = f.readlines()
    return lines


def myround(x, base=5):
    return base * round(x/base)
    

def load_yaml(path):
    """
    Load yaml at <path> to dictionary, d
    
    Returns
    =======
    Wrapper dictionary, D where
    D = {filename: d}
    """
    import zope.dottedname.resolve
    def constructor_dottedname(loader, node):
        value = loader.construct_scalar(node)
        return zope.dottedname.resolve.resolve(value)

    def constructor_paramlist(loader, node):
        value = loader.construct_sequence(node)
        return ParamList(value)

    yaml.add_constructor('!paramlist', constructor_paramlist)
    yaml.add_constructor('!dottedname', constructor_dottedname)

    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = yaml.load(f, Loader=yaml.FullLoader)   
    return d


def create_if_not_exists(path):
    """
    If the directory at <path> does not exist, create it empty
    """
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ''):
        os.makedirs(directory)


def cpath(*args):
    """
    Wrapper around os.path.join, create path concatenating args and
    if the containing directories do not exist, create them.
    """
    path = os.path.join(*args)
    create_if_not_exists(path)
    return path


def write_pitch_track(pitch_track, path, sep='\t'):
    """
    Write pitch contour to tsv at <path>
    """
    with open(path,'w') as file:
        for t, p in pitch_track:
            file.write(f"{t}{sep}{p}")
            file.write('\n')


def load_pitch_track(path, delim='\t'):
    """
    load pitch contour from tsv at <path>

    :param path: path to load pitch contour from
    :type path: str

    :return: Two numpy arrays of time and pitch values
    :rtype: tuple(numpy.array, numpy.array)
    """
    pitch_track = []
    with open(path) as fd:
        rd = csv.reader(fd, delimiter=delim, quotechar='"')
        for t,p in rd:
            pitch_track.append([float(t),float(p)])

    return np.array(pitch_track)


# Load annotations
def load_audacity_annotations(path, delim='\t'):
    vals = []
    with open(path) as fd:
        rd = csv.reader(fd, delimiter=delim, quotechar='"')
        for s,e,a in rd:
            vals.append([float(s), float(e), str(a)])
    
    return pd.DataFrame(vals, columns=['start', 'end', 'label'])


def write_pkl(o, path):
    create_if_not_exists(path)
    with open(path, 'wb') as f:
        pickle.dump(o, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    file = open(path,'rb')
    return pickle.load(file)