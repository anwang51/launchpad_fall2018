"""
Usage:

1. Download and unzip lpd5_cleansed dataset from
https://drive.google.com/uc?id=1XJ648WDMjRilbhs4hE3m099ZQIrJLvUB&export=download
2. Get testing (10%) and training (90%) iterators
>>> test, train = test_train_sets_lpd5("/path/to/lpd5/folder", track_name='Piano')
3. Use
>>> next(train) --> returns 2d array with each row = note vels at a given timestep from 0 to 127
4. Profit

"""
import os
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pypianoroll
import mido

import midi_proc

def iter_dir(dir, ext=None, ext_set=None, recursive=True):
    """
    Generator over all files in a directory with a certain extension.
    Yields paths to the files.

    dir -- the root directory to search. File paths are appended to this, so if
           it is absolute
    ext -- the extension, e.g. '.midi'
    ext_set -- set this INSTEAD of ext to get files with multiple extensions
               e.g. set('.midi', '.mid')
    recursive -- set to False if you don't want to get files in subdirectories
                 of the root directory
    """
    for path in (os.path.join(dir, sub) for sub in os.listdir(dir)):
        if os.path.isdir(path) and recursive:
            yield from iter_dir(path, ext, ext_set, recursive)
        else:
            if ext and os.path.splitext(path)[1] == ext:
                yield path
            elif ext_set and os.path.splitext(path)[1] in ext_set:
                yield path

def get_all_paths(dir, ext=None, ext_set=None, recursive=True):
    """
    Get paths to all files in a directory as a list.
    This is just a wrapper for iter_dir, so see its documentation for details.
    """
    return list(iter_dir(dir, ext, ext_set, recursive))

lpd5_valid_tracks = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']

def iter_lpd5_file(path, track_name='Piano', beat_resolution=4, split_len=None):
    """
    Generator over a '.npz' file from the lpd5 dataset.
    Typically, this will just yield the vectors array corresponding to the
    instrument you want, but if split_len is a number then it will yield
    multiple arrays.

    path -- path to npz file
    track_name -- which instrument to get. Must be in lpd5_valid_tracks
    beat_resolution -- number of time-steps per beat
                       e.g. in 4/4 time, beat_resolution=4 -> 16th note step
    split_len -- split track into subtracks if there is a silence with more
                 than split_len beats. Default is to not split.
    """
    multitrack = midi_proc.load_npz(path)
    if multitrack:
        tracks = (track for track in multitrack.tracks if track.name == track_name)
        for track in tracks:
            if split_len:
                yield from midi_proc.split_silence(track.pianoroll, split_len=split_len)
            else:
                if track.pianoroll.size > 0: yield track.pianoroll

def iter_lpd5_dataset(root_dir, track_name='Piano', beat_resolution=4, split_len=None):
    """
    Generator over all lpd5 files in a dataset with a root file
    See iter_lpd5_file for details of the parameters
    """
    for path in iter_dir(root_dir, '.npz'):
        yield from iter_lpd5_file(path, track_name, beat_resolution, split_len)

def iter_lpd5_paths(paths, track_name='Piano', beat_resolution=4, split_len=None):
    """
    Generator over a list of paths corresponding to lpd5 files.
    See iter_lpd5_file for parameters documentation
    """
    for path in paths:
        yield from iter_lpd5_file(path, track_name, beat_resolution, split_len)

def test_train_sets_lpd5(root_dir, track_name='Piano', beat_resolution=4, split_len=None):
    """
    Partition the lpd5 set into 90% train, 10% test and yield iterators over
    both sets. Returns [test_set_iterator, train_set_iterator].
    See iter_lpd5_file for what the other parameters do
    """
    paths = get_all_paths(root_dir, '.npz')
    random.shuffle(paths)
    split_point = len(paths)//10
    test, train = paths[:split_point], paths[split_point:]
    return [iter_lpd5_paths(x, track_name, beat_resolution, split_len) for x in [test, train]]

# # does not support drum tracks right now.
# def iter_midi_file(path, allowed_programs=range(0, 5), split_len=None, frame_dur=16):
#     midi = midi_proc.load_midi(path)
#     if midi:
#         for track in midi.tracks:
#             if midi_proc.program(track) in allowed_programs:
#                 vectorized = midi_proc.vectorize_track(track, midi.ticks_per_beat, frame_dur=frame_dur)
#                 if vectorized is not None:
#                     if split_len:
#                         yield from midi_proc.split_silence(vectorized, split_len)
#                     else:
#                         yield vectorized

# def iter_midi_dataset(root_folder, allowed_programs=range(0, 5), split_len=None):
#     for path in iter_dir(root_folder, '.mid'):
#         yield from iter_midi_file(path, allowed_programs, split_len)

# def iter_midi_paths(paths, allowed_programs=range(0, 5), split_len=None):
#     for path in paths:
#         yield from iter_midi_file(path, allowed_programs, split_len)
