import os
import pypianoroll
import mido
import midi_proc

## FUNCTIONS to USE: --> iter_lpd5_dataset and iter_midi_dataset

def iter_dir(dir, ext=None, ext_set=None, recursive=True):
    for path in (os.path.join(dir, sub) for sub in os.listdir(dir)):
        if os.path.isdir(path) and recursive:
            yield from iter_dir(path, ext, ext_set, recursive)
        else:
            if ext and os.path.splitext(path)[1] == ext:
                yield path
            elif ext_set and os.path.splitext(path)[1] in ext_set:
                yield path

def get_all_paths(dir, ext=None, ext_set=None, recursive=True):
    return list(iter_dir(dir, ext, ext_set, recursive))

lpd5_valid_tracks = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']

def iter_lpd5_file(path, track_name='Piano', split_len=None):
    multitrack = midi_proc.load_npz(path)
    if multitrack:
        tracks = (track for track in multitrack.tracks if track.name == track_name)
        for track in tracks:
            if split_len:
                yield from midi_proc.split_silence(track.pianoroll, split_len=split_len)
            else:
                if track.pianoroll.size > 0: yield track.pianoroll

def iter_lpd5_dataset(root_dir, track_name='Piano', split_len=None):
    # track_name should be one of lpd5_valid_tracks
    for path in iter_dir(root_dir, '.npz'):
        yield from iter_lpd5_file(path, track_name, split_len)

def iter_lpd5_paths(paths, track_name='Piano', split_len=None):
    for path in paths:
        yield from iter_lpd5_file(path, track_name, split_len)

# does not support drum tracks right now.
def iter_midi_file(path, allowed_programs=range(0, 5), split_len=None, frame_dur=16):
    midi = midi_proc.load_midi(path)
    if midi:
        for track in midi.tracks:
            if midi_proc.program(track) in allowed_programs:
                vectorized = midi_proc.vectorize_track(track, midi.ticks_per_beat, frame_dur=frame_dur)
                if vectorized is not None:
                    if split_len:
                        yield from midi_proc.split_silence(vectorized, split_len)
                    else:
                        yield vectorized

def iter_midi_dataset(root_folder, allowed_programs=range(0, 5), split_len=None):
    for path in iter_dir(root_folder, '.mid'):
        yield from iter_midi_file(path, allowed_programs, split_len)

def iter_midi_paths(paths, allowed_programs=range(0, 5), split_len=None):
    for path in paths:
        yield from iter_midi_file(path, allowed_programs, split_len)