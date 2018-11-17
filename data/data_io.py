import os
import pypianoroll
import mido
import midi_proc

def iter_dir(dir, ext=None, ext_set=None, recursive=True):
    for path in (os.path.join(dir, sub) for sub in os.listdir(dir)):
        if os.path.isdir(path) and recursive:
            yield from iter_dir(path, ext, ext_set, recursive)
        else:
            if ext and os.path.splitext(path)[1] == ext:
                yield path
            elif ext_set and os.path.splitext(path)[1] in ext_set:
                yield path

lpd5_valid_tracks = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']

def iter_lpd5_dataset(root_dir, track_name='Piano', min_len=None):
    # track_name should be one of lpd5_valid_tracks
    for path in iter_dir(root_dir, '.npz'):
        multitrack = pypianoroll.load(path)
        if multitrack:
            tracks = (track for track in multitrack.tracks if track.name == track_name)
            for track in tracks:
                if min_len:
                    yield from midi_proc.split_silence(track.pianoroll, min_len=min_len)
                else:
                    yield track

def iter_midi_dataset(root_folder, allowed_programs=range(0, 5), min_len=None):
    for path in iter_dir(root_folder, '.mid'):
        midi = midi_proc.load_midi(path)
        if midi:
            for track in midi.tracks:
                if midi_proc.program(track) in allowed_programs:
                    vectorized = midi_proc.vectorize_track(track, midi.ticks_per_beat)
                    if min_len:
                        yield from midi_proc.split_silence(vectorized, min_len)
                    else:
                        yield track