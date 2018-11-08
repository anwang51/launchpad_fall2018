import os
import pypianoroll

import midi_proc

def get_files(dir, ext, recursive=True):
    for sub in [os.path.join(dir, sub) for sub in os.listdir(dir)]:
        if os.path.isdir(sub):
            yield from get_files(sub, ext)
        elif os.path.splitext(sub)[1] == ext:
            yield sub

def lpd5_piano_generator(lpd5_root, min_len=16):
    piano = range(0, 5) # Tracks that are considered piano
    for file in get_files(lpd5_root, '.npz'):
        tracks = pypianoroll.load(file).tracks
        piano_tracks = (track for track in tracks if track.program in piano and (not track.is_drum))
        for track in piano_tracks:
            yield from midi_proc.split_silence(track.pianoroll, min_len=min_len)