import os
import pypianoroll

import dataset_io
import midi2vec

piano = range(0, 5)

def lpd5_piano_generator(lpd5_root):
    for file in dataset_io.get_files(lpd5_root, '.npz'):
        tracks = pypianoroll.load(file).tracks
        piano_tracks = (track for track in tracks if track.program in piano and (not track.is_drum))
        for track in piano_tracks:
            yield from midi2vec.split_silence(track.pianoroll)