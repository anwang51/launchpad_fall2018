# pianoroll_utils.py

import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import mido
import pypianoroll
import time

get_npzs = get_ext('.npz')
piano = range(0, 5)

def get_piano(npzs):
    tracks = []
    for npz in npzs:
        roll = pypianoroll.load(npz)
        tracks += [track for track in roll.tracks if track.program in piano and (not track.is_drum)]
    return tracks