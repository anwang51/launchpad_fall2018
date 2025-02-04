import numpy as np
from pypianoroll import Multitrack, Track
import pypianoroll
from matplotlib import pyplot as plt
import pygame
import os

playing = False
pygame.init()

# loaded.tracks[1].pianoroll.size # check size to see if instrument is used
#.binarize() makes pitch irrelevant, might be easier to do at first w/o worring about pitch
# SHAPE (time,pitch) (-1,128), 24 steps/times per beat
# for track in tracks:
    # if tracks.pianoroll.size < 100

songs = np.load('songs.npy')

def load(npz):
    return pypianoroll.load(npz)

def play(midi='temp.mid'): # midi name
    pygame.mixer.music.load(midi)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        playing = True
    playing = False

write = Multitrack.write
def write_2(self, name='temp.mid'):
    write(self,name)
Multitrack.write = write_2

def stop():
    pygame.mixer.music.stop()

def create_dataset(): #save this w/ np.save('songs.npy',songs)
    songs = []
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if name != '.DS_Store':
                songs = np.append(songs, os.path.join(root, name))
    return songs
