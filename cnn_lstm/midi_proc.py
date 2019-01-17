"""
Usage:
vectorize('./myfile.mid', 16) -> list of vectors for each track, qtized to 16th
"""

# see also https://github.com/callaunchpad/JazzImprov/blob/master/midi_dto_csv/midi_test.ipynb
import os
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mido
import pypianoroll

out = None

def load_npz(path):
    try:
        return pypianoroll.load(path)
    except Exception as e:
        return None

def load_midi(path):
    try:
        return mido.MidiFile(path)
    except Exception as e:
        return None

def num_frames(ticks, ticks_per_beat, note_denominator):
    # 1 beat = 1 quarter
    return int((note_denominator/4 * ticks) // ticks_per_beat)

def is_note(event):
    return event.type == 'note_on' or event.type == 'note_off'

def is_drum(note):
    return is_note(note) and note.channel == 9

def is_program_change(event):
    return event.type == 'program_change'

def program(track):
    for event in track:
        if is_program_change(event):
            return event.program

def programs(track):
    all_programs = set()
    for event in track:
        if is_program_change(event):
            all_programs.add(event.program)
    return all_programs

def vectorize_midi(midi, frame_dur=16, drum_mode=False, merge_tracks=False):
    if merge_tracks:
        return vectorize_track(mido.merge_tracks(midi.tracks), midi.ticks_per_beat, frame_dur, drum_mode)
    else:
        vectorized_tracks = [vectorize_track(track, midi.ticks_per_beat, frame_dur, drum_mode) for track in midi.tracks]
        return [track for track in vectorized_tracks if track is not None]

def vectorize_track(track, ticks_per_beat, frame_dur=16, drum_mode=False, trim=True):
    frames = list()
    current_frame = np.zeros(128)
    for event in track:
        if event.time:
            frames_to_add = num_frames(event.time, ticks_per_beat, frame_dur)
            for _ in range(frames_to_add):
                frames.append(np.array(current_frame))
        if is_note(event) and (drum_mode == is_drum(event)):
            current_frame[event.note] = event.velocity / 127.0
    if trim:
        frames = trim_silence(frames)
    if frames:
        return np.vstack(frames)

def trim_silence(frames):
    first_nonzero_index = None
    last_nonzero_index = None
    for i in range(len(frames)):
        if np.any(frames[i]):
            first_nonzero_index = i
            break
    for i in range(len(frames)-1, -1, -1):
        if np.any(frames[i]):
            last_nonzero_index = i
            break
    if first_nonzero_index is not None and last_nonzero_index is not None:
        return frames[first_nonzero_index:last_nonzero_index+1]

def split_len(frames, split_len=64):
    """
    Split into frames of length n
    """
    num_frames = len(frames)//split_len
    for frame in range(num_frames):
        yield frames[frame*split_len:(frame+1)*split_len]

def split_silence(frames, split_len=16):
    """
    >>> arr = np.array([[1], [0], [0], [2], [0], [3], [4], [5], [0], [0], [0], [6], [0]])
    >>> [list(x) for x in split_by_silence(arr, min_len=2)]
    [[array([1])],
     [array([2]), array([0]), array([3]), array([4]), array([5])],
     [array([6]), array([0])]]
    """
    left, right, num_silence = None, None, 0
    split_frames = list()
    for index, frame in enumerate(frames):
        if np.any(frame):
            if num_silence >= split_len:
                if left is not None and right is not None:
                    split_frames.append(frames[left:right])
                left = index
            num_silence = 0
            right = index + 1
        else:
            num_silence += 1
    if num_silence >= split_len:
        split_frames.append(frames[left:right])
    else:
        split_frames.append(frames[left:])
    return [frames for frames in split_frames if not len(frames) == 0]

def play_vector(vector, repeat=True, delay=0.03):
    global out
    note_hold_mask = np.zeros(128)
    if out is None:
        out = mido.open_output("play_vector", virtual=True)
    clear()
    for row in vector:
        for note, vel in enumerate(row):
            if not note_hold_mask[note]:
                out.send(mido.Message('note_on', note=note, velocity=int(min(127, int(127*vel)))))
            if vel:
                note_hold_mask[note] = 2
        time.sleep(delay)
        for note in range(len(row)):
            if not note_hold_mask[note]:
                out.send(mido.Message('note_off', note=note))
            note_hold_mask[note] = max(0, note_hold_mask[note] - 1)
    clear()

def clear():
    for note in range(128):
        out.send(mido.Message('note_off', note=note))

def histog_vectors(vectors):
    num_simul = np.zeros(128)
    for v in vectors:
        for t in v:
            pass

def plot(vector):
    plt.imshow(np.transpose(vector))
    plt.show()

def plots(vectors):
    for index, vector in enumerate(vectors):
        plt.imshow(np.transpose(vector))
        plt.title(f"{index}")
        plt.show()

def forever(songs):
    while True:
        for i, song in enumerate(songs):
            print(f"Yo {i}")
            if np.count_nonzero(song):
                play_vector(song)