# Based in part on the MIDI to CSV from for Jonathan's project (https://github.com/callaunchpad/JazzImprov/blob/master/midi_to_csv/midi_test.ipynb)

import mido
import numpy as np

def num_notes(ticks, ticks_per_beat, note_denominator):
    """Returns the floored number of notes that an event of duration ticks
    fits into. For instance, note_denominator=16 means you find the number
    of 16th notes, so with 4 ticks per beat, 1 tick = 1/4 beat = 1 16th note"""
    
    return int((note_denominator/4 * ticks) // ticks_per_beat)

def vectorize(midi, granularity):
    """Vectorize the contents of a midi file.
    Granularity = 16 to vectorize with 16th note timestep, etc.
    The end encoding is a 2d array with each row = the velocity (0.0-1.0) of
    each MIDI note (index 0-127) for a given time.
    
    The current implementation will just skip notes that are 
    shorter than granularity, so pick it carefullY!"""
    
    frames = list()
    current_frame = np.zeros(128) # Keep track of what notes are on
    # If we iterated over the midi itself, times would not be in ticks.
    for event in mido.merge_tracks(midi.tracks):
        if event.time:
            # We 'advance forward' in time by adding current frame.
            # So if we advanced by 1/8 note with 1/16 note granularity
            # We would just append the current state twice.
            for _ in range(num_notes(event.time, midi.ticks_per_beat, granularity)):
                frames.append(np.array(current_frame))
        if event.type == 'note_on' or event.type == 'note_off':
            # Taking advantage of fact that note_off has velocity of 0
            current_frame[event.note] = event.velocity / 127.0
    return np.vstack(frames)