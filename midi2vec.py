"""
Usage:
vectorize('./myfile.mid', 16) -> list of vectors for each track, qtized to 16th
"""

# see also https://github.com/callaunchpad/JazzImprov/blob/master/midi_to_csv/midi_test.ipynb
import mido
import numpy as np

def num_frames(ticks, ticks_per_beat, note_denominator):
    return int((note_denominator/4 * ticks) // ticks_per_beat)

def should_parse_as_note(event, drum_mode):
    if event.type != 'note_on' and event.type != 'note_off':
        return False
    if drum_mode:
        return event.channel == 9
    return event.channel != 9

def vectorize_path(path, frame_dur=16, drum_mode=False, merge_tracks=False):
    midi = mido.MidiFile(path)
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
        if should_parse_as_note(event, drum_mode):
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