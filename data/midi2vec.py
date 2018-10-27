# midi2vec.py

"""
Usage:
vectorize('./myfile.mid', 16) -> list of vectors for each track, qtized to 16th
"""

# see also https://github.com/callaunchpad/JazzImprov/blob/master/midi_dto_csv/midi_test.ipynb
import mido
import numpy as np

def num_frames(ticks, ticks_per_beat, note_denominator):
    # 1 beat = 1 quarter
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

def split_by_silence(frames, min_len=16):
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
            if num_silence >= min_len:
                if left is not None and right is not None:
                    split_frames.append(frames[left:right])
                left = index
            num_silence = 0
            right = index + 1
        else:
            num_silence += 1
    if num_silence >= min_len:
        split_frames.append(frames[left:right])
    else:
        split_frames.append(frames[left:])
    return split_frames
    
def play(vectorized):
    for row in vectorized:
        for note, vel in enumerate(row):
            out.send(mido.Message('note_on'), note=note, velocity=int(128*vel))
        time.sleep(0.1)
        for note in range(len(row)):
            out.send(mido.Message('note_off'), note=note)

def get_ext(ext):
    def get_files(dir):
        files = []
        for sub in [os.path.join(dir, sub) for sub in os.listdir(dir)]:
            if os.path.isdir(sub):
                files += get_files(sub)
            elif os.path.splitext(sub)[1] == ext:
                files.append(sub)
        return files
    return get_files