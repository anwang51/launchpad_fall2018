import midi_proc
import numpy as np

songs = open("songs.txt")
songs = songs.read().split("\n")[:-1]
tmp = []
i = 0
for row in songs:
	# print(i)
	row = row.split(",")[:-1]
	row = [float(e) for e in row]
	tmp.append(np.reshape(row, (512, 128)))
	i += 1
	
songs = tmp
