import midi_proc
import numpy as np

# Good songs
# 0, 1, 3, 4, 5, 6, 8, 9, 10, 14, 15, 16, 18, 19, 20, 24, 25, 26, 28, 29, 30...

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
