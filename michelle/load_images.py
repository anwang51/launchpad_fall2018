import skimage.io as skio
import numpy as np
import os


def load_images():
	imgs = []
	folder = "/Users/wangan/Documents/launchpad_githubs/launchpad_fall2018/michelle/"
	for filename in os.listdir(folder):
		if filename.endswith(".jpg"):
			imgs.append(skio.imread(folder + filename))
	return imgs