# Fahad Alsehami
# Age Estimation from the Adult Pubic Symphysis project

import os

# image file types the pipeline reads
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(root):
	"""Yield the path of every image file under root, walking subfolders in a stable order."""
	for folder, subfolders, names in os.walk(root):
		subfolders.sort()
		for name in sorted(names):
			if name.lower().endswith(IMAGE_EXTENSIONS):
				yield os.path.join(folder, name)
