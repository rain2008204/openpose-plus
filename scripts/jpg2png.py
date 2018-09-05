#!/usr/bin/env python3

import glob

from cv2 import imread, imwrite


def jpg2png(name):
    image = imread(name)
    new_name = name.replace('.jpg', '.png')
    imwrite(new_name, image)


for f in glob.glob('data/media/*.jpg'):
    jpg2png(f)
