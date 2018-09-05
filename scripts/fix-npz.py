#!/usr/bin/env python3

import os

import numpy as np

filename = os.path.join(os.getenv('HOME'), 'Downloads/vgg450000.npz')
new_filename = os.path.join(os.getenv('HOME'), 'Downloads/vgg450000_no_cpm.npz')

data = np.load(filename)

new_data = dict()
for name, t in data.items():
    new_name = name.replace('cpm/', '')
    new_data[new_name] = t

np.savez(new_filename, **new_data)
