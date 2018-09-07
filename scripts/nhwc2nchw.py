#!/usr/bin/env python3

import os

import numpy as np


def nhwc2nchw(filename, new_filename):
    """Convert model from NHWC to NCHW."""

    def rscd2dcrs(t):
        """Convert convolution kennel.

        For weight, it is actually RSCD to DCRS.

        in NHWC format
            conv :: [N, H, W, C], [R, S, C, D] -> [N, H, W, D]
        in NCHW format
            conv :: [N, C, H, W], [D, C, R, S] -> [N, D, H, W]

        where [R, S] is the filter size, [C, D] = [in_c, out_c].
        """

        r, s, c, d = t.shape
        print('%s -> %s' % ((r, s, c, d), (c, d, r, s)))
        return t.transpose([3, 2, 0, 1])

    data = np.load(filename)
    new_data = dict()

    for name, t in data.items():
        if len(t.shape) == 4:
            print('converting %s, assuming it is a conv weight' % name)
            t = rscd2dcrs(t)
        else:
            print('NOT convert %s, which has shape %s' % (name, t.shape))
        new_data[name] = t

    np.savez(new_filename, **new_data)


filename = os.path.join(os.getenv('HOME'), 'Downloads/vgg450000_no_cpm.npz')
new_filename = os.path.join(os.getenv('HOME'), 'Downloads/vgg450000_no_cpm_nchw.npz')

nhwc2nchw(filename, new_filename)
