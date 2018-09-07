#!/usr/bin/env python3

import argparse
import os
import sys
import time

import tensorflow as tf
import tensorlayer as tl

from inference.common import measure, plot_humans, read_imgfile
from inference.estimator2 import TfPoseEstimator as TfPoseEstimator2
from models import get_full_model_func

tf.logging.set_verbosity(tf.logging.INFO)
tl.logging.set_verbosity(tl.logging.INFO)


def inference(base_model_name, path_to_npz, input_files, plot):
    full_model = get_full_model_func(base_model_name)
    e = measure(lambda: TfPoseEstimator2(path_to_npz, full_model, target_size=(432, 368)), 'create TfPoseEstimator2')

    t0 = time.time()
    for idx, img_name in enumerate(input_files):
        image = measure(lambda: read_imgfile(img_name, None, None), 'read_imgfile')
        humans = measure(lambda: e.inference(image, resize_out_ratio=8.0), 'e.inference')
        tl.logging.debug('got %d humans from %s' % (len(humans), img_name))
        if humans:
            for h in humans:
                tl.logging.debug(h)
            if plot:
                plot_humans(e, image, humans, '%02d' % (idx + 1))
    tot = time.time() - t0
    tl.logging.info('inference all took: %f, mean: %f' % (tot, tot / len(input_files)))


def parse_args():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--path-to-npz', type=str, default='', help='path to npz', required=True)
    parser.add_argument('--images', type=str, default='', help='comma separate list of image filenames', required=True)
    parser.add_argument('--base-model', type=str, default='vgg', help='vgg | mobilenet')
    parser.add_argument('--plot', type=bool, default=False, help='draw the results')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    image_files = [f for f in args.images.split(',') if f]
    inference(args.base_model, args.path_to_npz, image_files, args.plot)
