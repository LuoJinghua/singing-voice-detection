#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import librosa
import numpy as np
import tensorflow as tf

from config import EvalConfig, ModelConfig
from model import Model
from preprocess import to_spectrogram, get_magnitude, get_phase


def decode_input(filename):
    data, rate = librosa.load(filename, mono=False, sr=ModelConfig.SR)

    print ('channels: %d samples: %d' % data.shape)

    n_channels = data.shape[0]
    total_samples = data.shape[1]
    result = []
    for ch in range(n_channels):
        result.append(np.array([data[ch, :]]).flatten())
    return total_samples, data, np.array(result, dtype=np.float32)


def separate(filename, channel):
    with tf.Graph().as_default():
        # Model
        model = Model(ModelConfig.HIDDEN_LAYERS, ModelConfig.HIDDEN_UNITS)
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        total_samples, origin_samples, samples = decode_input(filename)
        channels = origin_samples.shape[0]
        with tf.Session(config=EvalConfig.session_conf) as sess:
            # Initialized, Load state
            sess.run(tf.global_variables_initializer())
            model.load_state(sess, EvalConfig.CKPT_PATH)

            mixed_wav, src1_wav, src2_wav = samples, samples, samples

            mixed_spec = to_spectrogram(mixed_wav)
            mixed_mag = get_magnitude(mixed_spec)
            mixed_batch, padded_mixed_mag = model.spec_to_batch(mixed_mag)
            mixed_phase = get_phase(mixed_spec)

            pred_y = sess.run(tf.sigmoid(model()), feed_dict={model.x_mixed: mixed_batch})
            pred_y = model.batch_to_vad(pred_y, samples.shape[0])
            result = np.dstack(pred_y)[0]
            return result
    return None


def extract(filename, channel):
    result = separate(filename, channel)

    base_file_name = os.path.splitext(filename)[0]
    np.savetxt(base_file_name + '.csv', result, delimiter=",", fmt='%g')


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage='%prog music.wav')
    parser.add_option('-c', dest='channel', default=-1, type=int,
                      help="extract voice from specified channel, -1 to extract all channels")
    parser.add_option('-p', dest='check_point', default=EvalConfig.CKPT_PATH,
                      help="the path to checkpoint")
    parser.add_option('--hidden-units', dest='hidden_units', default=ModelConfig.HIDDEN_UNITS, type=int,
                      help='the hidden units per GRU cell')
    parser.add_option('--hidden-layers', dest='hidden_layers', default=ModelConfig.HIDDEN_LAYERS, type=int,
                      help='the hidden layers of network')
    parser.add_option('--case-name', dest='case_name', default=EvalConfig.CASE,
                      help='the name of this setup')
    options, args = parser.parse_args()
    if options.check_point:
        EvalConfig.CKPT_PATH = options.check_point
    ModelConfig.HIDDEN_UNITS = options.hidden_units
    ModelConfig.HIDDEN_LAYERS = options.hidden_layers
    EvalConfig.CASE = options.case_name
    for arg in args:
        extract(arg, options.channel)
