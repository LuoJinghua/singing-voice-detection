#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import os
import shutil
import time

import numpy as np
import tensorflow as tf

from config import EvalConfig, TrainConfig, ModelConfig
from data import Data
from model import Model
from preprocess import to_spectrogram, get_magnitude, to_db
from utils import Diff


def eval_model(model, eval_data, sess):
    mixed_wav, src1_wav, src2_wav, _ = eval_data.next_wavs(EvalConfig.SECONDS, EvalConfig.NUM_EVAL)

    mixed_spec = to_spectrogram(mixed_wav)
    mixed_mag = get_magnitude(mixed_spec)

    src1_spec, src2_spec = to_spectrogram(src1_wav), to_spectrogram(src2_wav)
    src1_mag, src2_mag = get_magnitude(src1_spec), get_magnitude(src2_spec)
    src2_label = np.array(np.greater_equal(to_db(src2_mag), -25), dtype=np.float32)

    src1_batch, _ = model.spec_to_batch(src1_mag)
    src2_batch, _ = model.spec_to_batch(src2_mag)
    mixed_batch, _ = model.spec_to_batch(mixed_mag)
    src2_label_batch, _ = model.vad_to_batch(src2_label)

    pred_y = sess.run(tf.sigmoid(model()),
                      feed_dict={model.x_mixed: mixed_batch,
                                 model.y_src1: src1_batch,
                                 model.y_src2: src2_batch,
                                 model.y_src2_label: src2_label_batch})
    pred_y = model.batch_to_vad(pred_y, EvalConfig.NUM_EVAL)
    pred_y = np.array(np.greater(pred_y, 0.5), dtype=np.float32)

    pred_y = np.reshape(pred_y, (-1, 1))
    y = np.reshape(src2_label, (-1, 1))

    tp = pred_y * y
    fp = pred_y * (1 - y)
    fn = (1 - pred_y) * y

    precesion = np.sum(tp) / (np.sum(tp) + np.sum(fp) + np.finfo(np.float32).eps)
    recall = np.sum(tp) / (np.sum(tp) + np.sum(fn) + np.finfo(np.float32).eps)

    # result = np.hstack([np.reshape(src2_label, (-1, 1)), np.reshape(pred_y, (-1, 1))])
    # np.savetxt("eval.csv", result, delimiter=",", fmt='%g')
    return precesion, recall


# TODO multi-gpu
def train():
    # Model
    model = Model(ModelConfig.HIDDEN_LAYERS, ModelConfig.HIDDEN_UNITS)

    # Loss, Optimizer
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    loss_fn = model.loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=TrainConfig.LR).minimize(loss_fn, global_step=global_step)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=TrainConfig.LR).minimize(loss_fn, global_step=global_step)

    model.precesion = tf.placeholder(dtype=tf.float32, shape=(), name='precesion')
    model.recall = tf.placeholder(dtype=tf.float32, shape=(), name='recall')

    # Summaries
    summary_ops = summaries(model, loss_fn)

    with tf.Session(config=TrainConfig.session_conf) as sess:

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        model.load_state(sess, TrainConfig.CKPT_PATH)

        print('num trainable parameters: %s' % (
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        writer = tf.summary.FileWriter(TrainConfig.GRAPH_PATH, sess.graph)

        # Input source
        data = Data(TrainConfig.DATA_PATH)
        eval_data = Data(EvalConfig.DATA_PATH)

        loss = Diff()
        precesion, recall = 0, 0
        intial_global_step = global_step.eval()
        for step in xrange(intial_global_step, TrainConfig.FINAL_STEP):
            start_time = time.time()

            eval_metric = step % 20 == 0 or step == intial_global_step
            if eval_metric:
                precesion, recall = eval_model(model, eval_data, sess)

            mixed_wav, src1_wav, src2_wav, _ = data.next_wavs(TrainConfig.SECONDS, TrainConfig.NUM_WAVFILE)

            mixed_spec = to_spectrogram(mixed_wav)
            mixed_mag = get_magnitude(mixed_spec)
            src2_spec = to_spectrogram(src2_wav)

            src1_spec, src2_spec = to_spectrogram(src1_wav), to_spectrogram(src2_wav)
            src1_mag, src2_mag = get_magnitude(src1_spec), get_magnitude(src2_spec)
            src2_label = np.array(np.greater_equal(to_db(src2_mag), -25), dtype=np.float32)

            src1_batch, _ = model.spec_to_batch(src1_mag)
            src2_batch, _ = model.spec_to_batch(src2_mag)
            mixed_batch, _ = model.spec_to_batch(mixed_mag)
            src2_label_batch, _ = model.vad_to_batch(src2_label)

            l, _, summary = sess.run([loss_fn, optimizer, summary_ops],
                                     feed_dict={model.x_mixed: mixed_batch,
                                                model.y_src1: src1_batch,
                                                model.y_src2: src2_batch,
                                                model.y_src2_label: src2_label_batch,
                                                model.precesion: precesion,
                                                model.recall: recall})
            loss.update(l)
            writer.add_summary(summary, global_step=step)

            # Save state
            if step % TrainConfig.CKPT_STEP == 0:
                tf.train.Saver().save(sess, TrainConfig.CKPT_PATH + '/checkpoint', global_step=step)

            elapsed_time = time.time() - start_time
            print('step-{}\ttime={:2.2f}\td_loss={:2.2f}\tloss={:2.3f}\t'.format(step,
                                                                                 elapsed_time,
                                                                                 loss.diff * 100,
                                                                                 loss.value))

        writer.close()


def summaries(model, loss):
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.summary.histogram(v.name, v)
        tf.summary.histogram('grad/' + v.name, tf.gradients(loss, v))
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('x_mixed', model.x_mixed)
    tf.summary.histogram('y_src1', model.y_src1)
    tf.summary.histogram('y_src2', model.y_src1)
    tf.summary.scalar('precesion', model.precesion)
    tf.summary.scalar('recall', model.recall)
    return tf.summary.merge_all()


def setup_path():
    if TrainConfig.RE_TRAIN:
        if os.path.exists(TrainConfig.CKPT_PATH):
            shutil.rmtree(TrainConfig.CKPT_PATH)
        if os.path.exists(TrainConfig.GRAPH_PATH):
            shutil.rmtree(TrainConfig.GRAPH_PATH)
    if not os.path.exists(TrainConfig.CKPT_PATH):
        os.makedirs(TrainConfig.CKPT_PATH)


if __name__ == '__main__':
    from optparse import OptionParser

    usage = """%prog"""
    parser = OptionParser(usage=usage)
    parser.add_option('--hidden-units', dest='hidden_units', default=ModelConfig.HIDDEN_UNITS, type=int,
                      help='the hidden units per GRU cell')
    parser.add_option('--hidden-layers', dest='hidden_layers', default=ModelConfig.HIDDEN_LAYERS, type=int,
                      help='the hidden layers of network')
    parser.add_option('--case-name', dest='case_name', default=TrainConfig.CASE,
                      help='the name of this setup')
    (options, args) = parser.parse_args()
    ModelConfig.HIDDEN_UNITS = options.hidden_units
    ModelConfig.HIDDEN_LAYERS = options.hidden_layers
    TrainConfig.CASE = options.case_name
    setup_path()
    train()
