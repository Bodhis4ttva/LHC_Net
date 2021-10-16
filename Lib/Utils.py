from abc import ABC

import os
import random
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile
import csv
from classification_models.tfkeras import Classifiers
import gc


def get_data(filename):
    csvfile = open(filename)
    reader = csv.reader(csvfile, delimiter=';')
    next(reader)
    data = []
    for row in reader:
        item = [row[0], row[2:]]
        data.append(item)
    images = np.zeros((len(data), 48, 48, 1), dtype='float32')
    labels = np.zeros((len(data)), dtype='float32')
    labels_full = np.zeros(shape=(len(data), 7), dtype='float32')
    for i in range(len(data)):
        images[i, :, :, :] = np.array(data[i][1]).reshape((48, 48, 1))
        labels[i] = np.array(data[i][0]).astype('float32')
        labels_full[i, int(labels[i])] = 1
    return images, labels_full


class cb3(tf.keras.callbacks.Callback):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.reports = []

    def on_epoch_end(self, epoch, logs={}):
        report = tf.keras.metrics.CategoricalAccuracy()(self.y, self.model.predict(self.x)).numpy()
        self.reports.append(report)
        print("Test Accuracy", report)
        print("")
        return


def augment(images, params):

    y = images

    if params['flip']:
        y = tf.image.flip_left_right(image=y)

    if params['zoom'] > 0 and params['zoom'] < 1:
        y = tf.image.central_crop(image=y,
                                  central_fraction=params['zoom'])
        y = tf.image.resize(images=y,
                            size=[images.shape[1], images.shape[2]],
                            method='bilinear',
                            preserve_aspect_ratio=False)

    if params['shift_h'] != 0 or params['shift_v'] != 0:
        y = tfa.image.translate(images=y,
                                translations=[params['shift_h'], params['shift_v']],
                                interpolation='bilinear',
                                fill_mode='nearest')
    if params['rot'] != 0:
        y = tfa.image.rotate(images=y,
                             angles=params['rot'],
                             interpolation='bilinear',
                             fill_mode='nearest')

    return y


def TTA_Inference(model, x):
    pred_test = model.predict(x)
    zooms = [1]  # 2
    rotations = [0, 0.4, -0.4]  # 5
    shifts_h = [0, 10, -10]  # 3
    shifts_v = [0, 10, -10]  # 3
    flips = [False, True]  # 2

    default_prediction_weight = 3
    count = default_prediction_weight
    predictions = default_prediction_weight*pred_test

    for i1 in range(len(zooms)):
        for i2 in range(len(rotations)):
            for i3 in range(len(shifts_h)):
                for i4 in range(len(shifts_v)):
                    for i5 in range(len(flips)):
                        params = {'zoom': zooms[i1],
                                  'rot': rotations[i2],
                                  'shift_h': shifts_h[i3],
                                  'shift_v': shifts_v[i4],
                                  'flip': flips[i5]}
                        if params['zoom'] < 1 or params['rot'] != 0 or params['shift_h'] != 0 or params['shift_v'] != 0 or params['flip']:

                            count = count + 1
                            d = augment(x, params)
                            preds = model.predict(d, batch_size=128)
                            predictions = predictions + preds
                            del d
                            del preds
                            del params
                            gc.collect()
                            gc.collect()
                            gc.collect()

    Params = [[0.9, 0, 0, 0, False],
              [0.9, 0, 0, 0, True],
              [0.9, 0.15, 0, 0, False],
              [0.9, 0.15, 0, 0, True],
              [0.9, -0.15, 0, 0, False],
              [0.9, -0.15, 0, 0, True]]

    for i in range(len(Params)):
        params = {'zoom': Params[i][0],
                  'rot': Params[i][1],
                  'shift_h': Params[i][2],
                  'shift_v': Params[i][3],
                  'flip': Params[i][4]}
        count = count + 1
        d = augment(x, params)
        preds = model.predict(d, batch_size=128)
        predictions = predictions + preds

        del d
        del preds
        del params
        gc.collect()
        gc.collect()
        gc.collect()

    predictions = predictions / count
    return predictions


def Check_Unique(x):
    lose = 0
    for i in range(x.shape[0]):
        if sum(x[i, :] == x[i, :].max()) > 1:
            lose = lose + 1
    return lose

