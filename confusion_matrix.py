# encoding: utf-8

import numpy as np
import sys
from os.path import join, exists
from progressbar import ProgressBar, Percentage, Bar
import logging


import chainer
from chainer import Function, FunctionSet, Variable, optimizers, serializers, gradient_check, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import dataset

import cPickle

""" network definition """

model = FunctionSet(
    conv1 = F.Convolution2D(1, 64, 5, stride=1, pad=0),
    conv2 = F.Convolution2D(64, 128, 5, stride=1, pad=0),
    conv3 = F.Convolution2D(128, 512, 3, stride=1, pad=0),
    conv4 = F.Convolution2D(512, 1024, 2, stride=1, pad=0),
    ip1 = L.Linear(36864, 5000),
    ip2 = L.Linear(5000, 799),
)

def forward(x_data, y_data, train=False):
    x, t = Variable(x_data), chainer.Variable(y_data)
    
    h = model.conv1(x)
    h = F.average_pooling_2d(h, 5, stride=2)

    h = model.conv2(h)
    h = F.average_pooling_2d(h, 5, stride=1)

#    h = F.batch_normalization(h)
    
    h = model.conv3(h)
    h = F.average_pooling_2d(h, 3, stride=1)
    
    h = model.conv4(h)
    h = F.average_pooling_2d(h, 2, stride=1)

    h = model.ip1(h)
    h = F.relu(h)
    h = model.ip2(h)
    
    y = h
    prob = F.softmax(y)
    prob = prob.data.sum(axis=0) / np.float32(prob.data.shape[0])
    print prob.shape


    """ softmax normalization + compute loss & validation """
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t), prob


""" test """    
serializers.load_npz('../data/snapshot/trained_1.model', model)
matrix = []

f = open('./label.txt', 'r')
chars = f.readlines()
f.close()

id = 50
prefix = './data'
pklfile = join(prefix, 'data_{}.pkl'.format(id))
assert exists(pklfile), 'pkl was not found!'


f = open(pklfile, 'r')
pkl = cPickle.load(f)
f.close()
data = pkl['data']
label = pkl['labels']

loss, acc, prob = forward(data, label)
ans = sorted(enumerate(prob), key=lambda x: x[1], reverse=True)
print chars[id]
for i in range(5):
    id = ans[i][0]
    print chars[id].rstrip()
