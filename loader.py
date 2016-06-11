#!/usr/local/bin/python2.7
# encoding: utf-8

from os.path import join, exists
import cPickle

import numpy as np

import chainer
from chainer import Function, FunctionSet, Variable, optimizers, serializers, gradient_check, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from progressbar import ProgressBar

import matplotlib.pyplot as plt


model = FunctionSet(
    conv1 = F.Convolution2D(1, 20, 5),
    norm1 = F.BatchNormalization(20),
    conv2 = F.Convolution2D(20, 50, 5),
    norm2 = F.BatchNormalization(50),
    ip1 = F.Linear(4050, 1000),
    ip2 = F.Linear(1000, 799),
)

def forward(x_data, y_data, train=False, normalized=False):
    x, t = Variable(x_data), chainer.Variable(y_data)
    h = model.conv1(x)
    h = model.norm1(h)
    h = F.relu(h)
    h = F.max_pooling_2d(h, 3, stride=2)
    h = model.conv2(h)
    h = model.norm2(h)
    h = F.relu(h)
    h = F.max_pooling_2d(h, 3, stride=2)
    h = model.ip1(h)
    h = F.relu(h)
    h = model.ip2(h)
    y = h

    if normalized:
        prob = F.softmax(y)
        prob = prob.data.sum(axis=0) / np.float32(prob.data.shape[0])
    else:
        prob = y.data.sum(axis=0) / np.float32(y.data.shape[0])

    return prob


def load():
    copus = dict()
    chars = dict()
    serializers.load_npz('../data/snapshot/trained_100.model', model)    
    
    with open('label.txt') as f:
        for line in f.readlines():
            i, char = line.split(' ')
            i = int(i)
            char = unicode(char, 'utf-8').rstrip()
            id = ord(char)
            """ ユニコードをキーとして参照 """
            copus[id] = i
            """ label.txt に基づいた参照 """
            chars[i] = char
            
    return copus, chars


prefix = '../data'

def confusion_matrix(chars):
    matrix = []
    # 0,1,2と見ていくのは危険かもしれない
    for i in range(len(chars.keys())):
        pklfile = join(prefix, 'data_{}.pkl'.format(i))
        assert exists(pklfile), 'pkl was not found!'
        print chars[i],
        print ' : ', 
        with open(pklfile, 'r') as f2:
            pkl = cPickle.load(f2)
            data = pkl['data']
            label = pkl['labels']
            prob = forward(data, label, normalized=True)
            matrix.append(prob)
            ret = showTop5(prob, chars)
            print ret[0], ret[1], ret[2], ret[3], ret[4]
    
    matrix = np.vstack(matrix)
    print matrix.shape
    plt.title('confusion matrix')
    plt.imshow(matrix, interpolation='none')
    plt.show()

def computeVec(chars):
    features = dict()
    N = len(chars.keys())
    prog = ProgressBar()
    prog.max_value = N
    prog.start()
    for i in chars.keys():
        prog.update(int(i)+1)
        pklfile = join(prefix, 'data_{}.pkl'.format(i))
        assert exists(pklfile), 'pkl was not found!'
        with open(pklfile, 'r') as f2:
            pkl = cPickle.load(f2)
            data = pkl['data']
            label = pkl['labels']
            prob = forward(data, label)
            features[i] = prob
    prog.finish()
    return features

def showTop5(prob, chars):
    ans = sorted(enumerate(prob), key=lambda x: x[1], reverse=True)
    topK = []
    for i in range(5):
        id = ans[i][0]
        topK.append(chars[id])
    return topK


def setup():
    features_file = 'features.pkl'
    copus, chars = load()
    if not exists(features_file):
        features = computeVec(chars)
        data = { 'copus' : copus,
                 'chars' : chars,
                 'features' : features,
        }
        with open(features_file, 'w') as f:
            cPickle.dump(data, f, -1)
    else:
        with open(features_file, 'rb') as f:
            data = cPickle.load(f)

    features = data['features']
    copus = data['copus']
    chars = data['chars']
    return (copus, features, chars)

if __name__ == '__main__':
    copus, chars = load()
    confusion_matrix(chars)
    #setup()
