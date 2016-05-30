# encoding: utf-8

import numpy as np
import sys
from os.path import join
from progressbar import ProgressBar, Percentage, Bar
import logging


import chainer
from chainer import Function, FunctionSet, Variable, optimizers, serializers, gradient_check, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import cuda

import dataset

""" load dataset """
train_data, train_labels = dataset.load_dataset()

""" network definition """
class myNet(chainer.Chain):
    def __init__(self):
        super(myNet, self).__init__(
            conv1 = F.Convolution2D(1, 20, 5),
            norm1 = F.BatchNormalization(20),
            conv2 = F.Convolution2D(20, 50, 5),
            norm2 = F.BatchNormalization(50),
            ip1 = F.Linear(4050, 500),
            ip2 = F.Linear(500, 100),
        )
        self.train = True
        

    def __call__(self, x_data, y_data):
        x, t = Variable(cuda.to_gpu(x_data)), chainer.Variable(cuda.to_gpu(y_data))
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.ip1(h)
        h = F.relu(h)
        h = self.ip2(h)
        y = h
        """ softmax + compute loss & validation """
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

""" training configuration """
iteration = 50
batchsize = 200
N = train_data.shape[0]
model = myNet()
model.to_gpu()
optim = optimizers.Adam()
optim.setup(model)
logging.basicConfig(filename='train.log', filemode='w', level=logging.DEBUG)

""" training """

for epoch in range(iteration):
    perm = np.random.permutation(N)
    progress = ProgressBar()
    progress.min_value = 0
    progress.max_value = N
    progress.start()
    sum_loss = 0.0
    sum_accuracy = 0.0
    
    for i in range(0, N, batchsize):
        
        data_batch = train_data[perm[i:i+batchsize]]
        label_batch = train_labels[perm[i:i+batchsize]]
        optim.zero_grads()
        loss, accuracy = model(data_batch, label_batch)
        loss.backward()
        optim.update()
        status = 'epoch: {} , loss: {:0.4f}, acc: {:0.4f}'.format(epoch, float(loss.data), float(accuracy.data))
        progress.widgets = [status, Percentage()]
        progress.update(i+1)
        
        sum_loss += loss.data
        sum_accuracy += accuracy.data

    status = ' | epoch: {}, loss: {:0.4f}, acc: {:0.4f}'.format(epoch+1, float(sum_loss * batchsize / N), float(sum_accuracy * batchsize / N))
    logging.info(status)

    prefix = '../data/snapshot'
    serializers.save_npz(join(prefix, 'trained_%d.model' % (epoch+1)), model)
    serializers.save_npz(join(prefix, 'state_%d.state' % (epoch+1)), optim)
