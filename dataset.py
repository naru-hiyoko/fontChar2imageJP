import numpy as np
import sys
from os.path import join, exists
from progressbar import ProgressBar, Percentage, Bar
import logging
import cPickle

""" train.py 実行時に必要 """
def load_dataset():
    prefix = '../data'
    data = None
    labels = np.asarray([], dtype=np.int32)
    with open('label.txt') as f: 
        classlabel = f.readlines()
    prog = ProgressBar()
    prog.max_value = len(classlabel)
    prog.start()

    for line in classlabel:
        id, c = line.split(' ')
        id = np.int32(id)
        #if id == 500:
        #    break
        prog.update(id + 1)
        pklfile = join(prefix, 'pkl', 'data_{}.pkl'.format(id))
        assert exists(pklfile), 'PKL FILE NOT FOUND'
        with open(pklfile, 'r') as f:
            pkl = cPickle.load(f)
            if data is None:
                data = pkl['data']
            else:
                data = np.append(data, pkl['data'])
            labels = np.append(labels, pkl['labels'])
            
    prog.finish()
    data = data.reshape(-1, 1, 48, 48)
    return data, labels
