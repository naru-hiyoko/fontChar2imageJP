import numpy as np
import sys
from os.path import join, exists
from progressbar import ProgressBar, Percentage, Bar
import logging
import cPickle

def load_dataset():
    prefix = '../data'
    data = None
    labels = np.asarray([], dtype=np.int32)
    
    f = open('./label.txt')
    classlabel = f.readlines()
    f.close()

    prog = ProgressBar()
    prog.min_value = 0
    prog.max_value = len(classlabel)
    prog.start()

    for line in classlabel:
        id, c = line.split(' ')
        id = np.int32(id)
        if id == 100:
            break
        prog.update(id + 1)
        pklfile = join(prefix, 'data_{}.pkl'.format(id))
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
