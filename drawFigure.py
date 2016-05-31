# encoding: utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt


acc = []
loss = []

with open(sys.argv[1], 'r') as f:
    for i, line in enumerate(f.readlines()):
        content = line.rstrip().split('|')[1].split(',')
        loss.append(float(content[1].split(':')[1]))
        acc.append(float(content[2].split(':')[1]))
    N = len(loss)
    loss = np.asarray(loss)
    acc = np.asarray(acc)
    x = range(1, N+1, 1)
    plt.ylim([0.5, 1.0])
    plt.subplot(1, 2, 1)
    plt.plot(x, acc)
    plt.subplot(1, 2, 2)
    plt.plot(x, loss)

    plt.show()
