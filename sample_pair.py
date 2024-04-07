
import matplotlib
matplotlib.use('AGG')

import random

import numpy as np
num_class=2

def create_pairs(x_left,x_right, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''

    data_left = []
    data_right = []
    pair_label = []
    class_label = []
    for d in range(num_class):
        for i in range(len(digit_indices[d])):
            z1 = digit_indices[d][i]
            data_left += [x_left[z1]]
            data_right += [x_right[z1]]
            class_label += [[d, d]]
            # **************构建负样本***************
            inc = random.randrange(1, num_class)
            dn = (d + inc) % num_class
            j = random.randint(0, len(digit_indices[dn])-1)
            z1, z2 = digit_indices[d][i], digit_indices[dn][j]
            data_left += [x_left[z1]]
            data_right += [x_right[z2]]
            class_label += [[d, dn]]
            pair_label += [1,0]
    return np.array(data_left), np.array(data_right), np.array(class_label), np.array(pair_label)