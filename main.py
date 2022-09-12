# -*- coding: utf-8 -*-
# @Time    : 2022/9/9 17:19
# @Author  : LuMing
# @File    : main.py
# @Software: PyCharm 
# @Comment :
import numpy as np

import bpnn
import data_load
from bpnn import Bpnn


def binarize(labels):
    bin_labels = []
    for label in labels:
        bin_list = []
        for i in range(10):
            if i == label:
                bin_list.append(1)
            else:
                bin_list.append(0)
        bin_labels.append(bin_list)
    return bin_labels


def unidimensional(images):
    unidimensional_images = []
    for i in range(len(images)):
        image = np.array(images[i]).reshape(1, -1)[0]
        unidimensional_images.append(image / 255)
    return unidimensional_images


def main():
    train_images = data_load.load_train_images()
    train_labels = data_load.load_train_labels()
    test_images = data_load.load_test_images()
    test_labels = data_load.load_test_labels()

    train_labels = binarize(train_labels)
    train_images = unidimensional(train_images)
    # test_labels = binarize(test_labels)
    test_images = unidimensional(test_images)
    u_bpnn = Bpnn(28 * 28, 280, 10)

    # 训练
    u_bpnn.train(train_images[0:60000], train_labels[0:60000])

    # test
    u_bpnn.test(test_images, test_labels)
    # bpnn.my_test()


if __name__ == "__main__":
    main()
