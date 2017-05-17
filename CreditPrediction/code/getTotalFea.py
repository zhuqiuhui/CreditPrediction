# coding = utf-8
import data
import numpy as np
import csv


def getTotalFea():
    trainFea = np.zeros(shape=(15000, 1))
    train_xy = data.load_train_oneHot()
    for row1 in range(train_xy.shape[0]):
        temp1 = 0
        for column1 in range(train_xy.shape[1]):
            if train_xy[row1, column1] == -1:``
                temp1 = temp1 + 1
        trainFea[row1, 0] = temp1

    testFea = np.zeros(shape=(5000, 1))
    test_x = data.load_test_oneHot()
    for row2 in range(test_x.shape[0]):
        temp2 = 0
        for column2 in range(test_x.shape[1]):
            if test_x[row2, column2] == -1:
                temp2 = temp2 + 1
        testFea[row2, 0] = temp2
    with open('missingValue/singleTrainFea.csv', 'w', encoding='utf-8') as f:
        for i in range(15000):
            f.write('%d\n' % (trainFea[i, 0]))
    with open('missingValue/singleTestFea.csv', 'w', encoding='utf-8') as f:
        for j in range(5000):
            f.write('%d\n' % (testFea[j, 0]))


if __name__ == '__main__':
    getTotalFea()
    print('done!')
