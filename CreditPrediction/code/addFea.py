# coding = utf-8
import data
import numpy as np
import csv


#  加载数据
def loadData():
    train_xy = data.load_train_oneHot()  # 包括uid,feature,y
    sumTrainFea = getSumMiss(train_xy, train_xy.shape[0], 1)
    train_xy = np.insert(train_xy, [train_xy.shape[1] - 1], sumTrainFea, 1)

    test_x = data.load_test_oneHot()
    sumTestFea = getSumMiss(test_x, test_x.shape[0], 0)
    test_x = np.insert(test_x, [test_x.shape[1]], sumTestFea, 1)

    writeToTrain(train_xy)
    writeToTest(test_x)

#  根据特征的缺失值对缺失特征的个数进行统计


def getSumMiss(X, row, flag):
    numFea = 0
    if flag == 1:
        numFea = X.shape[1] - 1
    else:
        numFea = X.shape[1]
    sumFea = np.zeros(shape=(row, 2))
    for row in range(X.shape[0]):
        temp = 0
        for column in range(numFea):
            if X[row, column] == -1:
                temp = temp + 1
        sumFea[row, 0] = temp
        sumFea[row, 1] = numFea - temp
    return sumFea


def writeToTrain(train_xy):
    variableName = np.array(
        [['x' + str(c + 1) for c in range(train_xy.shape[1] - 2)]])
    variableName = np.insert(variableName, [0], ['uid'], axis=1)
    _, v_c = variableName.shape
    variableName = np.insert(variableName, [v_c], ['y'], axis=1)

    testFile = open('train_x_oneHot_miss.csv', 'w', newline='')
    writer = csv.writer(testFile)
    writer.writerows(variableName)
    writer.writerows(train_xy)
    testFile.close()


def writeToTest(test_x):
    variableName = np.array(
        [['x' + str(c + 1) for c in range(test_x.shape[1] - 1)]])
    variableName = np.insert(variableName, [0], ['uid'], axis=1)

    testFile = open('test_x_oneHot_miss.csv', 'w', newline='')
    writer = csv.writer(testFile)
    writer.writerows(variableName)
    writer.writerows(test_x)
    testFile.close()

if __name__ == '__main__':
    loadData()
    print('done!')
