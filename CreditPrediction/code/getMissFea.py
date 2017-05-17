# coding = utf-8
import data
import numpy as np
import csv


#  加载数据
def loadData():
    train_xy = data.load_train_oneHot()  # 包括uid,feature,y
    sumTrainFea = getSumMiss(train_xy, train_xy.shape[0])
    train_xy = np.insert(train_xy, [train_xy.shape[1] - 1], sumTrainFea, 1)
    newFeaTrain = getMissVFromDataInteval(train_xy, train_xy.shape[0])
    train_xy = np.insert(train_xy, [train_xy.shape[1] - 1], newFeaTrain, 1)
    test_x = data.load_test_oneHot()
    sumTestFea = getSumMiss(test_x, test_x.shape[0])
    test_x = np.insert(test_x, [test_x.shape[1]], sumTestFea, 1)
    newFeaTest = getMissVFromDataInteval(test_x, test_x.shape[0])
    test_x = np.insert(test_x, [test_x.shape[1]], newFeaTest, 1)
    writeToTrain(train_xy)
    writeToTest(test_x)


def getSumMiss(X, row):
    sumFea = np.zeros(shape=(row, 1))
    for row in range(X.shape[0]):
        temp = 0
        for column in range(X.shape[1]):
            if X[row, column] == -1:
                temp = temp + 1
        sumFea[row, 0] = temp
    return sumFea


# 从数据中获得每个值的矩阵
def getMissVFromData(X, row):
    maxVSet = [19, 20, 52, 83, 92, 130, 131, 285, 4951]
    flag = 0
    for maxV in maxVSet:
        tempSet = []
        for fea in range(X.shape[1]):
            if X[:, fea].max() == maxV:
                tempSet.append(fea)
        tempFea = getMissValueFromMatrix(X[:, tempSet], row)
        if flag == 0:
            xFea = tempFea
            flag = flag + 1
        else:
            xFea = np.insert(xFea, [xFea.shape[1]], tempFea, 1)
    return xFea


# 每隔一定的间隔来添加特征，如100
def getMissVFromDataInteval(X, row):
    flag = 0
    interval = 50
    pre = 0
    tlen = int(X.shape[1]/interval)
    num = 0
    tempSet = []
    for fea in range(X.shape[1]):
        num = num + 1
        tempSet.append(fea)
        if num == interval or pre == tlen:
            pre = pre + 1
            tempFea = getMissValueFromMatrix(X[:, tempSet], row)
            if flag == 0:
                xFea = tempFea
                flag = flag + 1
            else:
                xFea = np.insert(xFea, [xFea.shape[1]], tempFea, 1)
            num = 0
            tempSet = []
    return xFea


# 统计矩阵的缺失值，放到tempFea一列中
def getMissValueFromMatrix(tempX, row):
    tempFea = np.zeros(shape=(row, 1))
    for i in range(row):
        missV = 0
        for j in range(tempX.shape[1]):
            if tempX[i, j] == -1:
                missV = missV + 1
        tempFea[i, 0] = missV
    return tempFea


def writeToTrain(train_xy):
    variableName = np.array(
        [['x' + str(c + 1) for c in range(train_xy.shape[1] - 2)]])
    variableName = np.insert(variableName, [0], ['uid'], axis=1)
    _, v_c = variableName.shape
    variableName = np.insert(variableName, [v_c], ['y'], axis=1)

    testFile = open('missingValue/train_x_oneHot_miss_50.csv', 'w', newline='')
    writer = csv.writer(testFile)
    writer.writerows(variableName)
    writer.writerows(train_xy)
    testFile.close()


def writeToTest(test_x):
    variableName = np.array(
        [['x' + str(c + 1) for c in range(test_x.shape[1]-1)]])
    variableName = np.insert(variableName, [0], ['uid'], axis=1)

    testFile = open('missingValue/test_x_oneHot_miss_50.csv', 'w', newline='')
    writer = csv.writer(testFile)
    writer.writerows(variableName)
    writer.writerows(test_x)
    testFile.close()

if __name__ == '__main__':
    loadData() 
    print('done!')
