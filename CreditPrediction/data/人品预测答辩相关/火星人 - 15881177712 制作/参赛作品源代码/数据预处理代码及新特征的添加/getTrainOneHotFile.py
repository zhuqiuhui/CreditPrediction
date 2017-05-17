# coding = utf-8
import data
import numpy as np
import csv


def getOneHotFile():
    X, y, _ = data.load_train()
    # delete the outlier sample
    miss_1050 = np.array([1043, 1153, 1219, 1956, 3221, 3257, 5231, 7668, 7912,
                         8261, 8498, 8514, 8548, 8876, 11027, 11102, 11247, 11419, 13904])  # line number of outlier sample
    miss = X[miss_1050, :]
    miss_y = y
    # X = np.delete(X, outliers, 0)
    # y = np.delete(y, outliers, 0)
    temp_y = np.array([y])
    y = np.transpose(temp_y)
    miss_y = y[miss_1050, :]
    miss = np.insert(miss, [miss.shape[1]], miss_y, 1)
    temp_uid = np.array([X[:, 0]])
    X_uid = np.transpose(temp_uid)
    X = np.delete(X, 0, 1)
    X_row, X_column = X.shape

    # 14981,1138

    print('begin oneHotEncoding...')

    flag = 0
    for i in range(X_column):
        exist = False
        feaInfo = data.load_feaInfo()
        feaInfo_r, feaInfo_c = feaInfo.shape
        for k in range(feaInfo_r):
            if feaInfo[k, 0] == i + 1:
                # print('feature:', 'x'+str(feaInfo[k, 0]))
                valuesNum = feaInfo[k, 2] - feaInfo[k, 1] + 1
                newFeaValue = np.zeros(shape=(X_row, valuesNum))
                # print(newFeaValue.shape)
                for j in range(X_row):
                    if X[j, i] == -1 or X[j, i] == -2:
                        newFeaValue[j, :] = -1
                    else:
                        index = X[j, i] - feaInfo[k, 1]
                        newFeaValue[j, index] = 1
                if flag == 0:
                    final = newFeaValue
                final = np.insert(final, [final.shape[1]], newFeaValue, 1)
                flag = flag + 1
                exist = True
                break
        if exist is False:
            temp_insert = np.transpose(np.array([X[:, i]]))
            if flag == 0:
                final = temp_insert
            else:
                final = np.insert(final, [final.shape[1]], temp_insert, 1)
            flag = flag + 1

    print('final:', final.shape)
    print('oneHotEncoding done!')
    final = np.insert(final, [0], X_uid, axis=1)
    final = np.insert(final, [final.shape[1]], y, axis=1)

    print('last final:', final.shape)

    variableName = np.array(
        [['x' + str(c + 1) for c in range(final.shape[1] - 2)]])
    variableName = np.insert(variableName, [0], ['uid'], axis=1)
    _, v_c = variableName.shape
    variableName = np.insert(variableName, [v_c], ['y'], axis=1)

    testFile = open('train_x_oneHot.csv', 'w', newline='')
    writer = csv.writer(testFile)
    writer.writerows(variableName)
    writer.writerows(final)
    testFile.close()

    variableName2 = np.array(
        [['x' + str(c + 1) for c in range(miss.shape[1] - 2)]])
    variableName2 = np.insert(variableName2, [0], ['uid'], axis=1)
    _, v_c = variableName2.shape
    variableName2 = np.insert(variableName2, [v_c], ['y'], axis=1)
    missFile = open('missTrain_1050.csv', 'w', newline='')
    writer2 = csv.writer(missFile)
    writer2.writerows(variableName2)
    writer2.writerows(miss)
    missFile.close()


def testGram():
    pass

if __name__ == '__main__':
    getOneHotFile()
    print('done!')
