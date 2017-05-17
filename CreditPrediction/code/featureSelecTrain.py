# coding = utf-8
import data
import numpy as np
import csv


def getOneHotFile():
    X, y, _ = data.load_train()

    temp_y = np.array([y])
    y = np.transpose(temp_y)
    temp_uid = np.array([X[:, 0]])
    X_uid = np.transpose(temp_uid)
    X = np.delete(X, 0, 1)
    X_row, X_column = X.shape

    # 15000,1138

    print('begin oneHotEncoding...')
    feaMissInfo = data.load_feaMissInfo()
    feaMiss = []  # store the used feature
    for fea in feaMissInfo:
        if fea[1] <= 24:
            feaMiss.append(fea[0] - 1)

    feaInfo = data.load_feaInfo()
    feaInfo_r, feaInfo_c = feaInfo.shape

    final_X = X_uid

    print('final_X:', final_X.shape)
    for i in range(X_column):
        print('i:', i)
        if i in feaMiss:  # ues
            flag = 0
            for k in range(feaInfo_r):
                if feaInfo[k, 0] == i + 1:  # need to oneHot
                    # print('feature:', 'x'+str(feaInfo[k, 0]))
                    flag = 1
                    valuesNum = feaInfo[k, 2] - feaInfo[k, 1] + 1
                    newFeaValue = np.zeros(shape=(X_row, valuesNum))

                    for j in range(X_row):
                        if X[j, i] == -1 or X[j, i] == -2:
                            newFeaValue[j, :] = -1
                        else:
                            index = X[j, i] - feaInfo[k, 1]
                            newFeaValue[j, index] = 1
                    final_X = np.insert(
                        final_X, [final_X.shape[1]], newFeaValue, 1)
                    break
            if flag == 0:  # need not to oneHot
                temp = np.array([X[:, i]])
                X_temp_i = np.transpose(temp)
                final_X = np.insert(
                    final_X, [final_X.shape[1]], X_temp_i, 1)
            print('final_X:', final_X.shape)

    print('oneHotEncoding done!')

    final_X = np.insert(final_X, [final_X.shape[1]], y, 1)

    print('last final_X:', final_X.shape)

    _, lastColumnVa = final_X.shape
    variableName = np.array(
        [['x' + str(c + 1) for c in range(lastColumnVa - 2)]])
    variableName = np.insert(variableName, [0], ['uid'], axis=1)
    _, v_c = variableName.shape
    variableName = np.insert(variableName, [v_c], ['y'], axis=1)

    testFile = open('data/01fea/train_x_oneHot_fea.csv', 'w', newline='')
    writer = csv.writer(testFile)
    writer.writerows(variableName)
    writer.writerows(final_X)
    testFile.close()


def testGram():
    pass

if __name__ == '__main__':
    getOneHotFile()
    # testGram()
    print('done!')
