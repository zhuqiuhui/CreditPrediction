# coding = utf-8
import data
import numpy as np
import csv


def getOneHotFile():
    test_x = data.load_test()

    temp_uid = np.array([test_x[:, 0]])
    test_uid = np.transpose(temp_uid)
    test_x = np.delete(test_x, 0, 1)
    test_x_row, test_x_column = test_x.shape

    # 5000,1138

    print('begin oneHotEncoding...')
    feaMissInfo = data.load_feaMissInfo()
    feaMiss = []  # store the used feature
    for fea in feaMissInfo:
        if fea[1] <= 24:
            feaMiss.append(fea[0] - 1)

    feaInfo = data.load_feaInfo()
    feaInfo_r, feaInfo_c = feaInfo.shape

    final_X = test_uid

    print('final_X:', final_X.shape)
    for i in range(test_x_column):
        print('i:', i)
        if i in feaMiss:  # ues
            flag = 0
            for k in range(feaInfo_r):
                if feaInfo[k, 0] == i + 1:  # need to oneHot
                    # print('feature:', 'x'+str(feaInfo[k, 0]))
                    flag = 1
                    valuesNum = feaInfo[k, 2] - feaInfo[k, 1] + 1
                    newFeaValue = np.zeros(shape=(test_x_row, valuesNum))

                    for j in range(test_x_row):
                        if test_x[j, i] == -1 or test_x[j, i] == -2:
                            newFeaValue[j, :] = -1
                        else:
                            index = test_x[j, i] - feaInfo[k, 1]
                            newFeaValue[j, index] = 1
                    final_X = np.insert(
                        final_X, [final_X.shape[1]], newFeaValue, 1)
                    break
            if flag == 0:  # need not to oneHot
                temp = np.array([test_x[:, i]])
                X_temp_i = np.transpose(temp)
                final_X = np.insert(
                    final_X, [final_X.shape[1]], X_temp_i, 1)
            print('final_X:', final_X.shape)

    print('oneHotEncoding done!')

    _, lastColumnVa = final_X.shape
    variableName = np.array(
        [['x' + str(c + 1) for c in range(lastColumnVa - 1)]])
    variableName = np.insert(variableName, [0], ['uid'], axis=1)

    testFile = open('data/01fea/test_x_oneHot_fea.csv', 'w', newline='')
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
