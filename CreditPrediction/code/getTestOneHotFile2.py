# coding = utf-8
import data
import numpy as np
import csv


def getOneHotFile():
    test_x = data.load_test()
    miss_1050 = np.array([1506, 4872, 2528])
    miss = test_x[miss_1050, :]
    temp_uid = np.array([test_x[:, 0]])
    test_uid = np.transpose(temp_uid)
    test_x = np.delete(test_x, 0, 1)
    test_x_row, test_x_column = test_x.shape

    # 4994,1138

    print('begin oneHotEncoding...')

    flag = 0
    for i in range(test_x_column):
        exist = False
        feaInfo = data.load_feaInfo()
        feaInfo_r, feaInfo_c = feaInfo.shape
        for k in range(feaInfo_r):
            if feaInfo[k, 0] == i + 1:
                # print('feature:', 'x'+str(feaInfo[k, 0]))
                valuesNum = feaInfo[k, 2] - feaInfo[k, 1] + 1
                newFeaValue = np.zeros(shape=(test_x_row, valuesNum))
                # print(newFeaValue.shape)
                for j in range(test_x_row):
                    if test_x[j, i] == -1 or test_x[j, i] == -2:
                        newFeaValue[j, :] = -1
                    else:
                        index = test_x[j, i] - feaInfo[k, 1]
                        newFeaValue[j, index] = 1
                if flag == 0:
                    final = newFeaValue
                final = np.insert(final, [final.shape[1]], newFeaValue, 1)
                flag = flag + 1
                exist = True
                break
        if exist is False:
            temp_insert = np.transpose(np.array([test_x[:, i]]))
            if flag == 0:
                final = temp_insert
            else:
                final = np.insert(final, [final.shape[1]], temp_insert, 1)
            flag = flag + 1

    print('final:', final.shape)
    print('oneHotEncoding done!')

    final = np.insert(final, [0], test_uid, axis=1)
    print('last final:', final.shape)

    variableName = np.array(
        [['x' + str(c + 1) for c in range(final.shape[1]-1)]])
    variableName = np.insert(variableName, [0], ['uid'], axis=1)

    testFile = open('data/new01/test_x_oneHot.csv', 'w', newline='')
    writer = csv.writer(testFile)
    writer.writerows(variableName)
    writer.writerows(final)
    testFile.close()

    variableName2 = np.array(
        [['x' + str(c + 1) for c in range(miss.shape[1]-1)]])
    variableName2 = np.insert(variableName2, [0], ['uid'], axis=1)

    missFile = open('data/new01/missTest_1050.csv', 'w', newline='')
    writer2 = csv.writer(missFile)
    writer2.writerows(variableName2)
    writer2.writerows(miss)
    missFile.close()


def testGram():
    pass

if __name__ == '__main__':
    getOneHotFile()
    # testGram()
    print('done!')
