# coding = utf-8
import data
import numpy as np
import csv


def getOneHotFile():
    test_x = data.load_test_oneHot()

    temp_uid = np.array([test_x[:, 0]])
    test_uid = np.transpose(temp_uid)
    test_x = np.delete(test_x, 0, 1)

    # 5000,1138

    print('begin select features...')
    feaScore = data.load_feaScore()  # have minus 1
    print('feaScore"s len:', len(feaScore))

    final_X = test_uid

    print('final_X:', final_X.shape)

    for fea in feaScore:
        temp = np.array([test_x[:, fea]])
        X_temp_i = np.transpose(temp)
        final_X = np.insert(
            final_X, [final_X.shape[1]], X_temp_i, 1)
        print('final_X:', final_X.shape)

    print('select features done!')

    variableName = np.array(
        [['x' + str(c + 1) for c in range(final_X.shape[1] - 1)]])
    variableName = np.insert(variableName, [0], ['uid'], axis=1)

    testFile = open('feature_score/test_x_oneHot_feaScore.csv',
                    'w', newline='')
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
