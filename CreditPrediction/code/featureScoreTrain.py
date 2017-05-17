# coding = utf-8
import data
import numpy as np
import csv


def getOneHotFile():
    X, y = data.load_train_oneHot()

    temp_y = np.array([y])
    y = np.transpose(temp_y)
    temp_uid = np.array([X[:, 0]])
    X_uid = np.transpose(temp_uid)
    X = np.delete(X, 0, 1)

    # 15000,1138

    print('begin select features...')
    feaScore = data.load_feaScore()  # have minus 1
    print('feaScore"s len:', len(feaScore))

    final_X = X_uid

    print('final_X:', final_X.shape)

    for fea in feaScore:
        temp = np.array([X[:, fea]])
        X_temp_i = np.transpose(temp)
        final_X = np.insert(
            final_X, [final_X.shape[1]], X_temp_i, 1)
        print('final_X:', final_X.shape)

    print('select features done!')

    final_X = np.insert(final_X, [final_X.shape[1]], y, 1)

    print('last final_X:', final_X.shape)

    variableName = np.array(
        [['x' + str(c + 1) for c in range(final_X.shape[1] - 2)]])
    variableName = np.insert(variableName, [0], ['uid'], axis=1)
    variableName = np.insert(
        variableName, [variableName.shape[1]], ['y'], axis=1)

    testFile = open(
        'feature_score/train_x_oneHot_feaScore.csv', 'w', newline='')
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
