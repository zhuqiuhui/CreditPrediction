# coding=utf-8
import numpy as np
import data
import csv


def summary():
    # id,缺失值数量,类别,（如果值）|（）
    temp = open(
        'E:\\zqh\\CreditPrediction\\data\\delete_19\\featureStatistic_del19.csv', 'w')

    X, y, ftype = data.load_train()
    _, n_features = X.shape
    for i in range(n_features):
        values = np.unique(X[:, i], return_counts=True)
        print('values[0]:', len(values[0]), 'values[1]:', len(values[1]))
        print('i:', i)
        j = 0
        f = 0
        while j < len(values[0]):
            if values[0][j] == -1:
                f = 1
                break
            j = j + 1

        if f == 0:
            numpos = 0
        else:
            numpos = values[1][j]

        # if the number of feature's values lower than 40,output into the file
        if len(values[0]) <= 40:
            # values[0] stores the number of differet feature values
            str1 = ', '.join([str(v) for v in values[0]])
            # values[1] stores the count number in values[0]
            str2 = ', '.join([str(v) for v in values[1]])
            temp.write('%s,%s,%d,"%s","%s",%d\n' % (
                'x' + str(i + 1), ftype[i], len(values[0]), str1, str2, numpos))
        else:
            temp.write('%s,%s,%d,%d_%d,%d_%d,%d\n' % (
                'x' + str(i + 1), ftype[i], len(values[0]), values[0][0], values[1][0], values[0][-1], values[1][-1], numpos))


def static_fea():
    temp = open('feaMissInfo.csv', 'w')

    X, y, ftype = data.load_train()
    _, n_features = X.shape
    for i in range(n_features):
        values = np.unique(X[:, i], return_counts=True)
        print('values[0]:', len(values[0]), 'values[1]:', len(values[1]))
        print('i:', i)
        j = 0
        f = 0
        while j < len(values[0]):
            if values[0][j] == -1:
                f = 1
                break
            j = j + 1

        if f == 0:
            numpos = 0
        else:
            numpos = values[1][j]
        temp.write('%d,%d\n' % ((i+1), numpos))


def summaryTest():
    temp = open(
        'E:\\zqh\\CreditPrediction\\data\\pre_data\\test_featureStatistic_proc.csv', 'w')

    X, y, ftype = data.load_train()
    X_test = data.load_test()
    _, n_features = X_test.shape
    for i in range(n_features):
        values = np.unique(X_test[:, i], return_counts=True)
        j = 0
        f = 0
        while j < len(values[0]):
            if values[0][j] == -1:
                f = 1
                break
            j = j + 1

        if f == 0:
            numpos = 0
        else:
            numpos = values[1][j]

        # if the number of feature's values lower than 40,output into the file
        if len(values[0]) <= 40:
            # values[0] stores the number of differet feature values
            str1 = ', '.join([str(v) for v in values[0]])
            # values[1] stores the count number in values[0]
            str2 = ', '.join([str(v) for v in values[1]])
            temp.write('%s,%s,%d,"%s","%s",%d\n' % (
                'x' + str(i + 1), ftype[i], len(values[0]), str1, str2, numpos))
        else:
            temp.write('%s,%s,%d,%d_%d,%d_%d,%d\n' % (
                'x' + str(i + 1), ftype[i], len(values[0]), values[0][0], values[0][-1], values[1][0], values[1][-1], numpos))


def findValue():
    X, y, ftype = data.load_train()
    _, n_features = X.shape
    # output the detail information of feature fea
    fea = 326
    values = np.unique(X[:, fea], return_counts=True)
    str1 = ', '.join([str(v) for v in values[0]])
    str2 = ', '.join([str(v) for v in values[1]])
    print("feature's value:", str1)
    print("feature's number:", str2)


def findMissValueRow():
    temp = open(
        'E:\\zqh\\CreditPrediction\\data\\delete_19\\missValueRow_test.csv', 'w')
    X, y, ftype = data.load_train()
    X_test = data.load_test()
    # print(X_test)
    n_row, _ = X_test.shape
    print('n_row:', n_row)
    for i in range(n_row):
        values = np.unique(X_test[i, :], return_counts=True)
        j = 0
        f = 0
        while j < len(values[0]):
            if values[0][j] == -1:
                f = 1
                break
            j = j + 1

        if f == 0:
            numpos = 0
        else:
            numpos = values[1][j]
        temp.write('%d\n' % (numpos))


if __name__ == '__main__':
    static_fea()
    print("over!")
    # findValue()
    # print("over!")
    # findMissValueRow()
    # print('wirte over!')

    # summary()
    # print('write over!')

    # X, y, ftype = data.load_train()
    # count = 0
    # for x in X[:, [437, 439]]:
    #     if x[0] != x[1]:
    #         print(x)
    #     else:
    #         count += 1
    # print(count)
    # X, y, ftype = data.load_train()
    # Xtest = data.load_test()
    # csvWriFile = open('D:\\aft_del_19_file.csv','a+',newline='')
    # writer = csv.writer(csvWriFile)
    # for ii in X[[2922],:]:
    #     writer.writerow(ii)
    # csvWriFile.close()
    # count = 0
    # csvWriFile = open('D:\\19_file.csv','a+',newline='')
    # writer = csv.writer(csvWriFile)
    # for ii in X[[1043,1153,1219,1956,3221,3257,5231,7668,7912,8261,8498,8514,8548,8876,11027,11102,11247,11419,13904],:]:
    #     writer.writerow(ii)
    # csvWriFile.close()
    # for x in X[[8514,1043], :]:
    #     if x[0] != x[1]:
    #         print(x)
    #     else:
    #         count += 1
    # print('count:',count)

    # X = data.load_test()
    # for x in X[:, [437, 439]]:
    #     if x[0] != x[1]:
    #         print(x)
    #     else:
    #         count += 1
    # print(count)
