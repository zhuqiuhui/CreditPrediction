import csv
import numpy as np


def load_val():
    with open('c:/val_result.csv') as f:
        next(f)
        val = [float(row[0]) for row in csv.reader(f)]
    return val


def load_missValue():
    with open('F:/CreditPrediction/data/missValueRow.csv') as f:
        next(f)
        missValue = [[float(c) for c in row[0:2]] for row in csv.reader(f)]
        missValue = np.asarray(missValue)
    return missValue


if __name__ == '__main__':
    val = load_val()
    missValue = load_missValue()
    print('val:', len(val))
    print('missValue:', missValue.shape)
    miss_r, miss_c = missValue.shape
    result = []
    flag = 0
    for i in val:
        for j in range(miss_r):
            if i == missValue[j, 0]:
                result.append(missValue[j, 1])
                flag = flag + 1
                break
    print('flag:', flag)
    result = np.transpose(np.array([result]))
    resultFile = open('C:/resultFile.csv', 'w', newline='')
    writer = csv.writer(resultFile)
    writer.writerows(result)
    resultFile.close()
