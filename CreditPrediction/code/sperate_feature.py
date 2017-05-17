# coding=utf-8
import csv
import numpy as np
originFile = open(
    "F:\\loan competetion\\featureSeperate\\svd\\train_label.csv", "r")
readerFile = csv.reader(originFile)

flag = 1
csvWriteFile = open(
    "F:\\loan competetion\\featureSeperate\\svd\\train_label2.csv", "a+", newline='')
writer = csv.writer(csvWriteFile)
na = []
na.append('uid')
nameIndex = 2
while nameIndex <= 1759:
    na.append('x' + str(nameIndex))
    nameIndex = nameIndex + 1
na.append('y')  # ------------------------------------test delete this line
writer.writerow(na)
ii = 0
for line in readerFile:
    lineLen = len(line)
    i = 1
    if ii == 0:
        ii = ii + 1
        continue
    while i < lineLen - 1:  # ---------------------------变为test后应该改的while i < lineLen:
        cname = 'x' + str(i)
        # 查找cname是否为category变量，若是，则拆分，否则过
        categoryFeature = open(
            "F:\\loan competetion\\data process\\catefea.csv", "r")
        feaFile = csv.reader(categoryFeature)
        for cfea in feaFile:
            #print ('cname:',cname)
            if cname == cfea[0]:
                tempLen = int(cfea[2]) - int(cfea[1]) + 1
                tempPortion = []
                if int(line[i]) == -1 or int(line[i]) == -2:  # 如果对应列值等于-1，对应的列值均为-1
                    j = 0
                    while j < tempLen:
                        tempPortion.append(-1)
                        j = j + 1
                else:
                    tempPortion = np.zeros(tempLen)  # 创建如一个列表含有4个0
                    index = int(line[i])
                    index = index - int(cfea[1])
                    tempPortion[index] = 1
                line[i] = tempPortion
        categoryFeature.close()
        i = i + 1
    print(flag, ' processing...')
    tempLine = []
    for tline in line:
        if isinstance(tline, str):
            tempLine.append(tline)
        else:
            for iii in tline:
                tempLine.append(str(iii))
    writer.writerow(tempLine)
    flag = flag + 1
csvWriteFile.close()
originFile.close()

print('Write all over!')
