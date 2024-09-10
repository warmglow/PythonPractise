
import numpy as np
import pandas
import scipy

trackNum = 81
repeatNum = 5
cenMatHeight = 10
cenMatWidth = 12
maskMatHeight = 111
maskMatWidth = 111

with open('suncen_20240704ljbCross.txt', 'r') as file:
    lines = file.readlines()
file.close()

# tmpLine = lines.pop(0)
#
# tmpLine = tmpLine.lstrip('%R ')
# tmpLine = tmpLine.rstrip('\n')
# tmpData = [float(itemLine) for itemLine in tmpLine.split()]
print('haha')

maskMat = np.empty([maskMatHeight, maskMatWidth], dtype=object)
maskTickMat = np.zeros([maskMatHeight, maskMatWidth])

for k in range(trackNum):
    tmpXMat = np.zeros([cenMatHeight, cenMatWidth, repeatNum])
    tmpYMat = np.zeros([cenMatHeight, cenMatWidth, repeatNum])
    rightIdxMask = np.zeros(repeatNum)
    upperIdxMask = np.zeros(repeatNum)
    rowNumMask = np.zeros(repeatNum)
    colNumMask = np.zeros(repeatNum)

    for t in range(repeatNum):
        flagValid = 0
        for r in range(cenMatHeight):
            for c in range(cenMatWidth):
                tmpLine = lines.pop(0)
                tmpLine = tmpLine.lstrip('%R ')
                tmpLine = tmpLine.rstrip('\n')
                tmpData = [float(itemLine) for itemLine in tmpLine.split()]
                if r == 1 and c == 1 and tmpData[4] != 0 and tmpData[5] != 0:
                    rowNumMask[t] = tmpData[4]
                    colNumMask[t] = tmpData[5]
                    rightIdxMask[t] = tmpData[7] + 6
                    upperIdxMask[t] = tmpData[6]
                    if rightIdxMask[t] - colNumMask[t] + 1 >= 0 and upperIdxMask[t] + rowNumMask[t] - 1 <= 111:
                        flagValid = 1

                if flagValid == 1:
                    tmpXMat[r, c, t] = tmpData[2]
                    tmpYMat[r, c, t] = tmpData[3]

    tmpXCMat = np.zeros([maskMatHeight, maskMatWidth])
    tmpYCMat = np.zeros([maskMatHeight, maskMatWidth])
    tmpCntCMat = np.zeros([maskMatHeight, maskMatWidth])
    for t in range(repeatNum):
        if rowNumMask[t] != 0 and colNumMask[t] != 0:
            for r in range(1, rowNumMask[t]):
                for c in range(colNumMask[t], 1, -1):
                    if np.abs(tmpXMat[r, c, t]) > 1e-6 and np.abs(tmpYMat[r, c, t]) > 1e-6:
                        tmpXCMat[upperIdxMask[t] + r - 1, rightIdxMask[t] - c + 1] += tmpXMat[r, c, t]
                        tmpYCMat[upperIdxMask[t] + r - 1, rightIdxMask[t] - c + 1] += tmpYMat[r, c, t]
                        tmpCntCMat[upperIdxMask[t] + r - 1, rightIdxMask[t] - c + 1] += 1

    tmpXCMat = tmpXCMat / tmpCntCMat
    tmpYCMat = tmpYCMat / tmpCntCMat
    deltaXCMat = np.sqrt((tmpXCMat[:, 2:] - tmpXMat[:, :-2]) ** 2 + (tmpYCMat[:, 2:] - tmpYMat[:, :-2]) ** 2) * 0.004
    deltaYCMat = np.sqrt((tmpXCMat[2:, :] - tmpXMat[:-2, :]) ** 2 + (tmpYCMat[2:, :] - tmpYMat[:-2, :]) ** 2) * 0.004

    tmpLine = lines.pop(0)
    tmpData = [float(itemLine) for itemLine in tmpLine.split()]
    for m in range(1, 112):
        for n in range(1, 112):
            if tmpCntCMat[m, n] != 0:
                if maskTickMat[m, n] != 0:
                    tmpMat = np.concatenate([maskMat[m, n], np.concatenate([np.array(tmpData), tmpXCMat[m, n], tmpYCMat[m, n], tmpCntCMat[m, n]], axis=1)], axis=0)
                else:
                    tmpMat = np.concatenate([np.array(tmpData), tmpXCMat[m, n], tmpYCMat[m, n], tmpCntCMat[m, n]], axis=1)
                    maskTickMat[m, n] = 1
                maskMat[m, n] = tmpMat

