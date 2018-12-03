import loadData
from sklearn.svm import SVC
import numpy as np

def calcScore(c, gamma, trainFeat, trainLabel, testFeat, testLabel):
    model = SVC(C=c, kernel='rbf', gamma=gamma)
    model.fit(trainFeat, trainLabel)

    predict = model.predict(testFeat)
    comp = np.array([testLabel, predict])
    comp = np.transpose(comp)

    score = 0
    for c in comp:
        if (c[0] == c[1]):
            score += 1

    return score


def main():
    print('Loading dataset...')
    apiKey = "18a7c1e4adc3bc81521a35f3f4f3a7bf"
    data = loadData.Dataset(apiKey)

    print("Preprocessing...")
    trainKNN, validateKNN, testKNN = data.preprocessKNN()

    trainFeat = trainKNN.iloc[:, 2:].values
    trainLabel = trainKNN.iloc[:, 1].values

    validateFeat = validateKNN.iloc[:, 2:].values
    validateLabel = validateKNN.iloc[:, 1].values


    # data will likely not be linearly seperable, so we will use rbf for kernel
    # large C: lower bias, higher variance
    # small C: higher bias, lower variance

    steps = np.array([0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50])
    for i in steps:
        for j in steps:
            print("C = %g, sig_squared = %g, percentage score = %g" % (i, j, calcScore(i, j, trainFeat, trainLabel, validateFeat, validateLabel) / len(validateLabel)))




if __name__ == '__main__':
    main()