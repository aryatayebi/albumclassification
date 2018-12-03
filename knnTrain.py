import loadData
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def calcScore(neighbors, weight, trainFeat, trainLabel, testFeat, testLabel):
    model = KNeighborsClassifier(n_neighbors=neighbors, weights=weight)
    model.fit(trainFeat, trainLabel)

    accuracy = model.score(testFeat, testLabel)
    return accuracy



def main():
    print('Loading dataset...')
    apiKey = "18a7c1e4adc3bc81521a35f3f4f3a7bf"
    data = loadData.Dataset(apiKey)

    print("Preprocessing...")
    trainKNN, validateKNN, testKNN = data.preprocessKNN()

    trainFeat = trainKNN.iloc[:,2:].values
    trainLabel = trainKNN.iloc[:,1].values


    validateFeat = validateKNN.iloc[:,2:].values
    validateLabel = validateKNN.iloc[:,1].values


    steps = np.array([1, 3, 5, 10, 25, 100])

    print("Uniform Weights")
    for s in steps:
        print("N = %g, accuracy = %g" % (s, calcScore(s, 'uniform', trainFeat, trainLabel, validateFeat, validateLabel)))

    print("Distance Weights")
    for s in steps:
        print(
            "N = %g, accuracy = %g" % (s, calcScore(s, 'distance', trainFeat, trainLabel, validateFeat, validateLabel)))


if __name__ == '__main__':
    main()