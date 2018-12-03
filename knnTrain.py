import loadData
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
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
    knnData = data.preprocessKNN()

    X = knnData.iloc[:,2:].values
    y = knnData.iloc[:,1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33, stratify=y)


    steps = np.array([1, 3, 5, 10, 25, 100])

    print("Uniform Weights")
    for s in steps:
        print("N = %g, accuracy = %g" % (s, calcScore(s, 'uniform', X_train, y_train, X_test, y_test)))

    print("Distance Weights")
    for s in steps:
        print(
            "N = %g, accuracy = %g" % (s, calcScore(s, 'distance', X_train, y_train, X_test, y_test)))


if __name__ == '__main__':
    main()