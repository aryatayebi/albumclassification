import loadData
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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
    allData = data.preprocessKNN()

    X = allData.iloc[:, 2:].values
    y = allData.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33, stratify=y)

    # data will likely not be linearly seperable, so we will use rbf for kernel
    # large C: lower bias, higher variance
    # small C: higher bias, lower variance

    steps = np.array([0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50])
    for i in steps:
        for j in steps:
            print("C = %g, gamma = %g, score = %g" %
                  (i, j, calcScore(i, j, X_train, y_train, X_test, y_test) / len(y_test)))


if __name__ == '__main__':
    main()