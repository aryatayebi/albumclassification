import loadData
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from matplotlib import pyplot as plt

def calcScore(c, gamma, X, y):
    model = SVC(C=c, kernel='rbf', gamma=gamma)
    scores = cross_val_score(model, X, y, cv=50, scoring='accuracy')

    return scores.mean()


def main():
    print('Loading dataset...')
    apiKey = "BQD0VrfBRMvOJgNXkW4PkoGTr4Deb4j8XDzTLW74Sf1WhEh0U95DEzprX5Iyw5-1gcHdcuX80Y4Xf-IHHov0R8xsN4tCHofK5Ua6tH1J9ueFvgB_V_FAqHYe60KeZ4P8E0aGO9aUxQCFVMn4ol0"
    data = loadData.Dataset(apiKey, debug=True)

    print("Preprocessing...")
    allData = data.preprocessKNN()

    X = allData.iloc[:, 2:].values
    y = allData.iloc[:, 1].values

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33, stratify=y)

    # data will likely not be linearly seperable, so we will use rbf for kernel
    # large C: lower bias, higher variance
    # small C: higher bias, lower variance

    steps = np.array([0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50])
    cv_scores = []
    for i in steps:
        score = calcScore(i, 'auto', X, y)
        cv_scores.append(score)
        print("C = %g score = %g" %
              (i, score))

    plt.plot(steps, cv_scores)
    plt.title('SVM Model')
    plt.xlabel('Parameter C')
    plt.ylabel('Model Accuracy')
    plt.show()

if __name__ == '__main__':
    main()