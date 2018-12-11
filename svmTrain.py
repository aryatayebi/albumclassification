import loadData
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from matplotlib import pyplot as plt

def calcScore(c, gamma, X, y):
    model = SVC(C=c, kernel='rbf', gamma=gamma)
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')

    return scores.mean()


def main():
    print('Loading dataset...')
    apiKey = "BQCa8qiUEUx-pkbwLh_zTy48tMzNkrBzGGxbR8sE0sy7D5LMR6jlgBKgLB63NDeXAFM4kvzKOKyv6PeE7kgcAFkCr3ldz0wTAfb-oPthEoX5tBTvqVelP6jJVxGbMY1E0KPJRozPhYfwIRxmFNc"
    data = loadData.Dataset(apiKey, debug=False)

    print("Preprocessing...")
    allData = data.preprocessKNN()

    X = allData.iloc[:, 2:].values
    y = allData.iloc[:, 1].values

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33, stratify=y)

    # data will likely not be linearly seperable, so we will use rbf for kernel
    # large C: lower bias, higher variance
    # small C: higher bias, lower variance

    steps = np.array([0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100])
    cv_scores = []
    for i in steps:
        score = calcScore(i, 'auto', X, y)
        cv_scores.append(score)
        print("C = %g score = %g" %
              (i, score))

    cv_scores = np.array(cv_scores)

    plt.plot(np.log(steps), cv_scores)
    plt.title('SVM Model')
    plt.xlabel('Parameter C (log)')
    plt.ylabel('Model Accuracy')
    plt.show()

    # best_c = np.argmax(cv_scores)
    # scores = []
    # for d in range(1, 20):
    #     print(d)
    #     scores.append(calcScore(steps[best_c], 'auto', X, y))
    #
    # for i in scores:
    #     print(i)
    #
    # print("Mean accuracy of 20 models using c = %d is %d" % (steps[best_c], np.mean(scores)))

if __name__ == '__main__':
    main()
