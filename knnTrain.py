import loadData
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from matplotlib import pyplot as plt



def calcScore(neighbors, weight, X, y):
    model = KNeighborsClassifier(n_neighbors=neighbors, weights=weight)
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')

    return scores.mean()

def main():
    print('Loading dataset...')
    apiKey = ""
    data = loadData.Dataset(apiKey, debug=False)

    print("Preprocessing...")
    knnData = data.preprocessKNN()

    X = knnData.iloc[:,2:].values
    y = knnData.iloc[:,1].values

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33, stratify=y)


    steps = np.array(range(2,52,2))

    cv_scores1 = []
    cv_scroes2 = []

    print("Uniform Weights")
    for s in steps:
        score = calcScore(s, 'uniform', X, y)
        cv_scores1.append(score)
        print("K = %g, accuracy = %g" % (s, score))

    print("Distance Weights")
    for s in steps:
        score = calcScore(s, 'distance', X, y)
        cv_scroes2.append(score)
        print(
            "K = %g, accuracy = %g" % (s, score))

    plt.plot(steps, cv_scores1)
    plt.plot(steps, cv_scroes2)
    plt.title('KNN Model')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Model Accuracy')
    plt.legend(['Uniform Weight', 'Distance Weight'], loc='lower right')
    plt.show()

    cv_scores = []
    cv_weighted_scores = []
    for d in range(1, 20):
        cv_scores.append(calcScore(20, 'uniform', X, y))

    for d in range(1, 20):
        cv_weighted_scores.append(calcScore(20, 'distance', X, y))

    print( "Mean accuracy of 20 cv models using k = 20 neighbors is %g" % (np.mean(cv_scores)))
    print( "Mean accuracy of 20 cv models using k = 20 neighbors with distance weight is %g" % (np.mean(cv_weighted_scores)))


if __name__ == '__main__':
    main()

