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
    apiKey = "BQCs9W_Fy8Dw1qlrYCuwRnt_E-ZPMvN7v6MNyVN8JHjgEFZURCmNmo93hH4mE2KgupSq5qYaq2r0YDIVpd4_WHX6fbLF7fBJc3YVKmYzlo-YgFNq6D3Y7A5CFVjF2MpngvHVTNNfT-AdfotKi0M"
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
            "K = %g, accuracy = %g" % (s, calcScore(s, 'distance', X, y)))

    plt.plot(steps, cv_scores1)
    plt.title('KNN Model (Using Uniform Weights)')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Model Accuracy')
    plt.show()

    plt.plot(steps, cv_scroes2)
    plt.title('KNN Model (Using Distance Weights)')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Model Accuracy')
    plt.show()

    cv_scores = []
    for d in range(1, 20):
        cv_scores.append(calcScore(25, 'uniform', X, y))

    print(np.mean(cv_scores))


if __name__ == '__main__':
    main()

