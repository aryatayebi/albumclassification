import loadData
from sklearn.neighbors import KNeighborsClassifier

def main():
    print('Loading dataset...')
    apiKey = "18a7c1e4adc3bc81521a35f3f4f3a7bf"
    data = loadData.Dataset(apiKey)

    print("Preprocessing...")
    trainKNN, validateKNN, testKNN = data.preprocessKNN()

    print(trainKNN.shape)
    print(validateKNN.shape)
    print(testKNN.shape)

    trainFeat = trainKNN.iloc[:,2:].values
    trainLabel = trainKNN.iloc[:,1].values
    print(trainFeat)
    print(trainLabel)

    validateFeat = validateKNN.iloc[:,2:].values
    validateLabel = validateKNN.iloc[:,1].values
    print(validateFeat)
    print(validateLabel)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(trainFeat, trainLabel)
    accuracy = model.score(validateFeat, validateLabel)
    print(accuracy)




if __name__ == '__main__':
    main()