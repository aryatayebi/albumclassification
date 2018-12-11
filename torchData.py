import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as lfunc
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import loadData

class Cnn(torch.nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 13 * 13, 1000)
        self.fc2 = torch.nn.Linear(1000, 500)
        self.fc3 = torch.nn.Linear(500, 120)
        self.fc4 = torch.nn.Linear(120, 84)
        self.fc5 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # First relu and convolutional transformation and maxPool
        x = self.pool(F.relu(self.conv1(x)))
        # Second relu and convolutional transformation and maxPool
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        # Third relu with first linear transformation
        x = F.relu(self.fc1(x))
        # Fourth relu with Second linear transformation
        x = F.relu(self.fc2(x))
        # Fifth relu with Third linear transformation
        x = F.relu(self.fc3(x))
        # Fourth linear transformation
        x = self.fc4(x)
        # Fifth linear transformation
        x = self.fc5(x)
        return x


class Train():
    def __init__(self, train_data, test_data):

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=2,shuffle=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=2,shuffle=False)

        self.net = Cnn()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = lfunc.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        self.training(trainloader)
        self.test(testloader)

    def training(self, trainloader):

        lossList = []

        for epoch in range(10):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, genres = data

                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimizing
                predicted_genres = self.net(inputs.float())
                loss = self.criterion(predicted_genres, genres)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 1000 == 999:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 500))
                    lossList.append(running_loss / 500)
                    running_loss = 0.0
        x = 0

        # plotting
        for i in lossList:
            plt.scatter(x, i)
            x = x + 1
        plt.title('')
        plt.xlabel('Iterations (Per 1000)')
        plt.ylabel('RunningLoss')
        plt.show()
        print('Finished Training')

    def test(self, testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, genres = data
                outputs = self.net(images.float())
                _, predicted = torch.max(outputs.data, 1)
                total += genres.size(0)
                correct += (predicted == genres).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
                100 * correct / total))

        genre_correct = list(0. for i in range(10))
        genre_total = list(0. for i in range(10))
        genre_list = ['electronic', 'indie', 'pop', 'metal', 'alternative%20rock', 'classic%20rock',
                      'jazz', 'folk', 'rap', 'classical']

        with torch.no_grad():
            for data in testloader:
                images, genres = data
                outputs = self.net(images.float())
                _, predicted = torch.max(outputs, 1)
                c = (predicted == genres).squeeze()
                genres = genres.numpy()
                for i in range(genres.size):
                    genre = genres[i]
                    genre_correct[genre] += c[i].item()
                    genre_total[genre] += 1

        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                genre_list[i], 100 * genre_correct[i] / genre_total[i]))


def main():
    print('Loading dataset...')
    apiKey = "BQBNpsweXBRxRQ_3YbpMnizfq76zf2Jt00HHpfQLFw-yU4-AMAy_HNxbF6UhKvX1wv5jqewbI_ppoE6ZRsd9sOun66TfzGlQgMhckMxV2Bwn7wlJQoNJZ9hdjR4QfoxjlhwJAwMe7PXcwtI"
    debug = False
    data = loadData.Dataset(apiKey, debug)

    print("Preprocessing...")
    cnnTrainData, cnnTestData = data.preprocessCNN()

    print("Training...")
    Train(cnnTrainData, cnnTestData)


if __name__ == '__main__':
    main()