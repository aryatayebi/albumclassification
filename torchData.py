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
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Train():
    def __init__(self, train_data, test_data):

        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_data, shuffle=True)

        self.net = Cnn()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = lfunc.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.training(trainloader)
        self.test(testloader)

    def training(self, trainloader):
        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs.float())
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        print('Finished Training')

    def test(self, testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self.net(images.float())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
                100 * correct / total))


def main():
    print('Loading dataset...')
    apiKey = "BQDFkNU70Qr8wmoyZ9Upq3T8SSIsKkjptBtm4H5AaGn9PXhprKYEpg9WseMFEEGbR0XOOk88xa-7uWL_qnXT4MkC9JJ3-b3kNx982HY9XVtnmMuvOAjTDg-31hcFAMNpnDbuBd9kvQ21mz0"
    debug = True
    data = loadData.Dataset(apiKey, debug)

    print("Preprocessing...")
    cnnTrainData, cnnTestData = data.preprocessCNN()

    Train(cnnTrainData, cnnTestData)


if __name__ == '__main__':
    main()