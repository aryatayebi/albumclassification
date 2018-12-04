import torch
import torch.nn.functional as F
import torch.optim as lfunc
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
        print(x.size())
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Train():
    def __init__(self, trainloader):
        self.net = Cnn()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = lfunc.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.training(trainloader)

    def training(self, trainloader):
        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                #inputs = inputs.numpy()
                #print((inputs.shape))
                #print()

                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs.float())
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')





class PreProcessCnn():

    def imshow(self, img):
        img = img
        npimg = img.numpy()
        plt.imshow(npimg)
        plt.show()

    def __init__(self, data):

        labels = np.zeros(1, dtype=int)
        #print(type(data.train.album_image))
        a = []
        for index, row in data.train.iterrows():
            genre = row['genre']
            if (genre == 'electronic'):
                labels[0] = 1
            img = (row['image'])
            img2 = cv2.resize(img,(32, 32))
            tp = np.transpose(img2)
            a.append(tp)
        a = np.array(a)
        labels = torch.from_numpy(labels)
        train_data = torch.utils.data.TensorDataset(torch.from_numpy(a), labels)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=2)
        trainData = Train(trainloader)

"""
    if (genre == 'indie'):
        labels.append(0)
    if (genre == 'pop'):
        labels.append(0)
    if (genre == 'metal'):
        labels.append(0)
    if (genre == 'alternative rock'):
        labels.append(0)
    if (genre == 'classic rock'):
        labels.append(0)
    if (genre == 'jazz'):
        labels.append(0)
    if (genre == 'folk'):
        labels.append(0)
    if (genre == 'Hip-Hop'):
        labels.append(0)
    if (genre == 'Classical'):
        labels.append(0)
"""