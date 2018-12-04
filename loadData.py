import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import os
from torchData import PreProcessCnn
import pandas as pd
import numpy as np
import json
import shutil
import requests
import cv2

_API_ROOT = "http://ws.audioscrobbler.com/2.0/"
_NUM_ROWS_TRAIN_PER_GENRE = 100

class Dataset():
    """Dataset class for album classification"""

    def __init__(self, param_1, param_2):
        """  Loads data from last.fm API

            ARGS:
                param_1: personal api_key to access last.fm API calls

        """

        num_rows_train_per_genre = "100"

        if param_2 == True:
            num_rows_train_per_genre = "25"


        self.api_key = param_1

        album_name = []
        album_genre = []
        album_image_url = []
        album_image = []
        genre_list = ['electronic', 'indie']
            #, 'indie', 'pop', 'metal', 'alternative rock', 'classic rock',
            #          'jazz', 'folk', 'Hip-Hop', 'Classical']

        for genre in genre_list:
            response = requests.get(_API_ROOT + "?method=tag.gettopalbums&tag=" + genre +
                                    "&limit=" + num_rows_train_per_genre + ",&api_key="+ self.api_key + "&format=json")
            data01 = response.json()

            if response.status_code == 200:
                for album in data01['albums']['album']:
                    if 'name' in album and 'image' in album and not album['name'] == "" and not album['image'][3]['#text'] == "":
                        album_name.append(album['name'])
                        album_genre.append(genre)
                        album_image_url.append(album['image'][3]['#text'])
                        # print(album['name'])
                        # print(album['image'][3]['#text'])
            else:
                print('Error requesting API for ' + genre)

        for url in album_image_url:
            print(url)
            img = requests.get(url, stream=True)

            if img.status_code == 200:
                with open('tempFile.png', 'wb') as out_file:
                    img.raw.decode_content = True
                    shutil.copyfileobj(img.raw, out_file)

                image = cv2.imread('tempFile.png', 1)
                print(image.shape)
                album_image.append(image)
                os.remove('tempFile.png')

            else:
                print("Error loading image " + url)


        self.train = pd.DataFrame(
            {'name': album_name,
             'genre': album_genre,
             'image_url': album_image_url,
             'image': album_image
             })


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

if __name__=='__main__':

    apiKey = "a613f20df66b863fd728baf41ff909d5"
    debug = True
    data = Dataset(apiKey, debug)

    #for index, row in data.train.iterrows():
    #    print(row['name'] + "  " + row['genre'])

    processing = PreProcessCnn(data)
    """

        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                        num_workers=2)
    
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
    
        classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
        for epoch in range(2):  # loop over the dataset multiple times
    
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs = inputs.numpy()
                print(type(inputs))
                print()
                break
    
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
    
            print('Finished Training')
    """