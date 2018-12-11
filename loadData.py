import os
import pandas as pd
import numpy as np
import json
import shutil
import requests
import cv2
import math
# import torch
# import torch.utils.data
# import torch.nn.functional as F
# import torch.optim as lfunc

_API_ROOT = "https://api.spotify.com/v1/search"
_NUM_ROWS_TRAIN_PER_GENRE = 100

class Dataset():
    """Dataset class for album classification"""

    def __init__(self, param_1, debug=False):
        """  Loads data from last.fm API
            ARGS:
                param_1: personal api_key to access last.fm API calls
        """

        num_rows = range(0, 1000, 50)

        if debug == True:
            num_rows = range(1)


        self.api_key = param_1

        album_name = []
        album_genre = []
        album_image_url = []
        album_image = []
        genre_list = ['electronic', 'indie', 'pop', 'metal', 'alternative%20rock', 'classic%20rock',
                      'jazz', 'folk', 'rap', 'classical']

        for genre in genre_list:
            for i in num_rows:
                response = requests.get(_API_ROOT + "?q=genre%3A" + genre + "&type=track&limit=50&offset=" + str(i), headers = {"Authorization": "Bearer " + self.api_key})
                data01 = response.json()
                print(i)

                if response.status_code == 200:
                    for item in data01['tracks']['items']:
                        if len(item['album']['images']) > 2:
                            album_name.append(item['album']['name'])
                            album_genre.append(genre)
                            album_image_url.append(item['album']['images'][2]['url'])

                else:
                    print('Error requesting API for ' + genre)
                    exit(1)

        for index, url in enumerate(album_image_url):
            print("Getting URL " + str(index))
            img = requests.get(url, stream=True)

            if img.status_code == 200:
                with open('tempFile.png', 'wb') as out_file:
                    img.raw.decode_content = True
                    shutil.copyfileobj(img.raw, out_file)

                image = cv2.imread('tempFile.png', 1)
                album_image.append(image)
                os.remove('tempFile.png')

            else:
                print("Error loading image " + url)
                exit(1)

        # create data frame with all samples loaded
        self.all_data = pd.DataFrame(
            {'name': album_name,
             'genre': album_genre,
             'image_url': album_image_url,
             'image': album_image
             })

        # drop null rows
        self.all_data = self.all_data.dropna()
        # drop duplicate rows
        self.all_data = self.all_data.drop_duplicates('image_url')
        self.all_data = self.all_data.reset_index()

        # shuffle the data
        # self.all_data = all_data.sample(frac=1).reset_index(drop=True)
        # dataLen = self.all_data.shape[0]
        # length80 = math.floor(dataLen * 0.8)
        # length10 = math.floor(dataLen * 0.1)

        # split data in train/validate/test with 80%/10%/10% of rows from all_data
        # self.train = self.all_data.iloc[:length80]
        # self.validate = self.all_data.iloc[length80:(length80 + length10)]
        # self.test = self.all_data.iloc[(length80 + length10):]

    def preprocessKNN(self):
        """Processes data for kNN
        Returns:
            Train data, validate data, test data
        """

        feature_list = []

        for index, row in self.all_data.iterrows():
            chans = cv2.split(row['image'])

            features = []
            for chan in chans:
                hist = cv2.calcHist(chan, [0], None, [64], [0,256])
                features.extend(hist)

            features = np.array(features).flatten()
            feature_list.append(features)

        df = self.all_data[['name', 'genre']].copy()

        feature_df = pd.DataFrame(feature_list)

        df = df.join(feature_df)

        return df

    def preprocessCNN(self):
        labels = np.zeros(self.all_data.shape[0], dtype=int)
        new_images = []

        self.all_data = shuffle(self.all_data)

        # Grab genres and put into an nparray
        for index, row in self.all_data.iterrows():
            genre = row['genre']

            if (genre == 'electronic'):
                labels[index] = 0

            if (genre == 'indie'):
                labels[index] = 1

            if (genre == 'pop'):
                labels[index] = 2

            if (genre == 'metal'):
                labels[index] = 3

            if (genre == 'alternative%20rock'):
                labels[index] = 4

            if (genre == 'classic%20rock'):
                labels[index] = 5

            if (genre == 'jazz'):
                labels[index] = 6

            if (genre == 'folk'):
                labels[index] = 7

            if (genre == 'rap'):
                labels[index] = 8

            if (genre == 'classical'):
                labels[index] = 9

            img = (row['image'])
            img2 = cv2.resize(img, (64, 64))
            tp = np.transpose(img2)
            new_images.append(tp)

        lSize = labels.size

        # Split Training and Testing into Train and Test Sizes
        trainSize = int(lSize * .8)
        testSize = int(lSize * .2)
        if ((testSize + trainSize) < lSize):
            trainSize = trainSize + 1

        # Numpy To Pytorch Tensor Transformation
        torch_labels = torch.from_numpy(labels)
        new_images = np.array(new_images)
        torch_images = torch.from_numpy(new_images)
        fullDataset = torch.utils.data.TensorDataset(torch_images, torch_labels)

        # Splitting Dataset
        trainDataset, testDataset = torch.utils.data.random_split(fullDataset, [trainSize, testSize])

        return trainDataset, testDataset


if __name__=='__main__':

    apiKey = "BQBhGY0fT-howXYjtpJxTtZJ6eX-bb7vyuVmbQUsOvsh0b4ok4yLLpSp7NgULpRIYI0gN5w83JhFwaVjyam15SfpFY8JK_tl7depMXR77b_t_KXn47zO0lB31UQMkdF3m35yEf01i7JhJQ2uflk"
    debug = False
    data = Dataset(apiKey, debug)

    print(data.all_data.shape)

    print("loadData is complete")