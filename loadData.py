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

        num_rows_train_per_genre = "50"

        if debug == True:
            num_rows_train_per_genre = "10"


        self.api_key = param_1

        album_name = []
        album_genre = []
        album_image_url = []
        album_image = []
        genre_list = ['electronic', 'indie', 'pop', 'metal', 'alternative%20rock', 'classic%20rock',
                      'jazz', 'folk', 'rap', 'classical']

        for genre in genre_list:
            for i in range(0, 1000, 50):
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

        for url in album_image_url:
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

        # create data frame with all samples loaded
        all_data = pd.DataFrame(
            {'name': album_name,
             'genre': album_genre,
             'image_url': album_image_url,
             'image': album_image
             })

        # shuffle the data
        self.all_data = all_data.sample(frac=1).reset_index(drop=True)
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

        # dataLen = df.shape[0]
        # length80 = math.floor(dataLen * 0.8)
        # length10 = math.floor(dataLen * 0.1)

        # return df.iloc[:length80], df.iloc[length80:(length80 + length10)], df.iloc[(length80 + length10):]

        return df

    # def preprocessCNN(self):
    #     labels = np.zeros(self.all_data.shape[0], dtype=int)
    #     new_images = []
    #
    #     for index, row in self.all_data.iterrows():
    #         genre = row['genre']
    #
    #         if (genre == 'electronic'):
    #             labels[index] = 0
    #
    #         if (genre == 'indie'):
    #             labels[index] = 1
    #
    #         if (genre == 'pop'):
    #             labels[index] = 2
    #
    #         if (genre == 'metal'):
    #             labels[index] = 3
    #
    #         if (genre == 'alternative rock'):
    #             labels[index] = 4
    #
    #         if (genre == 'classic rock'):
    #             labels[index] = 5
    #
    #         if (genre == 'jazz'):
    #             labels[index] = 6
    #
    #         if (genre == 'folk'):
    #             labels[index] = 7
    #
    #         if (genre == 'Hip-Hop'):
    #             labels[index] = 8
    #
    #         if (genre == 'Classical'):
    #             labels[index] = 9
    #
    #         img = (row['image'])
    #         img2 = cv2.resize(img, (32, 32))
    #         tp = np.transpose(img2)
    #         new_images.append(tp)
    #
    #     new_images = np.array(new_images)
    #     torch_images = torch.from_numpy(new_images)
    #     torch_labels = torch.from_numpy(labels)
    #     return torch.utils.data.TensorDataset(torch_images, torch_labels)



if __name__=='__main__':

    apiKey = "BQDRq_Htx58KYyXZ5qstoX9HQyNMpyC_5XcWiFL1Il3O9WciZeWURYzfhxF0NDTXGvgu7FcEHZOg9hLXZHAuVsxy9p0S5rmtNxk5NsdWBUGxxZP1jAFwjh4bBGSNowPHUSA8e8Kdhqkdq2PsTaM"
    debug = True
    data = Dataset(apiKey, debug)

    # sanity check for correct number of rows
    # assert data.train.shape[0] == 80
    # assert data.validate.shape[0] == 10
    # assert data.test.shape[0] == 10

    knnData = data.preprocessKNN()
    # assert trainKNN.shape[0] == 80
    # assert validateKNN.shape[0] == 10
    # assert testKNN.shape[0] == 10

    print(knnData)

    print("loadData is complete")