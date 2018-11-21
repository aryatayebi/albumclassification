
import os
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

    def __init__(self, param_1):
        """  Loads data from last.fm API

            ARGS:
                param_1: personal api_key to access last.fm API calls

        """

        self.api_key = param_1

        album_name = []
        album_genre = []
        album_image_url = []
        album_image = []
        genre_list = ['electronic', 'indie', 'pop', 'metal', 'alternative rock', 'classic rock',
                      'jazz', 'folk', 'Hip-Hop', 'Classical']

        for genre in genre_list:
            response = requests.get(_API_ROOT + "?method=tag.gettopalbums&tag=" + genre +
                                    "&limit=100&api_key="+ self.api_key + "&format=json")
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
        all_data = all_data.sample(frac=1).reset_index(drop=True)

        # split data in train/validate/test with 80%/10%/10% of rows from all_data
        self.train = all_data.iloc[:800]
        self.validate = all_data.iloc[800:900]
        self.test = all_data.iloc[900:1000]


if __name__=='__main__':

    apiKey = "18a7c1e4adc3bc81521a35f3f4f3a7bf"
    data = Dataset(apiKey)

    # sanity check for correct number of rows
    assert data.train.shape[0] == 800
    assert data.validate.shape[0] == 100
    assert data.validate.shape[0] == 100

    print("loadData is complete")
