
import os.path
import pandas as pd
import numpy as np
import json
import shutil
import requests

_API_ROOT = "http://ws.audioscrobbler.com/2.0/"
_NUM_ROWS_TRAIN = 30

class Dataset():

    def __init__(self, param_1):
        self.api_key = param_1

        album_name = []
        album_genre = []
        album_image_url = []

        response = requests.get(_API_ROOT + "?method=tag.gettopalbums&tag=disco&api_key="+ self.api_key + "&format=json")
        data01 = response.json()

        for album in data01['albums']['album']:
            album_name.append(album['name'])
            album_genre.append('disco')
            album_image_url.append(album['image'][3])


        self.train = pd.DataFrame(
            {'name': album_name,
             'genre': album_genre,
             'image_url': album_image_url
             })




if __name__ == '__main__':

    apiKey = "18a7c1e4adc3bc81521a35f3f4f3a7bf"
    data = Dataset(apiKey)

    for index,row in data.train.iterrows():
        print( row['name'] + "  " + row['genre'])

    print("loadData is complete")
