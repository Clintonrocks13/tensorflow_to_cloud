## Load Libraries
import os
import requests, zipfile, io


# load data 
url = requests.get('https://he-s3.s3.amazonaws.com/media/hackathon/deep-learning-challenge-1/identify-the-objects/a0409a00-8-dataset_dp.zip')
data = zipfile.ZipFile(io.BytesIO(url.content))
data.extractall("my_data")