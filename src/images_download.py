import os
import pandas as pd 
import numpy as np

DATASET_FOLDER = "/Users/viswa/Library/CloudStorage/OneDrive-IITHyderabad/AKSHINTALA VENKATA MAHVITH KUSUMAKAR's files - Amazon_ML/student_resource/dataset"
train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
# test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
# sample_test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
# sample_test_out = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test_out.csv'))

from utils import download_images
download_images(train['image_link'], '../train_images')