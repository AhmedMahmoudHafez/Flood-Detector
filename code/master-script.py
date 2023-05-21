




import pickle 
import numpy as np 
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf 
from tensorflow import keras 
from focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2 
from matplotlib import pyplot as plt 

import os 

folder_path= 'test_images'

file_names=os.listdir(folder_path)

for file_name in file_names : 
    
    image = cv2.imread(f'test_images/{file_name}', cv2.IMREAD_COLOR)
    print(image)
    image_rgb =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, (160, 160))
    print(resized_image.shape)
    model = keras.models.load_model("Deep_model.h5")
    print(model.predict(resized_image.reshape((1,160,160,3)))>0)

