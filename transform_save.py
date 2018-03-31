import cv2
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import math
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import pickle

def transform(img):    
    blurImg = cv2.GaussianBlur(img, (5, 5), 0)   
    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)  
    lower_green = (25, 40, 50)
    upper_green = (75, 255, 255)
    mask = cv2.inRange(hsvImg, lower_green, upper_green)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    bMask = mask > 0  
    clear = np.zeros_like(img, np.uint8)  
    clear[bMask] = img[bMask]  
    return clear

IMG_WIDTH = 70
IMG_HEIGHT = 70
num_classes = 12
train_dir = os.path.join("data", "train")
imgs = []
labels = []

files = glob(os.path.join(train_dir, "*/*.png"))
for file in files:
    img = cv2.imread(file)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = transform(img)
    # print()
    cv2.imwrite(file, img)