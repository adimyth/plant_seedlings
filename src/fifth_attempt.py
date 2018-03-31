'''
CHANGES:
1. Learning Rate adapted
2. Number of Epochs changed 
'''
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

IMG_WIDTH = 70
IMG_HEIGHT = 70
num_classes = 12
train_dir = os.path.join("plant_seedlings", "data", "train")
imgs = []
labels = []

print("Reading Images...")
files = glob(os.path.join(train_dir, "*/*.png"))
for file in files:
    img = cv2.imread(file)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    imgs.append(img)
imgs = np.asarray(imgs)
  
print("Generating Classes...")
for index,folder in enumerate(os.listdir(train_dir)):
  path = os.path.join(train_dir, folder, "*.png")
  for images in glob(path):
    labels.append(index)
encoder = LabelEncoder()
encoder.fit_transform(labels)
labels = to_categorical(labels)

print("Processing Images...")
trainingImgs = []
for img in imgs:
    # Use gaussian blur
    blurImg = cv2.GaussianBlur(img, (5, 5), 0)   
    # Convert to HSV image
    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)  
    # Create mask (parameters - green color range)
    lower_green = (25, 40, 50)
    upper_green = (75, 255, 255)
    mask = cv2.inRange(hsvImg, lower_green, upper_green)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Create bool mask
    bMask = mask > 0  
    # Apply the mask
    clear = np.zeros_like(img, np.uint8)  # Create empty image
    clear[bMask] = img[bMask]  # Apply boolean mask to the origin image
    trainingImgs.append(clear)  # Append image without background
trainingImgs = np.asarray(trainingImgs)
trainingImgs = trainingImgs / 255

x_train, x_validation, y_train, y_test = train_test_split(trainingImgs, labels, test_size=0.1, random_state=7, stratify = labels)
datagen = ImageDataGenerator(
        rotation_range=180,  
        zoom_range = 0.1,  
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True,
        vertical_flip=True
    )  
datagen.fit(x_train)

print("Creating Modle...")
np.random.seed(7)
model = Sequential()

model.add(Conv2D(64, (5, 5), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(256, (5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(256, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.4, 
                                            min_lr=0.00001)
csvlogger = CSVLogger("attempt5.csv", separator='\t')
checkpoint = ModelCheckpoint("plant_seedlings/models/best_model5.h5", save_best_only=True, mode='max')

model.fit_generator(datagen.flow(x_train, y_train, batch_size=75), 
                           epochs=35, validation_data=(x_validation, y_test),verbose=1, 
                           steps_per_epoch=x_train.shape[0], callbacks=[csvlogger, learning_rate_reduction, checkpoint])