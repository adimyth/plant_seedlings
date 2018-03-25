'''
CHANGES:
1. Method to read images changed from (load_img in keras) to (imread in opencv)
2. Image size changed from (128, 128) to (80, 80) to speed up training
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
from keras.callbacks import ModelCheckpoint, CSVLogger
import pickle

IMG_WIDTH = 80
IMG_HEIGHT = 80
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
newimgs = []
for img in imgs:
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
    newimgs.append(clear)  
newimgs = np.asarray(newimgs)
newimgs = newimgs / 255

print("Splitting into train & validation...")
x_train, x_validation, y_train, y_validation = train_test_split(newimgs, labels, test_size=0.1, stratify = labels)

datagen = ImageDataGenerator(
        rotation_range=180,  
        zoom_range = 0.1, 
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1    
    )  
datagen.fit(x_train)

print("Creating Model...")
model = Sequential()

model.add(Conv2D(64, (5, 5), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation='relu'))
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
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

filepath="models/best_weights4.hdf5"
modelcheckpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
csvlogger = CSVLogger("attempt4.csv", separator='\t')

print("Training...")
model.fit_generator(datagen.flow(x_train, y_train), epochs=20, validation_data=(x_validation, y_validation), callbacks=[csvlogger, modelcheckpoint])

print("Saving the model...")
trained_model = "models/trained_model4.hdf5"
model.save(trained_model)

# print("Pickling the ImageDataGenerator Object...")
# with open("plant_seedlings/models/pickled_datagen.pickle", "wb") as file:
#     pickle.dump(datagen, file, protocol=pickle.HIGHEST_PROTOCOL)