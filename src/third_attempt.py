'''
CHANGES:
1. Images sharpened
'''
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
import os
import glob
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten
import h5py

IMG_HEIGHT = 128
IMG_WIDTH = 128
train_dir = os.path.join("data", "train")
imgs = []
labels = []
num_classes = 12
epochs = 20
best_model = os.path.join("models", "best_model3.h5")
trained_model = os.path.join("models", "trained_model.h5")

print("Reading Data...")
files = glob.glob(os.path.join(train_dir, "*/*.png"))
for file in files:
    img = load_img(file, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img = img_to_array(img)
    imgs.append(img)
    
print("Generating Classes...")
for index,folder in enumerate(os.listdir(train_dir)):
  path = os.path.join(train_dir, folder, "*.png")
  for images in glob.glob(path):
    labels.append(index)

imgs = np.asarray(imgs)

encoder = LabelEncoder()
encoder.fit_transform(labels)
labels = to_categorical(labels)

final_img = []
for img in imgs:
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = (25, 40, 50)
    upper = (75, 255, 255)
    mask = cv2.inRange(img, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    bmask = mask > 0
    newimg = np.zeros_like(img, 'float')
    newimg[bmask] = img[bmask]
    final_img.append(newimg)

final_img = np.asarray(final_img)
print(final_img[0], type(final_img[0]))
final_img /= 255

x_train, x_validation, y_train, y_validation = train_test_split(final_img, labels, test_size=0.2)
datagen = ImageDataGenerator(
    rotation_range=180,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
    )
datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(64, (5,5), input_shape=(128, 128, 3), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(128, (5,5), input_shape=(128, 128, 3), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(128, (5,5), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(256, (5,5), input_shape=(128, 128, 3), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(256, (5,5), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))

csvlogger = CSVLogger(filename='log.csv', separator='\t')
model_checkpoint = ModelCheckpoint(best_model, monitor='val_loss', save_best_only=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(
    datagen.flow(x_train, y_train), 
    validation_data=(x_validation, y_validation),
    epochs = epochs,
    verbose=1,
    callbacks = [csvlogger, model_checkpoint])
