'''
CHANGES
1. Added BatchNormalization Layer
2. Filters in Convolutional Layer Changed
3. Logging methods added
'''

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
import os
import glob
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
best_model = os.path.join("models", "best_model2.h5")
trained_model = os.path.join("models", "trained_model2.h5")

print("Reading Images...")
files = glob.glob(os.path.join(train_dir, "*/*.png"))
for file in files:
    img = load_img(file, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img = img_to_array(img)
    imgs.append(img)
imgs = np.asarray(imgs)
imgs /= 255


print("Generating Classes...")
for index,folder in enumerate(os.listdir(train_dir)):
  path = os.path.join(train_dir, folder, "*.png")
  print(path)
  for images in glob.glob(path):
    labels.append(index)

encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
labels = to_categorical(labels)
print(labels[1:10])


x_train, x_validation, y_train, y_validation = train_test_split(imgs, labels, test_size=0.2)
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
    verbose=2,
    callbacks = [csvlogger, model_checkpoint])
model.save(trained_model)