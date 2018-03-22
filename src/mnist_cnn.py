from keras.layers import MaxPooling2D, Conv2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model, save_model
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import os
import h5py

class DeepCNN:
    def __init__(self, x_train, y_train, x_validation, y_validation, input_shape, num_classes):
        print("Loading Class...")
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.x_train = x_train
        self.y_train = y_train
        self.x_validation = x_validation
        self.y_validation = y_validation

    def define(self):
        model = Sequential()
        model.add(Conv2D(32, (3,3), input_shape=self.input_shape, activation='relu'))
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(Conv2D(128, (3,3), activation='relu'))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        self.model = model

    def compile(self, epochs=10):
        self.optimzer = Adam(lr=0.01)
        self.loss = 'categorical_crossentropy'
        self.epochs = epochs
        self.model.compile(optimizer=self.optimzer, loss=self.loss, metrics=['accuracy'])

    def train(self, batch_size=32):
        self.batch_size = batch_size
        self.model.fit(self.x_train, self.y_train, epochs=self.epochs, validation_data=(self.x_validation, self.y_validation), batch_size=self.batch_size, verbose=2)
        save_model(self.model, os.path.join("..", "models", "trained_model.h5"))
    

    def predict(self, path):
        self.img = load_img(path, target_size=(self.input_shape[0], self.input_shape[1]), grayscale=True)
        self.img = img_to_array(self.img)
        self.img = self.img.astype('float32')
        self.img /= 255
        self.img = np.expand_dims(self.img, axis=0)

        model = load_model(os.path.join("..", "models", "trained_model.h5"))
        predicted = np.argmax(model.predict(self.img))
        print("Predicted Class : {0}".format(predicted))