import cv2
from glob import glob, iglob
import numpy as np
from matplotlib import pyplot as plt
import math
import os
import pandas as pd
import shutil
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, multi_gpu_model
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import pickle

class DeepModel:

    def __init__(self, arg_dict, params, validation=False):
        self.__dict__.update(arg_dict)
        self.__dict__.update(params)
        self.validation = validation
        self.create_data()
        self.add_callbacks()
        self.encoder = LabelEncoder()

        self.trainX = self.process_image(self.TRAIN_DIR, "train")
        self.trainY = self.process_labels(self.TRAIN_DIR)

        if self.validation:
            self.validationX = self.process_image(self.VALIDATION_DIR, "validation")
            self.validationY = self.process_labels(self.VALIDATION_DIR)
    
    def read_images(self, path, purpose):
        imgs = []
        if purpose == "test":
            files = glob(os.path.join(path, "*"))
        else:
            files = glob(os.path.join(path, "*/*"))
        for file in files:
            img = cv2.imread(file)
            img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
            imgs.append(img)
        return imgs

    def process_image(self, path, purpose):
        imgs = self.read_images(path, purpose)
        trainingImgs = []
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
            trainingImgs.append(clear)
        trainingImgs = np.asarray(trainingImgs)
        trainingImgs = trainingImgs / 255
        return trainingImgs
        
    def read_labels(self, path):
        labels = []
        files = glob(os.path.join(path, "*/*"))
        for file in files:
            labels.append(file.split('/')[-2])
        return labels

    def process_labels(self, path):
        labels = self.read_labels(path)
        labels = self.encoder.fit_transform(labels)
        labels = pd.DataFrame(labels)[0]
        labels = to_categorical(labels)
        return labels

    def create_data(self):
      if self.validation:
            # trainSize = 0.8
            # validationSize = 0.2
            # length = []
            # directories = os.listdir(self.TRAIN_DIR)
            # for i in directories:
            #     length.append(len(glob(os.path.join(self.TRAIN_DIR, i, "*"))))

            # validation_images = []
            # for i,j in enumerate(directories):
            #     train_end = int(0.8*length[i])

            #     # validation images range
            #     validation_start = train_end 
            #     validation_end = train_end+int(0.2*length[i]/100)
            #     validation_images.append((validation_start, validation_end))

            # for index, j in enumerate(directories):
            #     if not os.path.exists(os.path.join(self.VALIDATION_DIR, j)):
            #         os.makedirs(os.path.join(self.VALIDATION_DIR, str(j)))
            #     src_directory=os.path.join(self.TRAIN_DIR,j)
            #     start, end = validation_images[index]
            #     for file in os.listdir(src_directory)[start:end]:
            #         shutil.move(file, os.path.join(self.VALIDATION_DIR, j))  

            label_freq = {}
            labels = os.listdir(self.TRAIN_DIR)
            for label in labels:
                label_freq[label] = len(os.listdir(os.path.join(self.TRAIN_DIR, label)))

            for label, count in label_freq.items():  
                files = os.listdir(os.path.join(self.TRAIN_DIR, label))
                train_files, validation_files = train_test_split(files, test_size=0.2, random_state=self.GLOBAL_SEED)
                for file in validation_files:
                    if not os.path.exists(os.path.join(self.VALIDATION_DIR, label)):
                        os.makedirs(os.path.join(self.VALIDATION_DIR, label))
                    shutil.move(os.path.join(self.TRAIN_DIR, label, file), os.path.join(self.VALIDATION_DIR, label, file))                              
            return self

    def build_nn(self):
        model = Sequential()

        model.add(Conv2D(64, (5, 5), input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3), activation='relu'))
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
        model.add(Dense(self.num_classes, activation='softmax'))
        if self.ngpus > 2:
            model = multi_gpu_model(model, gpus=self.ngpus)

        model.compile(loss=self.loss, optimizer=Adam(self.learning_rate), metrics=['accuracy'])
        self.model = model
        return self

    def train(self):
        self.model.fit_generator(self.datagen.flow(self.trainX, self.trainY, 
                                batch_size=self.batch_size), 
                                epochs=self.epochs, 
                                validation_data=(self.validationX, self.validationY),verbose=1, 
                                steps_per_epoch=self.trainX.shape[0], 
                                callbacks=self.callbacks_list)
        self.model.save(self.OUTPUT_BASE_NAME + "_model.h5")
        # self.model.fit_generator(
        #     self.trainGenerator,
        #     epochs=self.epochs,
        #     callbacks=self.callbacks_list,
        #     validation_data=self.validationGenerator)
        return self


    def add_callbacks(self):
        filepath = os.path.join(self.MODEL_CHECKPOINT_PATH, self.OUTPUT_BASE_NAME+"_best.h5")
        modelcheckpoint = ModelCheckpoint(filepath)

        filename = os.path.join(self.LOG_PATH, self.NAME)
        csvlogger = CSVLogger(filename, separator='\t')

        self.callbacks_list = [modelcheckpoint, csvlogger]
        return self

    def augment_data(self):

        datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            vertical_flip=True,
            height_shift_range=0.1,
            width_shift_range=0.1,
            rotation_range=180
        )
        datagen.fit(self.trainX)
        self.datagen = datagen
        return self

        # trainDatagen = ImageDataGenerator(
        #     rescale=1./255,
        #     horizontal_flcsvloggerip=True,
        #     vertical_flip=True,
        #     rotation_range=180,
        #     width_shift_range=0.1,
        #     height_shift_range=0.1
        # )
        # trainDatagen.fit(self.trainX)

        # validationDatagen = ImageDataGenerator(rescale=1./255)
        # testDatagen = ImageDataGenerator(rescale=1./255)
        
        # self.trainGenerator = trainDatagen.flow_from_directory(
        #     self.TRAIN_DIR,
        #     target_size=(self.IMG_WIDTH, self.IMG_HEIGHT),
        #     batch_size=self.batch_size,
        # )
        # self.validationGenerator = validationDatagen.flow_from_directory(
        #     directory = self.VALIDATION_DIR,
        #     target_size=(self.IMG_WIDTH, self.IMG_HEIGHT),
        #     batch_size=self.batch_size,
        #     shuffle=False
        # )
        # self.testGenerator = testDatagen.flow_from_directory(
        #     directory=self.TEST_DIR,
        #     target_size=(self.IMG_WIDTH, self.IMG_HEIGHT),
        #     batch_size=self.batch_size
        # )

    def predict(self):
        model = load_model("checkpoints/best_model.h5")
        self.testX = self.process_image(self.TEST_DIR, "test")
        predictions = model.predict(self.testX)
        predictions = np.argmax(predictions, axis=1)
        self.predictions = self.encoder.classes_[predictions]
        return self

    def save_as_csv(self):
        files = glob(os.path.join(self.TEST_DIR, "*"))
        ids = []
        for file in files:
            ids.append(file.split('/')[-1])
        pred = {'file':ids, 'predictions':self.predictions}
        pred = pd.DataFrame(pred)
        pred.to_csv("predictions.csv", index=False)
        return self