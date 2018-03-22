from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import to_categorical
import os
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.layers import MaxPooling2D, Conv2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.models import load_model, save_model
from keras.optimizers import Adam
import h5py

datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=1.0,
    rotation_range=5.0
)

def load_data():
    train_dir = os.path.join("..", "data", "train")
    test_dir = os.path.join("..", "data", "test")
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    label_names = []
    label_counts = []
    labels = []
    imgs = []

    print("Reading Images....")
    all_img_paths = glob(os.path.join(train_dir, "*/*.png"))
    for path in all_img_paths:
        img = load_img(path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        img = img_to_array(img)
        imgs.append(img)
    imgs = np.array(imgs, dtype='float32')

    print("Generating Classes...")
    for index,folder in enumerate(os.listdir(train_dir)):
        path = os.path.join(train_dir, folder, "*.png")
        label_names.append(folder)
        label_counts.append(len(glob(path)))
        for images in glob(path):
            labels.append(index)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    labels = to_categorical(labels)

    print("Splitting Images...")
    x_train, x_validation, y_train, y_validation = train_test_split(imgs, labels, test_size=0.2)

    print("Augmentating Images...")
    datagen.fit(x_train)
    datagen.fit(x_validation)
    print(x_train.shape, x_validation.shape)
    return (x_train, y_train), (x_validation, y_validation)

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
        print("Creating Model...")
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
        print(self.x_train.shape)
        self.model.fit_generator(datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size), epochs=self.epochs, verbose=2)
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

print("Preparing Data...")
num_classes = 2
input_shape = (150, 150, 3)

(x_train, y_train), (x_validation, y_validation) = load_data()

x_train = x_train.reshape(x_train.shape[0], 150, 150, 3)
x_validation = x_validation.reshape(x_validation.shape[0], 150, 150, 3)
print(x_train.shape, x_validation.shape, y_train.shape, y_validation.shape)

x_train = x_train.astype('float32')
x_validation = x_validation.astype('float32')
print(x_train.shape, x_validation.shape, y_train.shape, y_validation.shape)

x_train /= 255
x_validation /= 255
print(x_train.shape, x_validation.shape, y_train.shape, y_validation.shape)

mnist = DeepCNN(x_train, y_train, x_validation, y_validation, input_shape, num_classes)
mnist.define()
mnist.compile()
mnist.train()
# mnist.predict("test.png")