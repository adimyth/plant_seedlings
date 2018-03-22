from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import to_categorical
import os
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data():
    train_dir = os.path.join("..", "data", "train")
    test_dir = os.path.join("..", "data", "test")
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    label_names = []
    label_counts = []
    labels = []
    imgs = []

    print("Reading Images ....")
    all_img_paths = glob(os.path.join(train_dir, "*/*.png"))
    for path in all_img_paths:
        img = load_img(path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        imgs.append(img)

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
    (x_train, y_train), (x_validation, y_validation) = train_test_split(imgs, labels, test_size=0.2)

    print("Augmentating Images...")
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=1.0,
        rotation_range=5.0
    )
    datagen.fit(x_train)
    datagen.fit(x_validation)

    return (x_train, y_train), (x_validation, y_validation)