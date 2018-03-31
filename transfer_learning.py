from keras import applications
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.preprocessing import image
import numpy as np


IMG_WIDTH = 64
IMG_HEIGHT = 64
TRAIN_DIR = "data/train/"
VALIDATION_DIR = "data/validation/"
TEST_DIR = "data/test/"
batch_size = 32
epochs = 50
num_classes = 12

pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))

for layer in pretrained_model.layers[:-5]:
   layer.trainable = False

model = Sequential()
model.add(Flatten(input_shape = pretrained_model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model = Model(input=pretrained_model.input, output=model(pretrained_model.output))

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.3,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=30)

validation_datagen=ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size, 
    class_mode="categorical")

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = batch_size,
    class_mode = "categorical")

checkpoint = ModelCheckpoint("pretrained_model.h5", monitor = 'val_acc', verbose = 1, save_best_only = True)
early = EarlyStopping(monitor='val_acc', min_delta = 0, patience=10, verbose=1, mode = 'auto')
csvlogger = CSVLogger('logs/transfer_learning_new.csv', separator='\t')

model.fit_generator(
    train_generator,
    epochs = epochs,
    validation_data = validation_generator,
    verbose=1,
    callbacks = [checkpoint, early]
)
model.save('transfer_learning_new.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size = batch_size,
    class_mode = 'categorical'
)
predictions = model.predict_generator(test_generator)
predictions = np.argmax(predictions, axis=-1)
label_map = train_generator.class_indices
label_map = dict((v,k) for k,v in label_map.items())
predictions = [label_map[k] for k in predictions]

print(predictions)
print(test_generator.filenames)  