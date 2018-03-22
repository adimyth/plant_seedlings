from mnist_cnn import DeepCNN
import prepare_data
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

print("Preparing Data...")

num_classes = 12
input_shape = (150, 150, 3)

(x_train, y_train), (x_validation, y_validation) = prepare_data.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_validation = x_validation.reshape(x_validation.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_validation = x_validation.astype('float32')

x_train /= 255
x_validation /= 255

y_train = to_categorical(y_train, num_classes)
y_validation = to_categorical(y_validation, num_classes)

mnist = DeepCNN(x_train, y_train, x_validation, y_validation, input_shape, num_classes)
mnist.define()
mnist.compile()
mnist.train()
# mnist.predict("test.png")