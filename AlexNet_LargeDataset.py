import cv2

from tensorflow import keras
import os
import time
import numpy as np


# this function will extract the label since for our gender classification we need only two classes Male or Female
def extract_label(img_path, train=True):
    filename, _ = os.path.splitext(os.path.basename(img_path))

    subject_id, etc = filename.split('__')

    if train:
        gender, lr, finger, _, _ = etc.split('_')
    else:
        gender, lr, finger, _ = etc.split('_')

    gender = 0 if gender == 'M' else 1
    lr = 0 if lr == 'Left' else 1

    if finger == 'thumb':
        finger = 0
    elif finger == 'index':
        finger = 1
    elif finger == 'middle':
        finger = 2
    elif finger == 'ring':
        finger = 3
    elif finger == 'little':
        finger = 4
    return np.array([gender], dtype=np.uint16)


# setting the image size compatible with AlexNet
img_size = 227


# here we are loading the data and normalizing and standardizing
# resizing images for input accepted by AlexNet
def loading_data(path, train):
    print("loading data from: ", path)
    data = []
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_resize = cv2.resize(img_array, (img_size, img_size))
            label = extract_label(os.path.join(path, img), train)
            data.append([label[0], img_resize])
        except Exception as e:
            pass
    data
    return data

# path to datasets on my computer
Real_path = r"C:\Users\patry\PycharmProjects\AI_efficiency_net\SOCOFing\Real"
Easy_path = r"C:\Users\patry\PycharmProjects\AI_efficiency_net\SOCOFing\Altered\Altered-Easy"
# there are two more datasets that can be used
# I had to decrease number of images as my computer could not manage such a big dataset
# Medium_path = r"C:\Users\patry\PycharmProjects\AI_efficiency_net\SOCOFing\Altered\Altered-Medium"
# Hard_path = r"C:\Users\patry\PycharmProjects\AI_efficiency_net\SOCOFing\Altered\Altered-Hard"


# here we are loading dataset for training and test
Easy_data = loading_data(Easy_path, train=True)
# Medium_data = loading_data(Medium_path, train=True)
# Hard_data = loading_data(Hard_path, train=True)
test = loading_data(Real_path, train=False)

# join datasets if we are using more than one
data = np.concatenate([Easy_data], axis=0)

del Easy_data

import random

# shuffle both datasets
random.shuffle(data)
random.shuffle(test)
data

# normalizing raw image by dividing by the maximum value
# flatten normalised data into 1D vectors
img, labels = [], []
for label, feature in data:
    labels.append(label)
    img.append(feature)
train_data = np.array(img).reshape(-1, img_size, img_size, 1)
train_data = train_data / 255.0
from keras.utils.np_utils import to_categorical

# convert raw labels to one-hot vectors

train_labels = to_categorical(labels, num_classes=2)

del data
train_data

train_labels


from tensorflow.keras import optimizers

# AlexNet network
model = keras.models.Sequential([
    # 1st convolution layer
    keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 1)),
    keras.layers.BatchNormalization(),
    # max pooling layer
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    # 2nd convolution layer
    keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    # max pooling
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    # 3rd convolution layer
    keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    # 4th convolution layer
    keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    # 5 th convolution layer
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    # standardize and normalize the input values, after that the input values are transformed through scaling and
    # shifting operations
    keras.layers.BatchNormalization(),
    # max pooling for sub sampling
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    # passing to the fully connected layer therefore flatten to convert into 1D array
    keras.layers.Flatten(),
    # Fully connected layer
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.2),
    # Fully connected layer
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.2),
    # SoftMax layer
    keras.layers.Dense(2, activation='softmax')
])
model.summary()

# function below will send our epochs to Tensorboard platform and plot them on the graph
root_logdir = os.path.join(os.curdir, "logs\\fit\\")

root_logdir = os.path.join(os.curdir, "logs\\fit\\")


def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

model.compile(optimizer=optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
# early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


history = model.fit(train_data, train_labels, batch_size=32, epochs=25,
                    validation_split=0.2, callbacks=[tensorboard_cb], verbose=1)


test


# preparing test dataset for evaluation
test_images, test_labels = [], []

for label, feature in test:
    test_images.append(feature)
    test_labels.append(label)

# normalizing raw image by dividing by the maximum value
# flatten normalised data into 1D vectors
test_images = np.array(test_images).reshape(-1, img_size, img_size, 1)
test_images = test_images / 255.0
del test
# convert raw labels to one-hot vectors
test_labels = to_categorical(test_labels, num_classes=2)
test_images

# evaluation of the model
model.evaluate(test_images, test_labels)
