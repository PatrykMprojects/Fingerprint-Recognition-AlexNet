# import all libraries that are going to be used
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
import numpy as np

#check the tensorflow version in case need any update

print("Version: ", tf.__version__)
# If programme is using Eager then operations are executed immediately because they are called from Python.
# It helps work with TensorFlow
print("Eager mode: ", tf.executing_eagerly())
# Check if computer is using GPU
print("GPU: ", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# load datasets as numpy (creator make our life easier and convert images data into one dimensional arrays
# training dataset for feeding model
train_images = np.load(
    r'C:\Users\patry\PycharmProjects\AI_efficiency_net\dataset_FVC2000_DB4_B\dataset\np_data\img_train.npy')
train_labels = np.load(
    r'C:\Users\patry\PycharmProjects\AI_efficiency_net\dataset_FVC2000_DB4_B\dataset\np_data\label_train.npy')
# test dataset with unique fingerprint example that will check accuracy of our model at the end
test_images = np.load(
    r'C:\Users\patry\PycharmProjects\AI_efficiency_net\dataset_FVC2000_DB4_B\dataset\np_data\img_real.npy')
test_labels = np.load(
    r'C:\Users\patry\PycharmProjects\AI_efficiency_net\dataset_FVC2000_DB4_B\dataset\np_data\label_real.npy')

# shape of images
print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(lst[4:])
print(lst[5:])

# display of first 81 images whereas set of 80 is labelled 0 and last one is labelled 1
plt.figure(figsize=(160, 160))
for i in range(0, 81):
    plt.subplot(10, 10, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The dataset labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(train_labels[i][0])

# remove comment from below line if you want to display images dataset
# plt.show()


# from training dataset we sliced a bit of data which will be used to validate model
validation_images, validation_labels = train_images[:30], train_labels[:30]
train_images, train_labels = train_images[30:], train_labels[30:]

# check if data size is correct alongside number of labels is accurate
print(validation_images.shape, validation_labels.shape)
print(train_images.shape, train_labels.shape)

# in order to define input pipeline we have to customize data to Tensorflow dataset representation
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))


# function below will be use to normalize and standardize the images
# Also, it will change size for one suitable to use AlexNet network
def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227, 227))
    return image, label


# here we are simply displaying the dataset size (number of images per set)
train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
print("Training data size:", train_ds_size)
print("Test data size:", test_ds_size)
print("Validation data size:", validation_ds_size)

# we are using defined before function to normalize data.
# shuffle data and specify the size
train_ds = (train_ds
            .map(process_images)
            .shuffle(buffer_size=train_ds_size)
            .batch(batch_size=3, drop_remainder=True))
test_ds = (test_ds
           .map(process_images)
           .shuffle(buffer_size=train_ds_size)
           .batch(batch_size=1, drop_remainder=True))
validation_ds = (validation_ds
                 .map(process_images)
                 .shuffle(buffer_size=train_ds_size)
                 .batch(batch_size=2, drop_remainder=True))

# here we are creating augmentation layer to implement it to our model to prevent over fitting and improve accuracy
# that will produce additional data that will be Flip, Rotated, Zoomed

data_augmentation = keras.Sequential(
    [
        keras.layers.experimental.preprocessing.RandomFlip("horizontal",
                                                           input_shape=(227,
                                                                        227,
                                                                        1)),
        keras.layers.experimental.preprocessing.RandomFlip("vertical",
                                                           input_shape=(227,
                                                                        227,
                                                                        1)),
        keras.layers.experimental.preprocessing.RandomRotation(0.1),
        keras.layers.experimental.preprocessing.RandomZoom(0.1),

    ]
)

# AlexNet model network with additional GaussianNoise, data augmentation, dropout layers to improve accuracy
# and prevent over fitting
model = keras.models.Sequential([
    keras.layers.GaussianNoise(0.02, input_shape=(227, 227, 1)),
    data_augmentation,
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
    keras.layers.Dropout(0.1),
    # keras.layers.GaussianNoise(0.5),
    # Fully connected layer
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.1),
    # SoftMax layer
    keras.layers.Dense(10, activation='softmax')
])

# function below will send our epochs to Tensorboard platform and plot them on the graph
root_logdir = os.path.join(os.curdir, "logs\\fit\\")

root_logdir = os.path.join(os.curdir, "logs\\fit\\")


def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

#  here we are compiling our model
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
model.summary()

# we are setting traindata to be fed into our model, setting epochs number and validation data.
history = model.fit(train_ds,
                    epochs=25,
                    validation_data=validation_ds,
                    validation_freq=1,
                    callbacks=[tensorboard_cb])
# display performance of our model on unseen before dataset test
model.evaluate(test_ds)
