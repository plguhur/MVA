import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import numpy as np

from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D,Activation, UpSampling2D, Reshape
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import concatenate, Concatenate
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential, clone_model, Model
from keras.utils import np_utils, to_categorical, Sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D,Activation, UpSampling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.regularizers import l2, l1
from keras.optimizers import Adam
from keras.utils import np_utils

IMAGE_SIZE = 72


def sgd_model(n_classes=3, n_hidden=64):
    sgd = Sequential()
    sgd.add(Dense(n_hidden, activation='relu', input_shape=(IMAGE_SIZE**2,)))
    sgd.add(Dense(n_classes, activation='softmax'))
    return sgd


def adam_model(n_classes=3, n_hidden=64):
    adam = Sequential()
    adam.add(Dense(n_classes, input_shape=(IMAGE_SIZE**2,), activation='softmax'))
    return adam


def convnet_model():
    model = Sequential()
    model.add(Conv2D(32,(5,5), strides=3, activation='relu', input_shape=(IMAGE_SIZE,IMAGE_SIZE,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(16,(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model

def preprocessing(data):
    X, Y = data
    Y = Y.reshape(-1, 3,2)
    yy = Y.transpose(1, 0, 2)
    yy = (yy - np.mean(yy, axis = 0)).transpose(1, 0, 2)
    yy = 180 * np.arctan2(yy[:, :, 0], yy[:, :, 1]) / np.pi
    yy += (yy < 0) * 360
    args = np.argsort(yy, axis = -1)
    Y = np.array([y[arg] for y, arg in zip(Y, args)])
    return X, Y.reshape(-1, 6)

def regression_model(kernel_initializer="he_normal",
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
            activation='relu', reg=0.0, n_output=6):

    model = Sequential()
    model.add(Reshape(input_shape))
    model.add(Conv2D(16, activation=activation, strides=2, input_shape=input_shape,
        kernel_size=(5,5), kernel_regularizer=l2(reg), padding="same",
        kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(16, activation=activation, strides=2, input_shape=input_shape,
        kernel_size=(5,5), kernel_regularizer=l2(reg), padding="same",
        kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2,2), padding="same"))
    model.add(Conv2D(32, (5,5), activation=activation, padding="same", kernel_regularizer=l1(reg), kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(32, (5,5), activation=activation, padding="same", kernel_regularizer=l1(reg), kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2,2), padding="same"))
    model.add(Conv2D(64, (5,5), activation=activation, padding="same", kernel_regularizer=l1(reg), kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (5,5), activation=activation, padding="same", kernel_regularizer=l1(reg), kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2,2), padding="same"))
    model.add(Conv2D(128, (5,5), activation=activation, padding="same", kernel_regularizer=l1(reg), kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (5,5), activation=activation, padding="same", kernel_regularizer=l1(reg), kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2,2), padding="same"))
    model.add(Flatten())
    model.add(Dense(256, activation = activation, kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(128, activation = activation, kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(64, activation=activation, kernel_regularizer=l1(reg), kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(n_output,activation='sigmoid', kernel_initializer=kernel_initializer))
    return model

def autoencoder_model(input_size=(72,72,1)):

    inputs = Input(input_size)
    # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 36
    # conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    # conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 18
    #
    # conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    # conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    #
    # pool3 = MaxPooling2D(pool_size=(2, 2))(drop3) # 9
    #
    # conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    # conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)
    #
    # up5 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4)) # 18
    # merge5 = concatenate([drop3, up5], axis = 3)
    # conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    # conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #
    # up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop3)) # 18
    # merge8 = concatenate([conv2,up8], axis = 3)
    # conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    # conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    # up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    # merge9 = concatenate([conv1,up9], axis = 3)
    # conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    # conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    #
    # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    conv00 = BatchNormalization()(Conv2D(16, (3, 3), activation = "relu", padding = "same")(inputs))
    conv01 = BatchNormalization()(Conv2D(32, (3, 3), activation = "relu", padding = "same")(conv00))
    pool0 = MaxPooling2D((2, 2))(conv01)

    conv10 = BatchNormalization()(Conv2D(32, (3, 3), activation = "relu", padding = "same")(pool0))
    conv11 = BatchNormalization()(Conv2D(64, (3, 3), activation = "relu", padding = "same")(conv10))
    pool1 = MaxPooling2D((2, 2))(conv11)

    conv20 = BatchNormalization()(Conv2D(64, (3, 3), activation = "relu", padding = "same")(pool1))
    conv21 = BatchNormalization()(Conv2D(128, (3, 3), activation = "relu", padding = "same")(conv20))
    conv22 = BatchNormalization()(Conv2D(64, (3, 3), activation = "relu", padding = "same")(conv21))

    up0 = Concatenate()([UpSampling2D((2, 2))(conv22), conv11])
    conv20u = BatchNormalization()(Conv2D(64, (3, 3), activation = "relu", padding = "same")(up0))
    conv21u = BatchNormalization()(Conv2D(32, (3, 3), activation = "relu", padding = "same")(conv20u))


    up1 = Concatenate()([UpSampling2D((2, 2))(conv21u), conv01])
    conv20u = BatchNormalization()(Conv2D(32, (3, 3), activation = "relu", padding = "same")(up1))
    conv21u = BatchNormalization()(Conv2D(16, (3, 3), activation = "relu", padding = "same")(conv20u))
    conv22u = BatchNormalization()(Conv2D(1, (3, 3), activation = "relu", padding = "same")(conv21u))

    return inputs, conv22u
