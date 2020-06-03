import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

# read clean files
X_train = np.load("dataset/preprocess/X_train_processed.npy")
Y_train = np.load("dataset/preprocess/Y_train_processed.npy")
print(X_train.shape)

# define generator for augmentation & init
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=10,
    zoom_range=0.1,  # Randomly zoom image
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

# manually generate new images
trainX_aug = []
trainY_aug = []
generate_amount = len(X_train)
counter = 0
for bx, by in datagen.flow(X_train, Y_train, batch_size=1, shuffle=False):
    counter += 1
    if counter > generate_amount:
        break

    trainX_aug.append(bx)
    trainY_aug.append(by)

trainX_aug = np.concatenate(trainX_aug, axis=0)
print(trainX_aug.shape)
trainY_aug = np.vstack(trainY_aug)
print(trainY_aug.shape)

np.save("dataset/preprocess/X_train_aug", trainX_aug)
np.save("dataset/preprocess/Y_train_aug", trainY_aug)
