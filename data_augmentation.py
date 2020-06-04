import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

# read clean files
trainX = np.load("dataset/preprocess/trainX_processed.npy")
trainY = np.load("dataset/preprocess/trainY_processed.npy")
print(trainX.shape)

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


# manually generate new images
def gen_aug_img(x, y):
    datagen.fit(x)
    x_aug = []
    y_aug = []
    generate_amount = len(x)
    counter = 0
    for bx, by in datagen.flow(x, y, batch_size=1, shuffle=False):
        counter += 1
        if counter > generate_amount:
            break

        x_aug.append(bx)
        y_aug.append(by)

    x_aug = np.concatenate(x_aug, axis=0)
    print(x_aug.shape)
    y_aug = np.vstack(y_aug)
    print(y_aug.shape)

    return x_aug, y_aug


if not os.path.exists('dataset/augment'):
    os.makedirs('dataset/augment')

trainX_aug, trainY_aug = gen_aug_img(trainX, trainY)
np.save("dataset/augment/trainX_aug", trainX_aug)
np.save("dataset/augment/trainY_aug", trainY_aug)

