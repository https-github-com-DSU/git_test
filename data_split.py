import os
import numpy as np

from sklearn.model_selection import train_test_split


# read files needed to be splitted
# trainX = np.load("dataset/preprocess/trainX_processed.npy")
# trainY = np.load("dataset/preprocess/trainY_processed.npy")
trainX = np.load("dataset/augment/origin/trainX_aug.npy")
trainY = np.load("dataset/augment/origin/trainY_aug.npy")
print(trainX.shape)

# Set the random seed
random_seed = 2
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(trainX, trainY, test_size = 0.1, random_state=random_seed)

# create output path & save files
'''
if not os.path.exists('dataset/preprocess/split'):
    os.makedirs('dataset/preprocess/split')

np.save("dataset/preprocess/split/trainX_processed_split", X_train)
np.save("dataset/preprocess/split/trainY_processed_split", Y_train)
print(X_train.shape)
np.save("dataset/preprocess/split/valX_processed_split", X_val)
np.save("dataset/preprocess/split/valY_processed_split", Y_val)
print(X_val.shape)
'''

if not os.path.exists('dataset/augment/split'):
    os.makedirs('dataset/augment/split')

np.save("dataset/augment/split/trainX_aug_split", X_train)
np.save("dataset/augment/split/trainY_aug_split", Y_train)
print(X_train.shape)
np.save("dataset/augment/split/valX_aug_split", X_val)
np.save("dataset/augment/split/valY_aug_split", Y_val)
print(X_train.shape)




