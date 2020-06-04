import os
import pandas as pd
import numpy as np

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from sklearn.model_selection import train_test_split


# Load the data
train = pd.read_csv("./dataset/raw/train.csv")
print(train.shape)
test = pd.read_csv("./dataset/raw/test.csv")
print(test.shape)

Y_train = train["label"]
print(Y_train[:10])

# Drop 'label' column
X_train = train.drop(labels=["label"], axis = 1)

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)


if not os.path.exists('dataset/preprocess'):
    os.makedirs('dataset/preprocess')

np.save("dataset/preprocess/trainX_processed", X_train)
np.save("dataset/preprocess/trainY_processed", Y_train)

