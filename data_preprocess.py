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

# Set the random seed
random_seed = 2
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

if not os.path.exists('dataset/preprocess'):
    os.makedirs('dataset/preprocess')

np.save("dataset/preprocess/X_train_processed", X_train)
np.save("dataset/preprocess/Y_train_processed", Y_train)
np.save("dataset/preprocess/X_val_processed", X_val)
np.save("dataset/preprocess/Y_val_processed", Y_val)

