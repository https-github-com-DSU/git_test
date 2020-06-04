from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
import numpy as np
import sys
import yaml


# read clean files
trainX_path, trainY_path = sys.argv[1], sys.argv[2]
X_train = np.load(trainX_path)
Y_train = np.load(trainY_path)
print(X_train.shape)

# Set the CNN model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

with open('params.yaml') as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

# Define the optimizer
optimizer = RMSprop(lr=params['optimizer']['lr'], rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model.compile(optimizer=optimizer,
              loss="categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


epochs = params['train']['epochs']  # Turn epochs to 30 to get 0.9967 accuracy
batch_size = params['train']['batch_size']

# Set the random seed
random_seed = 2

# Training wWithout data augmentation
# history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,
#          validation_data = (X_val, Y_val), verbose = 2)
history = model.fit(X_train, Y_train, batch_size=batch_size,
                    epochs=epochs, verbose=2)

model.save('./model/model.h5')
