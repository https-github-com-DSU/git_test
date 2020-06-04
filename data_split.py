import os
import numpy as np
import sys
from sklearn.model_selection import train_test_split

trainX_path, trainY_path = sys.argv[1], sys.argv[2]
# read files needed to be splitted
trainX = np.load(trainX_path)
trainY = np.load(trainY_path)
print(trainX.shape)

# Set the random seed
random_seed = 2
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(trainX, trainY, test_size = 0.1, random_state=random_seed)

# create output path & save files
# print(trainX_path.replace('\\', '/'))
# print(trainX_path.replace('\\', '/').split('/'))
split_path = '/'.join(trainX_path.replace('\\', '/').split('/')[:3]) + '/split'
if not os.path.exists(split_path):
    os.makedirs(split_path)


np.save(trainX_path.replace('origin', 'split'), X_train)
np.save(trainY_path.replace('origin', 'split'), Y_train)
print(X_train.shape)
np.save(trainX_path.replace('origin', 'split').replace('train', 'val'), X_val)
np.save(trainY_path.replace('origin', 'split').replace('train', 'val'), Y_val)
print(X_train.shape)




