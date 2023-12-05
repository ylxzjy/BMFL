import os
import numpy as np
from torchvision import datasets

# Setting the Save Folder Path
train_save_file = '../data1/cifar10/train/'
test_save_file = '../data1/cifar10/test/'

# Creating Save Folders
os.makedirs(train_save_file, exist_ok=True)
os.makedirs(test_save_file, exist_ok=True)

# Loading the CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='../data1', train=True, download=True)
test_dataset = datasets.CIFAR10(root='../data1', train=False, download=True)

# Access to training and test data
x_train, y_train = np.array(train_dataset.data), np.array(train_dataset.targets)
x_test, y_test = np.array(test_dataset.data), np.array(test_dataset.targets)

# Save as npy file
np.save(os.path.join(train_save_file, 'data.npy'), x_train)
np.save(os.path.join(train_save_file, 'label.npy'), y_train)
np.save(os.path.join(test_save_file, 'data.npy'), x_test)
np.save(os.path.join(test_save_file, 'label.npy'), y_test)
