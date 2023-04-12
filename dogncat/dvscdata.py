import tensorflow as tf
from tensorflow import keras
from keras import utils
from keras import layers
from keras import datasets
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout,Flatten,Conv2D,MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

cifar_mnist = datasets.cifar10
(train_images, train_labels), (test_images, test_labels)=cifar_mnist.load_data()

