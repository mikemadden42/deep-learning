#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import utils
from keras.datasets import mnist
from typing import Tuple

# Set random seed for reproducibility
np.random.seed(123)

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess input data
X_train: np.ndarray = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test: np.ndarray = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train: np.ndarray = X_train.astype("float32") / 255
X_test: np.ndarray = X_test.astype("float32") / 255

# Preprocess class labels
Y_train: np.ndarray = utils.to_categorical(y_train, 10)
Y_test: np.ndarray = utils.to_categorical(y_test, 10)

# Define model architecture
model: Sequential = Sequential()

model.add(Convolution2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(Convolution2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

# Print model summary
model.summary()

# Compile model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit model on training data
history = model.fit(
    X_train, Y_train, batch_size=32, epochs=10, verbose=1, validation_split=0.2
)

# Evaluate model on test data
score: Tuple[float, float] = model.evaluate(X_test, Y_test, verbose=0)

# Print test accuracy
print("Test Accuracy:", score[1])
