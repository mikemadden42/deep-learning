#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load CIFAR-100 dataset
cifar: tf.keras.datasets.cifar100 = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Use data augmentation
data_augmentation: tf.keras.Sequential = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

# Split the training data into training and validation sets
split: int = int(0.8 * len(x_train))
x_train, x_val = x_train[:split], x_train[split:]
y_train, y_val = y_train[:split], y_train[split:]

# Use pre-trained weights from ImageNet
base_model: Model = tf.keras.applications.ResNet50(
    include_top=False,  # Exclude the top (fully connected) layers
    weights="imagenet",
    input_shape=(32, 32, 3),
    pooling="avg",  # Global average pooling for variable input sizes
)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Extract features using the base model
x: tf.Tensor = base_model.output

# Add custom output layer for CIFAR-100
output_layer: Dense = tf.keras.layers.Dense(100, activation="softmax")(x)

# Create the model
model: Model = tf.keras.Model(inputs=base_model.input, outputs=output_layer)

# Compile the model with a learning rate scheduler
initial_learning_rate: float = 0.001
lr_schedule: tf.keras.optimizers.schedules.ExponentialDecay = (
    tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
    )
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

# Train the model with early stopping
early_stopping: EarlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

model.fit(
    data_augmentation(x_train),
    y_train,
    epochs=50,
    batch_size=64,
    validation_data=(data_augmentation(x_val), y_val),
    callbacks=[early_stopping],
)
