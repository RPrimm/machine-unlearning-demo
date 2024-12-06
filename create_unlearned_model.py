import random

import numpy as np
import tensorflow as tf

"""
ADD DATA FOR DATA AUGMENTATION
"""
# import mnist dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# generates random noise for all images labeled "3"
images = []
for data_item in range(len(X_train)):
    if y_train[data_item] == 3:
        image = X_train[data_item]
        for i in range(28):
            for j in range(28):
                image[i][j] = random.randint(0, 255)

"""
TRAIN NEW UNLEARNED MODEL
"""
# scaling
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# reshape array into one 728 digit array instead of 28*28
X_train_flat = X_train.reshape(len(X_train), (28 * 28))
X_test_flat = X_test.reshape(len(X_test), (28 * 28))

# model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# training
model.fit(X_train, y_train, epochs=5)

# testing
model.evaluate(X_test, y_test)

# save model
model.save('./models/unlearned_model.keras')
