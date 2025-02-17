# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#
train_images = train_images / 255.0

test_images = test_images / 255.0
#
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
#
# #
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
#
# model.compile(optimizer=tf.train.AdamOptimizer(),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(train_images, train_labels, epochs=5)
#
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# #
# print('Test accuracy:', test_acc)

from Preprocess import tf_analaysis
test_model = tf_analaysis.nn_model()
test_model.build_nn_model(hidden_layer_structure=[({'input_shape': (28,28)}, 'flatten'),
                                                           {'units': 128, 'activation': tf.nn.relu},
                                                           {'units': 10, 'activation': tf.nn.softmax}])

test_model.compile_nn_model(loss='sparse_categorical_crossentropy', metrics=['accuracy','mse'])
test_model.train_model(train_images, train_labels, epochs=1)
test_loss, test_acc =  test_model.evaluate_model(test_images, test_labels)
