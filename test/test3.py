# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#the ML stuff
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#the neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    keras.layers.Dense(10)
])
#the compiler
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
#the model evaluation
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

model2=keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	tf.keras.layers.Dense(1000),
	tf.keras.layers.Dense(1000),
	keras.layers.Dense(10)
	])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss2, test_acc2 = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc2)

if test_acc > test_acc2:
	print("the first model is better")
elif test_acc2 > test_acc :
	print("the second model is better")