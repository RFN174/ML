# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#the ML stuff
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#the neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(1000),
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
#model prediction
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]



#diplay stuff
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

