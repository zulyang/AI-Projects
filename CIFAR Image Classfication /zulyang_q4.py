import csv
import math
from tensorflow import keras

import cv2
import numpy as np
from keras import Model
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.layers import MaxPooling2D, Conv2D, Reshape, Dense, Flatten
import tensorflow as tf
from keras.datasets import cifar10
import sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Class names for different classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

# Load training data, labels; and testing data and their true labels
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print ('Training data seize:', train_images.shape, 'Test data size', test_images.shape)

# Normalize pixel values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0


# Upsize all training and testing images to 96x96 for use with mobile net
minSize = 96 #minimum size requried for mobileNet
#<Write code> You may use cv2 package. Look for function:

#Resize Images
resized_train_images = []
for image in train_images:
    resized_train_images.append(cv2.resize(image, dsize=(minSize, minSize), interpolation=cv2.INTER_CUBIC))
train_images = np.array(resized_train_images)

resized_test_images = []
for image in test_images:
    resized_test_images.append(cv2.resize(image, dsize=(minSize, minSize), interpolation=cv2.INTER_CUBIC))
test_images = np.array(resized_test_images)

#Import MobileNetV2
mobileNet = MobileNetV2(input_shape=(96,96,3), alpha=1.0, depth_multiplier=1,
include_top=False, weights='imagenet', input_tensor=None, pooling=None)

#Layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from sklearn.metrics import classification_report

model = Sequential()
model.add(mobileNet)
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (3,3,1280)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(1,1), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation=tf.nn.softmax))
model.summary()

# Compile the model with appropriate Loss function
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.train.AdamOptimizer(),
              metrics=['accuracy'])

# Run the stochastic gradient descent for specified epochs
epochs = 50
batch_size = 76

history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs)
print(history)
print(history.history['acc'])

# summarize history for loss
plt.plot(history.history['loss'])

# summarize history for accuracy
plt.plot(history.history['acc'])

plt.title('model')

plt.xlabel('epoch')
plt.legend(['loss', 'accuracy'], loc='upper left')
plt.savefig('assignment2_model_plot.png')

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


Y_test = np.argmax(test_labels, axis=1) # Convert one-hot to index
y_true = test_labels.flatten()
y_pred = model.predict(test_images)
y_pred = np.array(y_pred)
# print(type(y_true[0]))
# print(type(y_pred[0]))
y_classes = np.argmax(y_pred, axis=1)
print('Classification report: ', classification_report(y_true, y_classes))

from keras.models import model_from_json

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# # serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('my_model_weights.h5')
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('my_model_weights.h5')
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
test_loss, test_acc = loaded_model.evaluate(test_images, test_labels)

print('Loaded test accuracy:', test_acc)
print('Loaded test loss:', test_loss)
