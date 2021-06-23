import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Remove the Tensorflow Warnings and errors

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# Load the model
model = tf.keras.models.load_model('Le_net_5/')

# extract data from folder for test (digits_images)
image = 9
Path = 'digits_images\\' + str(image) + '.png'

data = cv.imread(Path)
data = cv.cvtColor(data, cv.COLOR_BGR2GRAY) # convert RGB image to gray scale image -> (28, 28, 3) to (28, 28, 1)

# Prediction
test = data.reshape(-1, 28, 28, 1).astype('float')/255.0 # Normalize the data
print('prediction: ',np.argmax(model.predict(test)))

# show the test image
plt.imshow(data)
plt.show()
