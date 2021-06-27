import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Remove the Tensorflow warnings

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AvgPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist # Tensorflow Hand Written digit recognition dataset

#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

"""
# Load the mnist dataset

x_train: training image dataset
y_train: training label of image dataset {0, 1, ........, 9}

x_test: test image dataset
y_test: test labels 

"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the dataset to (28, 28, 1) format
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")

# Normalize the dataset -> (set all the pixel values between (0,1))
x_train = x_train/255.0
x_test = x_test/255.0

# Regularization to control over fitting
rg = tf.keras.regularizers.l1(0.01)

# Model Architecture (Le-Net 5)
"""
Le-Net 5 is a free forward convolution neural network architecture.
It consist of 3 convolution and 2 Avj Pool layers with 2 fully connected layers
to solve the hand digit recognition problem.

"""
model = tf.keras.Sequential([

    Conv2D(6, (5, 5), activation='tanh', padding='same', kernel_regularizer=rg),
    BatchNormalization(),
    AvgPool2D((2, 2), 2),

    Conv2D(16, (5, 5), activation='tanh', kernel_regularizer=rg),
    BatchNormalization(),
    AvgPool2D((2, 2), 2),

    Conv2D(120, (5, 5), activation='tanh', kernel_regularizer=rg),
    BatchNormalization(),

    Flatten(),

    Dense(84, activation='tanh', kernel_regularizer=rg),
    Dense(10, activation='softmax', kernel_regularizer=rg)
])

# Call the architecture of model
model.build(input_shape=(1, 28, 28, 1))
print(model.summary())

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

# Train and Test of the model
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)
model.evaluate(x_test, y_test, batch_size=64)

# Save the model
model.save('Le_net_5/')
