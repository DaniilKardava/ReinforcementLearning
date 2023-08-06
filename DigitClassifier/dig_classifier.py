import numpy as np
import tensorflow as tf
from network import Network
from activation_classes import *
from initializers import *
from optimizers import *

(train_images, train_expected), (
    test_images,
    test_expected,
) = tf.keras.datasets.mnist.load_data()


def integer_ohe(orig_data, unique_vals):
    ohe_data = np.zeros([len(orig_data), unique_vals])
    for i in range(len(orig_data)):
        ohe_data[i][orig_data[i]] = 1

    reshaped_ohe = []
    for i in range(len(ohe_data)):
        reshaped_ohe.append(ohe_data[i].reshape(-1, 1))

    return np.array(reshaped_ohe)


# OHE expected network output and flatten into column vector
train_expected = integer_ohe(train_expected, 10)
test_expected = integer_ohe(test_expected, 10)

# Normalize input data
train_images = train_images / 255
test_images = test_images / 255

# Flatten 2D image into column vector
flat_train_images = []
for i in range(len(train_images)):
    flat_train_images.append(train_images[i].flatten().reshape(-1, 1))

flat_test_images = []
for i in range(len(test_images)):
    flat_test_images.append(test_images[i].flatten().reshape(-1, 1))


# Init network
dimensions = [784, 24, 10]
step_size = 0.001
activation_functions = [leaky_relu(0.3), leaky_relu(0.3)]
initializer = HeUniform
adam = Adam(0.9, 0.999, dimensions, ZeroInitializer)
arguments = {
    "Dimensions": dimensions,
    "Step Size": step_size,
    "Layer Activations": activation_functions,
    "Network Initializer": initializer,
    "Optimizer": adam,
}
ann = Network(arguments)

for i in range(1000):

    advantages = []
    # Create lambda expressions that can be evaluated at time of gradient calculation when network output is known.
    for i in train_expected[:1000]:
        advantages.append(lambda output, i=i: output - i)
    advantages = np.array(advantages)

    print(ann.train(flat_train_images[:1000], advantages, train_expected[:1000]))
