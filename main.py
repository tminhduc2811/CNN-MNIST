import numpy as np
import matplotlib.pyplot as plt
from keras import *
"""
An example of CNN for Image Classification using MNIST data set
Model used: Input (Gray Image 28*28) => 2 x Convolutional layer(Kernel = 32, Stride = 1, Padding = 1) ->
output(28*28*32) => Pooling layer(2*2) -> output(14*14*32) => FLATTEN(14*14*32 nodes) -
=> Fully connected layer 1 (128 nodes, activation: sigmoid function)
=> Fully connected layer 2 (10 nodes for [0..9], activation: soft-max function)

"""

# Load MNIST dataset from Keras, include 60,000 training set & 10,000 test set
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Set 10,000 from training set for validation set which is used for tuning the parameters of a classifier in MLP
# Also comparing the performances of the prediction algorithms that we create based on the training set
x_val, y_val = x_train[50000:60000, :], y_train[50000:60000]

# 50,000 remaining dataset is used for training
x_train, y_train = x_train[:50000, :], y_train[:50000]

# Input for CNN is a tensor with 4 dimensions (N W H D). Therefore, we need to reshape the input (3 dimensions (N W H))
# Input(N W H) => Input(N W H D), with D = 1
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

# Creating labels for input using one-hot encoding
# For example, a number 4 after being encoded will be like this [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
y_train = utils.np_utils.to_categorical(y_train, 10)
y_test = utils.np_utils.to_categorical(y_test, 10)
y_val = utils.np_utils.to_categorical(y_val, 10)

# After preparing dataset, we need to define the models so that Keras will know what kind of layers we will use
model = Sequential()

# Add 2 Convolutional layers with 32 kernel, and the kernel size: 3*3, activation function is sigmoid
model.add(layers.Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='sigmoid'))

# Add Max-Pooling layer to reduce the size of input
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Flatten layer, converting tensor to vector
model.add(layers.Flatten())

# Add Fully Connected Layer with 128 nodes & sigmoid function for activation
model.add(layers.Dense(128, activation='sigmoid'))

# Finally, an output layer with 10 nodes because the output is for 10 numbers
# However, this time we use soft-max function so that we can get the probability of the output
model.add(layers.Dense(10, activation='softmax'))

# Compiling model, define loss function & method to optimize the loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Start training model
# Batch size = 32, Epochs = 10, Verbose = 1
batch_size = 32
epochs = 5
verbose = 1
H = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(x_val, y_val))

# After training completed, calculate the score with test set
score = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy for the model is: ', score)

# Plot figures for loss & accuracy
fig = plt.figure()
plt.plot(np.arange(0, epochs), H.history['loss'], label='training loss')
plt.plot(np.arange(0, epochs), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, epochs), H.history['acc'], label='accuracy')
plt.plot(np.arange(0, epochs), H.history['val_acc'], label='validation accuracy')
plt.title('Figures for Accuracy & Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss & Accuracy')
plt.legend()
