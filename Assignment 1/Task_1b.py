# %% [markdown]
# # Task 1
# ## 2
# 
# Next, take the two well-known datasets: Fashion MNIST (introduced in Ch 10, p.  298) and CIFAR-10.The first dataset contains 2D (grayscale) images of size 28x28, split into 10 categories; 60,000 images for training  and  10,000  for  testing,  while  the  latter  contains  32x32x3  RGB  images  (50,000/10,000train/test). Apply two reference networks on the fashion MNIST dataset
# 
# ### b
# A convolutional neural network

# %%
# Importing libraries
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.datasets import load_sample_image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
# Loading the data
fashion_mnist = keras.datasets.fashion_mnist

# %%
# Creating the train and test sets
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Check the dimensions 
batch_size, height, width = X_train_full.shape
print(batch_size, height, width)
input_shape = (height, width, 1) #because 1 channel

# As it only has 1 channel we want to reshape it for symmetry
X_train_full = X_train_full.reshape(batch_size, height, width, 1)
X_test = X_test.reshape(X_test.shape[0], height, width, 1)

# Normalising the pixel values to the range [0, 1]
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Creating a validation set of 10%
validation_size = int(0.1 * len(X_train_full))

X_valid, X_train = X_train_full[:validation_size], X_train_full[validation_size:]
y_valid, y_train = y_train_full[:validation_size], y_train_full[validation_size:]

# %%
# For more efficient code we"ll change the data types to float32
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')

# %%
# Lets first try the model from the book
model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation="relu", padding="same",
                        input_shape = input_shape), # No stride because images are not very large
    keras.layers.MaxPooling2D(2), # divides each spatial dimension by 2
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"), # double filters
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2), # divide dimensions by 2
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"), # again double filters
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2), 
    # fully connected network:
    keras.layers.Flatten(), # needs to be 1D for dense network
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5), # droupout of 50% to prevent overfitting
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax") # output layer
])

# %%
model.summary()

# %%
model.compile(loss = "sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# %% [markdown]
# Let's train the model on the data

# %%
model.fit(X_train, y_train, batch_size=64, epochs=12, validation_data = (X_valid, y_valid))

# %% [markdown]
# I put a break in between for the health of my laptop. If needed ill keep adding epochs in steps of 2

# %%
# lets try 2 more. 14 in total now
model.fit(X_train, y_train, batch_size=64, epochs=2, validation_data = (X_valid, y_valid))

# %%
# 2 more, now 16
model.fit(X_train, y_train, batch_size=64, epochs=2, validation_data = (X_valid, y_valid))

# %%
# 4 more, now 20
model.fit(X_train, y_train, batch_size=64, epochs=4, validation_data = (X_valid, y_valid))

# %%
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %% [markdown]
# This took quite a while. In theory this could give an accuracy of 92% on the test set according to the book. This is already better than the results from the MLP model. The official repository mentioned in the assignment contained a file called example_mnist_cnn.py. They made a different cnn model which they ran on the MNIST dataset. They claim to score over 99% accuracy on the test set in only 12 epochs. I used this model with some modifications in the following section to see how it would perform on the fashion_MNIST set.

# %%
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_valid = keras.utils.to_categorical(y_valid, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model_cnn = keras.models.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3,3),
                                activation= 'relu',
                                input_shape=input_shape), #uses a smaller filter size. No padding function is suplied so the default "valid"
                                                        #will be used. This means a reduction in dimension will be applied
        keras.layers.Conv2D(64, (3,3), activation='relu'), #uses less convolutional layers and of smaller size
        keras.layers.MaxPooling2D(pool_size=(2, 2)), # only one time maxpooling used
        keras.layers.Dropout(0.25), # smaller dropout
        # fully connected network:
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax') # output layer
])


model_cnn.compile(loss=keras.losses.categorical_crossentropy, # because output is changed to categorical instead of integers
                optimizer=keras.optimizers.Adadelta(), # uses a different optimizer. Adadelta has an adaptive learning rate so no need for manual tweeking
                metrics=['accuracy'])

model_cnn.fit(X_train, y_train,
        batch_size=128,
        epochs=12,
        verbose=1,
        validation_data=(X_valid, y_valid))

# %%
score = model_cnn.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %% [markdown]
# This performs worse on the data than the example from the book but we can use it to modify things

# %% [markdown]
# Now lets try to edit some things to see if it becomes more efficient. Ideas:
# - add an extra convolutional layer 
# - keep output categories as integers as this won't require convertion
# - decrease dropout last time
# - keep adadelta for the learnig rate
# - change relu to LeakyReLU
# - changing the padding

# %%
# undo the transformation
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis= 1)
y_valid = np.argmax(y_valid, axis=1)

# %%
model_imp = keras.models.Sequential([
        keras.layers.Conv2D(64, 7, padding="same",
                        input_shape = input_shape), # No stride because images are not very large
        keras.layers.LeakyReLU(alpha = 0.1), # add leakyrelu
        keras.layers.MaxPooling2D(2), 
        keras.layers.Conv2D(128, 3, padding="same"),
        keras.layers.LeakyReLU(alpha = 0.1), 
        keras.layers.Conv2D(128, 3, padding="same"),
        keras.layers.LeakyReLU(alpha = 0.1),
        keras.layers.MaxPooling2D(2), # 
        keras.layers.Conv2D(256, 3, padding="same"), 
        keras.layers.LeakyReLU(alpha = 0.1),
        keras.layers.Conv2D(256, 3, padding="same"),
        keras.layers.LeakyReLU(alpha = 0.1),
        keras.layers.MaxPooling2D(2), 
        # fully connected network:
        keras.layers.Flatten(), 
        keras.layers.Dense(128),
        keras.layers.LeakyReLU(alpha = 0.1),
        keras.layers.Dropout(0.5), # droupout of 50% to prevent overfitting
        keras.layers.Dense(64),
        keras.layers.LeakyReLU(alpha = 0.1),
        keras.layers.Dropout(0.25), # change to 25%
        keras.layers.Dense(10, activation="softmax") # output layer
])


model_imp.compile(loss="sparse_categorical_crossentropy", 
                optimizer=keras.optimizers.Adadelta(), # keeep adadelta
                metrics=['accuracy'])

model_imp.fit(X_train, y_train, batch_size=64, epochs=3, validation_data=(X_valid, y_valid)) # only 3 to see if it performs similar in the beginning to previous models


# %%
score = model_imp.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %% [markdown]
# Unfortunately, this performed a lot worse. Lets edit the original code one feature at the time. I'll only do 3 epochs per test because of computation time. <br>
# At 3 epochs the original model performed like this: <br>
# -accuracy: 0.7248 - loss: 0.7464 - val_accuracy: 0.8045 - val_loss: 0.5210 <br>
# 
# - First changing relu to leaky relu

# %%
model_imp = keras.models.Sequential([
        keras.layers.Conv2D(64, 7, padding="same",
                        input_shape = input_shape), # No stride because images are not very large
        keras.layers.LeakyReLU(alpha = 0.1), # add leakyrelu
        keras.layers.MaxPooling2D(2), 
        keras.layers.Conv2D(128, 3, padding="same"),
        keras.layers.LeakyReLU(alpha = 0.1), 
        keras.layers.Conv2D(128, 3, padding="same"),
        keras.layers.LeakyReLU(alpha = 0.1),
        keras.layers.MaxPooling2D(2), # 
        keras.layers.Conv2D(256, 3, padding="same"), 
        keras.layers.LeakyReLU(alpha = 0.1),
        keras.layers.Conv2D(256, 3, padding="same"),
        keras.layers.LeakyReLU(alpha = 0.1),
        keras.layers.MaxPooling2D(2), 
        # fully connected network:
        keras.layers.Flatten(), 
        keras.layers.Dense(128),
        keras.layers.LeakyReLU(alpha = 0.1),
        keras.layers.Dropout(0.5), 
        keras.layers.Dense(64),
        keras.layers.LeakyReLU(alpha = 0.1),
        keras.layers.Dropout(0.5), 
        keras.layers.Dense(10, activation="softmax") # output layer
])


model_imp.compile(loss="sparse_categorical_crossentropy", 
                optimizer="sgd", 
                metrics=['accuracy'])

model_imp.fit(X_train, y_train, batch_size=64, epochs=3, validation_data=(X_valid, y_valid)) # only 3 to see if it performs similar in the beginning to previous models


# %% [markdown]
# It seems to perform slightly worse

# %% [markdown]
# - Decreasing the last dropout percentage to 25%

# %%
model_imp = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation="relu", padding="same",
                        input_shape = input_shape), # No stride because images are not very large
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"), 
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2), 
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"), 
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2), 
    # fully connected network:
    keras.layers.Flatten(), 
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5), 
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.25), # decrease dropout
    keras.layers.Dense(10, activation="softmax") # output layer
])

model_imp.compile(loss="sparse_categorical_crossentropy", 
                optimizer="sgd", 
                metrics=['accuracy'])

model_imp.fit(X_train, y_train, batch_size=64, epochs=3, validation_data=(X_valid, y_valid)) # only 3 to see if it performs similar in the beginning to previous models

# %% [markdown]
# This performs similar to the original but it overfits more as the validation set performs worse. For this reason I believe that completely removing the dropout layer would result in worse results 

# %% [markdown]
# - Using adadelta

# %%
model_imp = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation="relu", padding="same",
                        input_shape = input_shape), # No stride because images are not very large
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"), 
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2), 
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"), 
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2), 
    # fully connected network:
    keras.layers.Flatten(), 
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5), 
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5), # decrease dropout
    keras.layers.Dense(10, activation="softmax") # output layer
])

model_imp.compile(loss="sparse_categorical_crossentropy", 
                optimizer=keras.optimizers.Adadelta(), 
                metrics=['accuracy'])

model_imp.fit(X_train, y_train, batch_size=64, epochs=3, validation_data=(X_valid, y_valid)) # only 3 to see if it performs similar in the beginning to previous models

# %% [markdown]
# That is a lot worse

# %% [markdown]
# - What if we add another convolutional layer

# %%
model_imp = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation="relu", padding="same",
                        input_shape = input_shape), # No stride because images are not very large
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"), 
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2), 
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"), 
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(256, 3, activation='relu', padding= "same"), # additional layer
    keras.layers.MaxPooling2D(2), 
    # fully connected network:
    keras.layers.Flatten(), 
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5), 
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5), 
    keras.layers.Dense(10, activation="softmax") # output layer
])

model_imp.compile(loss="sparse_categorical_crossentropy", 
                optimizer="sgd", 
                metrics=['accuracy'])

model_imp.fit(X_train, y_train, batch_size=64, epochs=3, validation_data=(X_valid, y_valid)) # only 3 to see if it performs similar in the beginning to previous models

# %% [markdown]
# This is worse than the base model.

# %% [markdown]
# - Changing the padding

# %%
model_imp = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation="relu", padding="valid",
                        input_shape = input_shape), 
    keras.layers.MaxPooling2D(2), 
    keras.layers.Conv2D(128, 3, activation="relu", padding="valid"),
    keras.layers.Conv2D(128, 3, activation="relu", padding="valid"), #MaxPooling layer removed as output is smaller with valid padding
    keras.layers.Conv2D(256, 3, activation="relu", padding="valid"), 
    keras.layers.Conv2D(256, 3, activation="relu", padding="valid"),
    keras.layers.MaxPooling2D(2), 
    # fully connected network:
    keras.layers.Flatten(), 
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5), 
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax") 
])

model_imp.compile(loss="sparse_categorical_crossentropy", 
                optimizer="sgd", 
                metrics=['accuracy'])

model_imp.fit(X_train, y_train, batch_size=64, epochs=3, validation_data=(X_valid, y_valid))

# %% [markdown]
# Again this is worse than the base level.

# %% [markdown]
# Unfortunately non of the modifications performed better than the base model.


