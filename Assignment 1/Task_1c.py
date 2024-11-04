# %% [markdown]
# # Task 1
# ## 2
# 
# After you have found the best-performing hyperparameter sets, take the 3 best onesand train new models on the CIFAR-10 dataset to see whether your performance gains translate to a different dataset.

# %%
# Importing libraries
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.datasets import load_sample_image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# Lets first import the data

# %%
cifar10 = keras.datasets.cifar10

# %%
# Creating the train and test sets
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

print(X_train_full.shape)

# %% [markdown]
# When we compare this to the fashion_MNIST dataset we can see the following:
# - the dimensions of each image is bigger
# - it has 3 channels instead of one

# %%
batch_size, height, width, channels = X_train_full.shape
input_shape = (height, width, channels)

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

# %% [markdown]
# We'll first try to run the final MLP model on this dataset. For the fashion_mnist dataset this model had an accuracy of 88.2% and a loss of 33.2% on the test set.

# %%
model_mlp = keras.models.Sequential()
model_mlp.add(keras.layers.Flatten(input_shape = input_shape)) # adjusted input shape

model_mlp.add(keras.layers.Dense(300))
model_mlp.add(keras.layers.LeakyReLU(alpha = 0.1))
model_mlp.add(keras.layers.Dropout(0.5))

model_mlp.add(keras.layers.Dense(200))
model_mlp.add(keras.layers.LeakyReLU(alpha = 0.1))
model_mlp.add(keras.layers.Dropout(0.5))

model_mlp.add(keras.layers.Dense(100))
model_mlp.add(keras.layers.LeakyReLU(alpha = 0.1))
model_mlp.add(keras.layers.Dropout(0.5))

model_mlp.add(keras.layers.Dense(10, activation="softmax"))


learning_rate = 0.01
model_mlp.compile(loss = "sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate = learning_rate), metrics=["accuracy"])

early_stopping = keras.callbacks.EarlyStopping(patience= 7, restore_best_weights= True) 
history = model_mlp.fit(X_train, y_train, epochs=70, # try 70
                    validation_data=(X_valid, y_valid),
                    callbacks = [early_stopping]) 

# final score
score_all = model_mlp.evaluate(X_test, y_test, verbose=0)
best_epoch = np.argmin(history.history['val_loss']) + 1
print('epochs used:', best_epoch)
print('Test loss:', score_all[0])
print('Test accuracy:', score_all[1])

# %% [markdown]
# The accuracy seems to still be increasing so lets add 30 more epochs

# %%
history = model_mlp.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks = [early_stopping]) 

# final score
score_all = model_mlp.evaluate(X_test, y_test, verbose=0)
best_epoch = np.argmin(history.history['val_loss']) + 1
print('epochs used:', best_epoch)
print('Test loss:', score_all[0])
print('Test accuracy:', score_all[1])

# %% [markdown]
# This is quite a bad score. The model's performance does not seem to translate to the CIFAR10 dataset. This could be because of the fact that this CIFAR10 dataset is a lot more complex than the fashion_MNIST dataset. The MLP model therefore most likely is not complex enough due to insufficient layers. Also the model was made with data with only 1 channel in mind, the model now flattens CIFAR10's imagages. This results in a loss of information on the arrangement of the pixels. This makes it very hard for the model to recognize patterns.

# %% [markdown]
# Now let's also try the best CNN model. 

# %%
model_cnn1 = keras.models.Sequential([
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

model_cnn1.compile(loss = "sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

model_cnn1.fit(X_train, y_train, batch_size=64, epochs=12, validation_data = (X_valid, y_valid))

# %% [markdown]
# lets do 12 more

# %%
model_cnn1.fit(X_train, y_train, batch_size=64, epochs=12, validation_data = (X_valid, y_valid))

# %%
score = model_cnn1.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %% [markdown]
# This is a big improvement compared to the MLP model. However, compared to the models performance on the fashion_mnist dataset it scores bad....

# %% [markdown]
# Lastly, let's try the second best CNN model. This was the model with a decreased value for the second dropout layer.

# %%
model_cnn2 = keras.models.Sequential([
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
    keras.layers.Dropout(0.25), # decreased dropout
    keras.layers.Dense(10, activation="softmax") # output layer
])

model_cnn2.compile(loss="sparse_categorical_crossentropy", 
                optimizer="sgd", 
                metrics=['accuracy'])

model_cnn2.fit(X_train, y_train, batch_size=64, epochs=12, validation_data=(X_valid, y_valid)) # only 3 to see if it performs similar in the beginning to previous models

# %% [markdown]
# Right now it seems to do better than the previous models at epoch 12. Let's do 12 more.

# %%
model_cnn2.fit(X_train, y_train, batch_size=64, epochs=12, validation_data=(X_valid, y_valid)) 

# %%
score = model_cnn2.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %% [markdown]
# Compared to the other CNN model after 24 epochs this model scored a lot better, especially looking at the loss. Both models still seem to be improving with more epochs. Overall the CNN models seemed to perfrom better than the MLP model. This could be due to the fact that ...


