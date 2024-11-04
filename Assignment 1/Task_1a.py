# %% [markdown]
# # Task 1
# ## 2
# 
# Next, take the two well-known datasets: Fashion MNIST (introduced in Ch 10, p.  298) and CIFAR-10.The first dataset contains 2D (grayscale) images of size 28x28, split into 10 categories; 60,000 images for training  and  10,000  for  testing,  while  the  latter  contains  32x32x3  RGB  images  (50,000/10,000train/test). Apply two reference networks on the fashion MNIST dataset
# 
# ### a
# A multi-layer perceptron

# %%
# Importing libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
# Loading the data
fashion_mnist = keras.datasets.fashion_mnist

# %%
# Creating the train and test sets
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape)  # This is usefull to know for the input_shape of the weight matrix of the model as 
                    # it depends on the number of inputs

# Normalising the pixel values to the range [0, 1]
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Creating a validation set of 10%
validation_size = int(0.1 * len(X_train_full))

X_valid, X_train = X_train_full[:validation_size], X_train_full[validation_size:]
y_valid, y_train = y_train_full[:validation_size], y_train_full[validation_size:]


# %%
# Setting the class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# %% [markdown]
# Now lets build a basic network <br>
# MLP with 2 hidden layers

# %%
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28, 28])) # we know the input shape from earlier
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax")) # 10 because 10 possible classes. Softmax because categorical output

# %%
model.summary() # overview of model and its layers

# %%
model.compile(loss = "sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
# sparse_categorical_crossentropy: we have sparse labels (0 to 9) and the are exclusive
# sgd: =keras.optimiz ers.SGD(lr=???)
# accuracy: we want to compare the accuracies of different models

# %%
# save weights to reset and compare models
initial_weights = model.get_weights()

# %% [markdown]
# Now train the model on the data

# %%
history = model.fit(X_train, y_train, epochs=30, 
                    validation_data=(X_valid, y_valid)) 

# %% [markdown]
# Plotting the accuracy and loss per epoch

# %%
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

# %% [markdown]
# Loss and accuracy on the test set:

# %%
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %% [markdown]
# ------------------------------------------------------
# Now we'll try to get better results by changing parameters and comparing the accuracy and loss to the basic model <br>
# 
# 1. Changing the model layers <br>
# - Increasing the model complexity by adding more layers

# %%
model_im = keras.models.Sequential()
model_im.add(keras.layers.Flatten(input_shape = [28, 28])) # we know the input shape from earlier
model_im.add(keras.layers.Dense(300, activation="relu"))
model_im.add(keras.layers.Dense(200, activation="relu"))
model_im.add(keras.layers.Dense(100, activation="relu"))
model_im.add(keras.layers.Dense(10, activation="softmax"))

model_im.summary()

# %%
model_im.compile(loss = "sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# %%
history = model_im.fit(X_train, y_train, epochs=30, 
                    validation_data=(X_valid, y_valid)) 

score_im = model_im.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score_im[0])
print('Test accuracy:', score_im[1])

# %% [markdown]
# This increased the accuracy and decreased the loss!

# %% [markdown]
# - Using Batch Normalisation <br>
# This normalises the inputs to each layer. This increases the learning rate

# %%
model_im = keras.models.Sequential()
model_im.add(keras.layers.Flatten(input_shape = [28, 28])) # we know the input shape from earlier

model_im.add(keras.layers.Dense(300))
model_im.add(keras.layers.BatchNormalization())
model_im.add(keras.layers.Activation("relu"))

model_im.add(keras.layers.Dense(100))
model_im.add(keras.layers.BatchNormalization())
model_im.add(keras.layers.Activation("relu"))

model_im.add(keras.layers.Dense(10, activation="softmax"))

model_im.summary()

# %%
model_im.compile(loss = "sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

history = model_im.fit(X_train, y_train, epochs=30, 
                    validation_data=(X_valid, y_valid)) 

score_im = model_im.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score_im[0])
print('Test accuracy:', score_im[1])

# %% [markdown]
# This slightly increased the accuracy but the loss increased. This might therefore not be a good adjustment.

# %% [markdown]
# - Adding a dropout layer <br>
# This prevents overfitting of the model

# %%
model_im = keras.models.Sequential()
model_im.add(keras.layers.Flatten(input_shape = [28, 28])) # we know the input shape from earlier
model_im.add(keras.layers.Dense(300, activation="relu"))
model_im.add(keras.layers.Dropout(0.5))

model_im.add(keras.layers.Dense(100, activation="relu"))
model_im.add(keras.layers.Dropout(0.5))

model_im.add(keras.layers.Dense(10, activation="softmax"))

model_im.summary()

# %%
model_im.compile(loss = "sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

history = model_im.fit(X_train, y_train, epochs=30, 
                    validation_data=(X_valid, y_valid)) 

score_im = model_im.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score_im[0])
print('Test accuracy:', score_im[1])

# %% [markdown]
# This gives slightly better results

# %% [markdown]
# - Changing the activation function <br>
# As ReLU can suffer from the dying ReLU problem we can try LeakyReLU

# %%
model_im = keras.models.Sequential()
model_im.add(keras.layers.Flatten(input_shape = [28, 28])) # we know the input shape from earlier
model_im.add(keras.layers.Dense(300))
model_im.add(keras.layers.LeakyReLU(alpha = 0.1))

model_im.add(keras.layers.Dense(100))
model_im.add(keras.layers.LeakyReLU(alpha = 0.1))

model_im.add(keras.layers.Dense(10, activation="softmax"))

model_im.summary()

# %%
model_im.compile(loss = "sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

history = model_im.fit(X_train, y_train, epochs=30, 
                    validation_data=(X_valid, y_valid)) 

score_im = model_im.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score_im[0])
print('Test accuracy:', score_im[1])

# %% [markdown]
# This improved the model!

# %% [markdown]
# 2. Changing the compile function <br>
# 
# - Adjusting the learning rate
# 

# %%
# This is just the basic model again
model_im = keras.models.Sequential()
model_im.add(keras.layers.Flatten(input_shape = [28, 28])) 
model_im.add(keras.layers.Dense(300, activation="relu"))
model_im.add(keras.layers.Dense(100, activation="relu"))
model_im.add(keras.layers.Dense(10, activation="softmax"))

# Manually add the learning rate
learning_rate = 0.01
model_im.compile(loss = "sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate = learning_rate), metrics=["accuracy"])

history = model_im.fit(X_train, y_train, epochs=30, 
                    validation_data=(X_valid, y_valid)) 

score_im = model_im.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score_im[0])
print('Test accuracy:', score_im[1])

# %% [markdown]
# This is already better than before. This is kind of unexpected as the SGD optimizer defaults to lr=0.01 as well. Lets try whether a learning rate of 0.001 would increase it even more

# %%
# This is just the basic model again
model_im = keras.models.Sequential()
model_im.add(keras.layers.Flatten(input_shape = [28, 28])) 
model_im.add(keras.layers.Dense(300, activation="relu"))
model_im.add(keras.layers.Dense(100, activation="relu"))
model_im.add(keras.layers.Dense(10, activation="softmax"))

# Manually add the learning rate
learning_rate = 0.001
model_im.compile(loss = "sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate = learning_rate), metrics=["accuracy"])

history = model_im.fit(X_train, y_train, epochs=30, 
                    validation_data=(X_valid, y_valid)) 

score_im = model_im.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score_im[0])
print('Test accuracy:', score_im[1])

# %% [markdown]
# It now performed worse so a learning rate of 0.01 seems best

# %% [markdown]
# 3. Changing the epoch <br>
# Looking at the plot of the basic model we can see that the accuracy is still increasing and loss decreasing. Let's see what happens if we increase epoch
# 

# %%
model.set_weights(initial_weights) # reset the basic model's weights
history = model.fit(X_train, y_train, epochs=40, # increase to 40
                    validation_data=(X_valid, y_valid)) 

# plot
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

# final score
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %% [markdown]
# The model seems to still be slightly improving. Lets try adding an earlyStopping argument to find the most optimal value for epoch

# %%
model.set_weights(initial_weights) # reset the model's weights

early_stopping = keras.callbacks.EarlyStopping(patience= 7, restore_best_weights= True) # patience at 7 because tendency to stop early
history = model.fit(X_train, y_train, epochs=50, # try 50
                    validation_data=(X_valid, y_valid),
                    callbacks = [early_stopping]) 

# final score
score = model.evaluate(X_test, y_test, verbose=0)
best_epoch = np.argmin(history.history['val_loss']) + 1
print('epochs used:', best_epoch)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %% [markdown]
# -------------------------------------------------------------
# We found that the model increased with the following changes:
# - adding an additional layer: <br> model_im.add(keras.layers.Dense(200, activation="relu"))
# - adding a dropout layer: <br> model_im.add(keras.layers.Dropout(0.5))
# - changing the activation function: <br> model_im.add(keras.layers.LeakyReLU(alpha = 0.1))
# - manually adding a learning rate of 0.01: <br> learning_rate = 0.01 <br>
#     model_im.compile(loss = "sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate = learning_rate), metrics=["accuracy"])
# - adding an earlyStopping argument: <br> early_stopping = keras.callbacks.EarlyStopping(patience= 7,       restore_best_weights= True) <br>
#     history = model.fit(X_train, y_train, epochs=50, # try 50
#                         validation_data=(X_valid, y_valid),
#                         callbacks = [early_stopping]) 
# 
# 
# Lets try to combine them
# 

# %%
model_all = keras.models.Sequential()
model_all.add(keras.layers.Flatten(input_shape = [28, 28]))

model_all.add(keras.layers.Dense(300))
model_all.add(keras.layers.LeakyReLU(alpha = 0.1))
model_all.add(keras.layers.Dropout(0.5))

model_all.add(keras.layers.Dense(200))
model_all.add(keras.layers.LeakyReLU(alpha = 0.1))
model_all.add(keras.layers.Dropout(0.5))

model_all.add(keras.layers.Dense(100))
model_all.add(keras.layers.LeakyReLU(alpha = 0.1))
model_all.add(keras.layers.Dropout(0.5))

model_all.add(keras.layers.Dense(10, activation="softmax"))

model_all.summary()

# %%
learning_rate = 0.01
model_all.compile(loss = "sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate = learning_rate), metrics=["accuracy"])

early_stopping = keras.callbacks.EarlyStopping(patience= 7, restore_best_weights= True) # patience at 7 because tendency to stop early
history = model_all.fit(X_train, y_train, epochs=50, # try 50
                    validation_data=(X_valid, y_valid),
                    callbacks = [early_stopping]) 

# final score
score_all = model_all.evaluate(X_test, y_test, verbose=0)
best_epoch = np.argmin(history.history['val_loss']) + 1
print('epochs used:', best_epoch)
print('Test loss:', score_all[0])
print('Test accuracy:', score_all[1])

# %% [markdown]
# This is better than the basic model which had a test loss of 0.3585... and a test accuracy of 0.8741... . However, the models where I made changes in the layers, activation function and epochs performed better on their own. One of the possible reasons could be that I did not use enough epochs. Even though we added an early stopping argument it still used all 50 epochs. Lets see what happens if we run it another 20 epochs (so 70 total).

# %%
history = model_all.fit(X_train, y_train, epochs=20, # try 20 more
                    validation_data=(X_valid, y_valid),
                    callbacks = [early_stopping]) 

# final score
score_all = model_all.evaluate(X_test, y_test, verbose=0)
best_epoch = np.argmin(history.history['val_loss']) + 1
print('epochs used:', best_epoch)
print('Test loss:', score_all[0])
print('Test accuracy:', score_all[1])

# %% [markdown]
# This indeed improved the model. It now used 68 epochs. It is unfortunately still a bit less accurate than the earlier mentioned models and takes longer to run. Lets see what happens if we combine the models that scored higher than this model (lower loss and higher accuracy). They are:
# - the manual learning rate
# - early stopping

# %%
# This is just the basic model again
model_im = keras.models.Sequential()
model_im.add(keras.layers.Flatten(input_shape = [28, 28])) 
model_im.add(keras.layers.Dense(300, activation="relu"))
model_im.add(keras.layers.Dense(100, activation="relu"))
model_im.add(keras.layers.Dense(10, activation="softmax"))

# Manually add the learning rate
learning_rate = 0.001
model_im.compile(loss = "sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate = learning_rate), metrics=["accuracy"])

early_stopping = keras.callbacks.EarlyStopping(patience= 7, restore_best_weights= True) # patience at 7 because tendency to stop early
history = model_im.fit(X_train, y_train, epochs=50, # try 50
                    validation_data=(X_valid, y_valid),
                    callbacks = [early_stopping]) 

# final score
score_all = model_im.evaluate(X_test, y_test, verbose=0)
best_epoch = np.argmin(history.history['val_loss']) + 1
print('epochs used:', best_epoch)
print('Test loss:', score_all[0])
print('Test accuracy:', score_all[1])

# %% [markdown]
# This strangely enough performed worse again.


