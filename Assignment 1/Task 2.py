# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf

# Import specific layers and utilities from Keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input
from keras.models import Model
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load and preprocess images
images = np.load('75/images.npy')
images = images.astype('float32')

# Function to shuffle and split images and labels into training, validation, and test sets
def initalize():
    global imgs, train_imgs, val_imgs, test_imgs, train_labels, val_labels, test_labels, distributed
    indices = np.random.permutation(images.shape[0])
    imgs = images[indices]

    split_1 = int(18000*0.8)
    split_2 = int(18000*0.9)

    train_imgs = imgs[:split_1]
    val_imgs = imgs[split_1:split_2]
    test_imgs = imgs[split_2:]

    # Normalize the images
    train_imgs = train_imgs / 255.0
    test_imgs = test_imgs / 255.0
    val_imgs = val_imgs / 255.0

    labels = np.load('75/labels.npy')
    labels = labels.astype('int32')
    labels = labels[indices]
    train_labels = labels[:split_1]
    val_labels = labels[split_1:split_2]
    test_labels = labels[split_2:]

    train_imgs = train_imgs.reshape((train_imgs.shape[0], 75, 75, 1))
    val_imgs = val_imgs.reshape((val_imgs.shape[0], 75, 75, 1))
    test_imgs = test_imgs.reshape((test_imgs.shape[0], 75, 75, 1))

# Initialize the data
initalize()
train_imgs

# Function to convert time into 24 separate labels
def conv_time_24(time):
    ntime = 0
    if time[1] > 30:
        ntime = (time[0] + 0.5)
    else:
        ntime = time[0]
    return ntime

# Function to convert time into 720 separate labels
def conv_time_720(time):
    return time[0]*60 + time[1]

conv_time = conv_time_24
while True:
    initalize()
    train_labels_converted = np.array([conv_time(time) for time in train_labels])
    test_labels_converted = np.array([conv_time(time) for time in test_labels])
    val_labels_converted = np.array([conv_time(time) for time in val_labels])

    encoder = LabelEncoder()
    test_labels_encoded = encoder.fit_transform(test_labels_converted.reshape(-1))
    train_labels_encoded = encoder.fit_transform(train_labels_converted.reshape(-1))
    val_labels_encoded = encoder.fit_transform(val_labels_converted.reshape(-1))

    OHencoder = OneHotEncoder(sparse_output=False)
    train_labels_oh = OHencoder.fit_transform(train_labels_encoded.reshape(-1, 1))
    val_labels_oh = OHencoder.fit_transform(val_labels_encoded.reshape(-1, 1))
    print(val_labels_encoded)
    print(val_labels_oh)
    
    # Check if all labels are present in the validation set, if not, reshuffle
    try:
        val_labels_oh = val_labels_oh.reshape((val_labels_oh.shape[0], 24))
        break
    except:
        pass

# Function to calculate common sense error between true and predicted values
def common_sense_error(true, pred):
    true = K.cast(true, 'float32')
    diff1 = K.abs(pred-true)
    diff2 = K.abs(pred+12-true)
    return K.minimum(diff1, diff2)

# Define the input shape for the model
input_shape = (75, 75, 1)
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(kernel_size=(3,3), strides = (2,2), activation="relu", filters=32))
model.add(keras.layers.Conv2D(activation="relu", filters=32, kernel_size=(3,3), input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Conv2D(kernel_size=(3,3), activation="relu", filters=32))
model.add(keras.layers.Conv2D(kernel_size=(3,3), activation="relu", filters=32))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Conv2D(kernel_size=(3,3), activation="relu", filters=64))
model.add(keras.layers.Conv2D(kernel_size=(3,3), activation="relu", filters=64))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=625, activation="relu"))
model.add(keras.layers.Dense(units=512, activation="relu"))
model.add(keras.layers.Dense(units=256, activation="relu"))
model.add(keras.layers.Dense(units=256, activation="relu"))
model.add(keras.layers.Dense(units=128, activation="relu"))
model.add(keras.layers.Dense(units=64, activation="relu"))
model.add(keras.layers.Dense(units=24, activation="softmax"))

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[common_sense_error])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Train the model
model.fit(train_imgs, train_labels_oh, epochs=10, batch_size=256, validation_data=(val_imgs, val_labels_oh), callbacks=[early_stop])

# Predict the labels for the test set
preds = model.predict(test_imgs)
preds = np.argmax(preds, axis=1)

results = encoder.inverse_transform(preds)
accuracy = np.sum(results == test_labels_converted) / len(test_labels_converted)
print(accuracy*100, '%')

# Function to calculate common sense error between true and predicted values for regression
def common_sense_error(true, pred):
    true = K.cast(true, 'float32')
    diff_1 = K.abs(true - pred)
    diff_2 = K.abs(true - (pred + 12))

    return K.minimum(diff_1, diff_2)

# Initialize the data
initalize()

# Function to convert time into a single value
def conv_time(time):
    return round(time[0] + time[1]/60, 3)

train_labels_reg = np.array([conv_time(time) for time in train_labels])
test_labels_reg = np.array([conv_time(time) for time in test_labels])
val_labels_reg = np.array([conv_time(time) for time in val_labels])

# Define the regression model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(activation='relu', filters=32, kernel_size=(3,3), input_shape=(75, 75, 1)))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.Conv2D(filters=32 ,kernel_size=(3,3), activation='relu'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=512, activation='relu'))
model.add(keras.layers.Dense(units=256, activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=1, activation="softplus"))
model.compile(loss="mse", optimizer="adam", metrics=[common_sense_error])

early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 10)

# Train the regression model
model.fit(train_imgs, train_labels_reg, epochs=10, batch_size = 512, validation_data = (val_imgs, val_labels_reg), callbacks = [early_stop])

# Predict the labels for the test set
reg_preds = model.predict(test_imgs)
accuracy = np.mean(abs(reg_preds - test_labels_reg) < 0.16)
print(accuracy*100, '%')

# Split the labels into hours and minutes for the two-headed model
train_hours = train_labels[:, 0]
train_minutes = train_labels[:, 1]

val_hours = val_labels[:, 0]
val_minutes = val_labels[:, 1]

test_hours = test_labels[:, 0]
test_minutes = test_labels[:, 1]

# Define the two-headed model
inp = keras.layers.Input(shape = (75,75,1))
model = keras.layers.Convolution2D(32,kernel_size = (5,5), strides= (2,2), activation = "relu")(inp)
model = keras.layers.MaxPooling2D(pool_size =2)(model)
model = keras.layers.Convolution2D(32,kernel_size = (3,3),activation = "relu")(model)
model = keras.layers.Convolution2D(32,kernel_size = (3,3),activation = "relu")(model)
model = keras.layers.MaxPooling2D(pool_size =2)(model)
model = keras.layers.Convolution2D(64,kernel_size = (3,3),activation = "relu")(model)
model = keras.layers.Convolution2D(64,kernel_size = (1,1),activation = "relu")(model)
model = keras.layers.Flatten()(model)

d = keras.layers.Dense(256,activation = "relu")(model)
d = keras.layers.Dense(256,activation = "relu")(d)
d = keras.layers.Dropout(0.1)(d)
d = keras.layers.Dense(256,activation = "relu")(d)

hour = keras.layers.Dense(256,activation = "relu")(d)
hour = keras.layers.Dense(128,activation = "relu")(hour)
hour = keras.layers.Dense(64,activation = "relu")(hour)
hour = keras.layers.Dense(32,activation = "relu")(hour)
hour = keras.layers.Dense(16,activation = "relu")(hour)
hour = keras.layers.Dense(12,activation= "softmax", name= "hour")(hour)

minute = keras.layers.Dense(256,activation = "relu")(d)
minute = keras.layers.Dense(256,activation = "relu")(minute)
minute = keras.layers.Dense(256,activation = "relu")(minute)
minute = keras.layers.Dense(128,activation = "relu")(minute)
minute = keras.layers.Dense(64,activation = "relu")(minute)
minute = keras.layers.Dense(32,activation = "relu")(minute)
minute = keras.layers.Dense(16,activation = "relu")(minute)
minute = keras.layers.Dense(1, activation = "softplus", name = "minute")(minute)

model = tf.keras.models.Model(inputs=inp, outputs=[hour, minute])
optim = tf.keras.optimizers.Adam()
model.compile(loss=['sparse_categorical_crossentropy', 'mse'], optimizer=optim, metrics=['accuracy',"mae"])

early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 10)

# Train the two-headed model
model.fit(train_imgs, [train_hours, train_minutes], epochs=30, batch_size = 512, validation_data = (val_imgs, [val_hours, val_minutes]), callbacks = [early_stop])

# Predict the labels for the test set
predictions = model.predict(test_imgs)
hour_p = np.argmax(predictions[0], axis = 1)
minutes_p = predictions[1]

accuracy = np.mean(np.abs(hour_p - test_hours) < 1) * np.mean(np.abs(minutes_p - test_minutes) < 5)
print(accuracy*100, '%')

# Transform labels using periodic function
sine_time_train = (train_hours*60 + train_minutes) 
sine_time_test = (test_hours*60 + test_minutes)  
sine_time_valid = (val_hours*60 + val_minutes) 

# Convert time to sine angle
sine_angle_test = (sine_time_test/720)*2*np.pi
sine_angle_train = (sine_time_train/720)*2*np.pi
sine_angle_valid = (sine_time_valid/720)*2*np.pi

# Define the sine regression model
sin_reg = keras.models.Sequential()
sin_reg.add(keras.layers.Conv2D(activation='relu', filters=32, kernel_size=(3,3), strides = (2,2),input_shape=(75, 75, 1)))
sin_reg.add(keras.layers.MaxPooling2D(pool_size=2))
sin_reg.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
sin_reg.add(keras.layers.Conv2D(filters=32 ,kernel_size=(3,3), activation='relu'))
sin_reg.add(keras.layers.MaxPooling2D(pool_size=2))
sin_reg.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
sin_reg.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))

sin_reg.add(keras.layers.Flatten())
sin_reg.add(keras.layers.Dense(units=512, activation='relu'))
sin_reg.add(keras.layers.Dense(units=512, activation='relu'))
sin_reg.add(keras.layers.Dense(units=256, activation='relu'))
sin_reg.add(keras.layers.Dense(units=256, activation='relu'))
sin_reg.add(keras.layers.Dropout(0.2))
sin_reg.add(keras.layers.Dense(units=256, activation='relu'))
sin_reg.add(keras.layers.Dense(units=256, activation='relu'))
sin_reg.add(keras.layers.Dropout(0.2))
sin_reg.add(keras.layers.Dense(units=128, activation='relu'))
sin_reg.add(keras.layers.Dense(units=64, activation='relu'))
sin_reg.add(keras.layers.Dense(units=32, activation='relu'))
sin_reg.add(keras.layers.Dense(units=1, activation="softplus"))
early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 10)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
sin_reg.compile(loss='mse', optimizer= optimizer, metrics=['mae'])
sin_reg.fit(train_imgs, sine_angle_train, epochs=45, batch_size = 512, validation_data = (val_imgs, sine_angle_valid), callbacks = [early_stop])

# Predict the labels for the test set
predictions = sin_reg.predict(test_imgs)

# Function to calculate the difference between predicted and true values
def difference_func(pred,y):
  pred = np.transpose(pred)
  diff_one = np.maximum(pred,y) - np.minimum(pred,y)
  diff_two = np.minimum(pred,y) + 1 - np.maximum(pred,y)
  return np.minimum(diff_one,diff_two)

result = difference_func(predictions,sine_angle_test).reshape(-1)

accuracy = np.mean(result < np.pi/6)
print(accuracy*100, '%')