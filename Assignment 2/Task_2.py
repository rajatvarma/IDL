# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, RNN, LSTM, Flatten, TimeDistributed, LSTMCell
from tensorflow.keras.layers import RepeatVector, Conv2D, SimpleRNN, GRU, Reshape, ConvLSTM2D, Conv2DTranspose

# %%
from scipy.ndimage import rotate


# Create plus/minus operand signs
def generate_images(number_of_images=50, sign='-'):
    blank_images = np.zeros([number_of_images, 28, 28])  # Dimensionality matches the size of MNIST images (28x28)
    x = np.random.randint(12, 16, (number_of_images, 2)) # Randomized x coordinates
    y1 = np.random.randint(6, 10, number_of_images)       # Randomized y coordinates
    y2 = np.random.randint(18, 22, number_of_images)     # -||-

    for i in range(number_of_images): # Generate n different images
        cv2.line(blank_images[i], (y1[i], x[i,0]), (y2[i], x[i, 1]), (255,0,0), 2, cv2.LINE_AA)     # Draw lines with randomized coordinates
        if sign == '+':
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA) # Draw lines with randomized coordinates
        if sign == '*':
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)
            # Rotate 45 degrees
            blank_images[i] = rotate(blank_images[i], -50, reshape=False)
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)
            blank_images[i] = rotate(blank_images[i], -50, reshape=False)
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)

    return blank_images

def show_generated(images, n=5):
    plt.figure(figsize=(2, 2))
    for i in range(n**2):
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.imshow(images[i])
    plt.show()

show_generated(generate_images())
show_generated(generate_images(sign='+'))

# %%
def create_data(highest_integer, num_addends=2, operands=['+', '-']):
    """
    Creates the following data for all pairs of integers up to [1:highest integer][+/-][1:highest_integer]:

    @return:
    X_text: '51+21' -> text query of an arithmetic operation (5)
    X_img : Stack of MNIST images corresponding to the query (5 x 28 x 28) -> sequence of 5 images of size 28x28
    y_text: '72' -> answer of the arithmetic text query
    y_img :  Stack of MNIST images corresponding to the answer (3 x 28 x 28)

    Images for digits are picked randomly from the whole MNIST dataset.
    """

    num_indices = [np.where(MNIST_labels==x) for x in range(10)]
    num_data = [MNIST_data[inds] for inds in num_indices]
    image_mapping = dict(zip(unique_characters[:10], num_data))
    image_mapping['-'] = generate_images()
    image_mapping['+'] = generate_images(sign='+')
    image_mapping['*'] = generate_images(sign='*')
    image_mapping[' '] = np.zeros([1, 28, 28])

    X_text, X_img, y_text, y_img = [], [], [], []

    for i in range(highest_integer + 1):      # First addend
        for j in range(highest_integer + 1):  # Second addend
            for sign in operands: # Create all possible combinations of operands
                query_string = to_padded_chars(str(i) + sign + str(j), max_len=max_query_length, pad_right=True)
                query_image = []
                for n, char in enumerate(query_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    query_image.append(image_set[index].squeeze())

                result = eval(query_string)
                result_string = to_padded_chars(result, max_len=max_answer_length, pad_right=True)
                result_image = []
                for n, char in enumerate(result_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    result_image.append(image_set[index].squeeze())

                X_text.append(query_string)
                X_img.append(np.stack(query_image))
                y_text.append(result_string)
                y_img.append(np.stack(result_image))

    return np.stack(X_text), np.stack(X_img)/255., np.stack(y_text), np.stack(y_img)/255.

def to_padded_chars(integer, max_len=3, pad_right=False):
    """
    Returns a string of len()=max_len, containing the integer padded with ' ' on either right or left side
    """
    length = len(str(integer))
    padding = (max_len - length) * ' '
    if pad_right:
        return str(integer) + padding
    else:
        return padding + str(integer)


# %%
# Illustrate the generated query/answer pairs

unique_characters = '0123456789+- '       # All unique characters that are used in the queries (13 in total: digits 0-9, 2 operands [+, -], and a space character ' '.)
highest_integer = 99                      # Highest value of integers contained in the queries

max_int_length = len(str(highest_integer))# Maximum number of characters in an integer
max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
max_answer_length = 3    # Maximum length of the answer string (the longest resulting query string is ' 1-99'='-98')

# Create the data (might take around a minute)
(MNIST_data, MNIST_labels), _ = tf.keras.datasets.mnist.load_data()
X_text, X_img, y_text, y_img = create_data(highest_integer)
print(X_text.shape, X_img.shape, y_text.shape, y_img.shape)


## Display the samples that were created
def display_sample(n):
    labels = ['X_img:', 'y_img:']
    for i, data in enumerate([X_img, y_img]):
        plt.subplot(1,2,i+1)
        # plt.set_figheight(15)
        plt.axis('off')
        plt.title(labels[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
    print('='*50, f'\nQuery #{n}\n\nX_text: "{X_text[n]}" = y_text: "{y_text[n]}"')
    plt.show()

for _ in range(10):
    display_sample(np.random.randint(0, 10000, 1)[0])

# %%
# One-hot encoding/decoding the text queries/answers so that they can be processed using RNNs
# You should use these functions to convert your strings and read out the output of your networks

def encode_labels(labels, max_len=3):
  n = len(labels)
  length = len(labels[0])
  char_map = dict(zip(unique_characters, range(len(unique_characters))))
  one_hot = np.zeros([n, length, len(unique_characters)])
  for i, label in enumerate(labels):
      m = np.zeros([length, len(unique_characters)])
      for j, char in enumerate(label):
          m[j, char_map[char]] = 1
      one_hot[i] = m

  return one_hot


def decode_labels(labels):
    pred = np.argmax(labels, axis=1)
    predicted = ''.join([unique_characters[i] for i in pred])

    return predicted

X_text_onehot = encode_labels(X_text)
y_text_onehot = encode_labels(y_text)

print(X_text_onehot.shape, y_text_onehot.shape)

# %% [markdown]
# ## Text to Text model

# %%
def build_text2text_model():

    # We start by initializing a sequential model
    text2text = tf.keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    text2text.add(LSTM(256, input_shape=(None, len(unique_characters)), return_sequences=True))


    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    text2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    text2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    text2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    text2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    text2text.summary()

    return text2text

# %%
# Build the model
text2text_model = build_text2text_model()

# One-hot encode the labels for training
X_text_onehot = encode_labels(X_text, max_len=max_query_length)
y_text_onehot = encode_labels(y_text, max_len=max_answer_length)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_text_onehot, y_text_onehot, test_size=0.2, random_state=42
    
)

# Add Early Stopping to monitor validation loss
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
history = text2text_model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,        # Maximum number of epochs to train
    batch_size=64,    # Batch size for training
    callbacks=[early_stop]
)

# Plot the training history
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# %% [markdown]
# I want to test if this model is working and predicting correctly:
# 
# 

# %%
# Decode the labels from one-hot to text
def decode_labels(labels):
    """
    Decode one-hot encoded labels into their original string representation.
    """
    decoded = []
    for seq in labels:  # Iterate over each sequence
        pred = np.argmax(seq, axis=1)  # Find the index of the max value for each timestep
        predicted = ''.join([unique_characters[i] for i in pred])  # Convert indices to characters
        decoded.append(predicted)
    return decoded

# Predict on the test data
predictions = text2text_model.predict(X_test)

# Decode predictions and ground truths
decoded_predictions = decode_labels(predictions)
decoded_ground_truths = decode_labels(y_test)

# Display a few test samples with predictions
print("Sample Predictions:")
for i in range(5):  # Adjust the range for the number of examples you want to display
    print(f"Test Query #{i + 1}")
    input_query = ''.join([unique_characters[np.argmax(x)] for x in X_test[i]])  # Decode input query
    print(f"Input Query: {input_query}")
    print(f"True Output: {decoded_ground_truths[i]}")
    print(f"Predicted Output: {decoded_predictions[i]}")
    print("=" * 50)

# Calculate test accuracy
correct = sum([pred.strip() == true.strip() for pred, true in zip(decoded_predictions, decoded_ground_truths)])
accuracy = correct / len(decoded_predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# %% [markdown]
# ## Image to text model

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    ConvLSTM2D, BatchNormalization, MaxPooling3D, Dropout,
    Flatten, RepeatVector, LSTM, TimeDistributed, Dense
)
from tensorflow.keras import regularizers
def build_image2text_model_optimized(input_shape, vocab_size, max_answer_length):
    image2text = Sequential()

    # ConvLSTM2D Encoder
    image2text.add(ConvLSTM2D(
        filters=16,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        return_sequences=True,
        input_shape=input_shape,
        kernel_regularizer=regularizers.l2(0.001)
    ))
    image2text.add(BatchNormalization())
    image2text.add(MaxPooling3D(pool_size=(1, 2, 2)))
    image2text.add(Dropout(0.2))

    image2text.add(ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        return_sequences=False,
        kernel_regularizer=regularizers.l2(0.001)
    ))
    image2text.add(BatchNormalization())
    image2text.add(Dropout(0.2))

    # Flatten and Prepare for Decoding
    image2text.add(Flatten())
    image2text.add(RepeatVector(max_answer_length))

    # LSTM Decoder
    image2text.add(LSTM(128, return_sequences=True))
    image2text.add(Dropout(0.2))
    image2text.add(LSTM(128, return_sequences=True))
    image2text.add(LSTM(128, return_sequences=True))

    # Output Layer
    image2text.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

    # Compile Model
    image2text.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )

    image2text.summary()
    return image2text


# %%
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Define highest_integer as per the earlier setup
highest_integer = 99

# Load the data
X_text, X_img, y_text, y_img = create_data(highest_integer)

# One-hot encode the labels
y_text_onehot = encode_labels(y_text, max_answer_length)

# Assuming X_img and y_text_onehot were created using your `create_data` function
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_img, y_text_onehot, test_size=0.2, random_state=42
)

# Add channel dimension for ConvLSTM2D
X_train = np.expand_dims(X_train, axis=-1)  # Adds a channel dimension
X_test = np.expand_dims(X_test, axis=-1)

# Ensure shapes are correct
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Build and compile the model
vocab_size = len(unique_characters)
input_shape = (max_query_length, 28, 28, 1)  # Includes channel dimension
max_answer_length = 3  # Maximum length of the output sequence

# Assuming build_image2text_model_optimized is correctly defined
model = build_image2text_model_optimized(input_shape, vocab_size, max_answer_length)

batch_size = 32
epochs = 30
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# Plot training history
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training History")
plt.xlabel("Epochs")
plt.ylabel("Metrics")
plt.show()


# %%
def decode_labels(labels, unique_characters):
    """
    Decodes one-hot encoded labels into their original string representation.
    """
    idx_to_char = {idx: char for idx, char in enumerate(unique_characters)}
    decoded = [''.join([idx_to_char[np.argmax(char)] for char in seq]) for seq in labels]
    return decoded


# %%
# Predict using the trained model
predictions = model.predict(X_test)

# Decode predictions and true labels
decoded_predictions = decode_labels(predictions, unique_characters)
decoded_ground_truths = decode_labels(y_test, unique_characters)

accuracy = 0
for i in range(len(X_test)):  # Adjust the range for the number of examples you want to display
#     # print(f"Test Query #{i + 1}")
#     print(f"Predicted Output: {decoded_predictions[i]}")
#     print(f"True Output: {decoded_ground_truths[i]}")
    if decoded_predictions[i].strip() == decoded_ground_truths[i].strip():
        accuracy += 1

print(f"Test Accuracy: {accuracy / len(X_test) * 100:.2f}%")


# %%
import matplotlib.pyplot as plt

# Display a few test samples and their predictions
for i in range(5):  # Adjust the range to display more or fewer samples
    plt.figure(figsize=(12, 3))
    
    # Display the query images
    for j in range(max_query_length):
        plt.subplot(1, max_query_length, j + 1)
        plt.imshow(X_test[i, j, :, :, 0], cmap="gray")  # Extract single channel
        plt.axis("off")
    
    # Show predictions and ground truths
    plt.suptitle(f"Prediction: {decoded_predictions[i]} | True: {decoded_ground_truths[i]}")
    plt.show()


# %%
correct = 0
total = len(decoded_predictions)

for pred, true in zip(decoded_predictions, decoded_ground_truths):
    if pred.strip() == true.strip():  # Compare after stripping whitespace
        correct += 1

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# %% [markdown]
# ## Text to Image Model

# %%
X_test, X_img, y_test, y_img = create_data(highest_integer, num_addends=3, operands=['+', '-'])
X_text_onehot = encode_labels(X_test, max_len=max_query_length)
y_text_onehot = encode_labels(y_test, max_len=max_answer_length)

shuffle = np.random.permutation(len(X_img))

X_img = X_img[shuffle]
y_img = y_img[shuffle]
X_text_onehot = X_text_onehot[shuffle]
y_text_onehot = y_text_onehot[shuffle]

# Resize y_img to be (3, 25, 25)
# y_img = np.array([[cv2.resize(img_part, (25, 25)) for img_part in img] for img in y_img])

print(X_text_onehot.shape, y_text_onehot.shape)
print(X_img.shape, y_img.shape)


X_train_text_onehot = X_text_onehot[:16000,:,:]
X_test_text_onehot = X_text_onehot[16000:,:,:]
y_train_text_onehot = y_text_onehot[:16000,:,:]
y_test_text_onehot = y_text_onehot[16000:,:,:]
y_train_img = y_img[:16000]
y_test_img = y_img[16000:]
print(decode_labels(X_train_text_onehot[10]), decode_labels(y_train_text_onehot[10]))
print(decode_labels(X_test_text_onehot[10]), decode_labels(y_test_text_onehot[10]))
fig, ax = plt.subplots(1,3)
ax[0].imshow(y_train_img[10][1])
ax[1].imshow(y_train_img[10][2])
ax[2].imshow(y_train_img[10][0])
fig1, ax1 = plt.subplots(1,3)
ax1[0].imshow(y_test_img[10][1])
ax1[1].imshow(y_test_img[10][2])
ax1[2].imshow(y_test_img[10][0])
# plt.imshow(y_test_img[10][0])

# print(X_train_text_onehot.shape, X_test_text_onehot.shape, y_train_text_onehot.shape, y_test_text_onehot.shape)

# %%
# Improved text2img model with LSTMs and ConvLSTM2D
text2img = tf.keras.Sequential()

# Encoder with LSTMs
text2img.add(tf.keras.layers.LSTM(512, input_shape=(5, len(unique_characters)), return_sequences=True))
text2img.add(tf.keras.layers.BatchNormalization())
text2img.add(tf.keras.layers.Dropout(0.3))
text2img.add(tf.keras.layers.LSTM(256, return_sequences=True))
text2img.add(tf.keras.layers.BatchNormalization())
text2img.add(tf.keras.layers.Dropout(0.3))
text2img.add(tf.keras.layers.LSTM(128, return_sequences=True))
text2img.add(tf.keras.layers.BatchNormalization())
text2img.add(tf.keras.layers.Dropout(0.3))

# # GRUs
# text2img.add(tf.keras.layers.GRU(256, return_sequences=True))
# text2img.add(tf.keras.layers.BatchNormalization())
# text2img.add(tf.keras.layers.Dropout(0.3))
# text2img.add(tf.keras.layers.GRU(128, return_sequences=True))
# text2img.add(tf.keras.layers.BatchNormalization())
# text2img.add(tf.keras.layers.Dropout(0.3))

# Reshape and Convolutional Layers
text2img.add(tf.keras.layers.Flatten())
text2img.add(tf.keras.layers.Dense(3 * 7 * 7 * 128, activation='relu'))
text2img.add(tf.keras.layers.Reshape((3 * 7, 7, 128)))  # Reshape to 4D tensor
text2img.add(tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu'))
text2img.add(tf.keras.layers.BatchNormalization())
text2img.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu'))
text2img.add(tf.keras.layers.BatchNormalization())
text2img.add(tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=1, padding='same', activation='relu'))
text2img.add(tf.keras.layers.BatchNormalization())
text2img.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='sigmoid'))
text2img.add(tf.keras.layers.Reshape((3, 28, 28)))

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-6)
text2img.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])
text2img.summary()


# %%
t2i = text2img.fit(X_train_text_onehot[0:15000], y_train_img[0:15000], epochs=50, batch_size=50, validation_data = (X_train_text_onehot[15000:16000], y_train_img[15000:16000]), verbose=1)

# %%
print(t2i)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(t2i.history['accuracy'])
ax[0].plot(t2i.history['val_accuracy'])
ax[0].legend(['train_accuracy', 'val_accuracy'])

ax[1].plot(t2i.history['loss'])
ax[1].plot(t2i.history['val_loss'])
ax[1].legend(['train_loss', 'val_loss'])
plt.show()

# %%
print(X_test_text_onehot.shape)
prediction = text2img.predict(X_test_text_onehot)

# %%

random_index = np.random.randint(0, 4000, 12)

fig, ax = plt.subplots(4, 3)
for i in range(4):
    for j in range(3):
        ax[i, j].imshow(np.hstack(prediction[random_index[i*j]]), cmap='gray')
        ax[i, j].axis('off')
        ax[i, j].set_title(f'{decode_labels(X_test_text_onehot[random_index[i*j]])}={decode_labels(y_test_text_onehot[random_index[i*j]])}')


