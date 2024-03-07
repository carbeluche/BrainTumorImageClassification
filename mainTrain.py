import os
import cv2
import numpy as np
from PIL import Image
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import normalize
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

image_directory = "datasets/"

no_tumor_images = os.listdir(os.path.join(image_directory, 'no'))
yes_tumor_images = os.listdir(os.path.join(image_directory, 'yes'))
dataset = []
label = []
INPUT_SIZE = 64

for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[-1] == 'jpg':
        image_path = os.path.join(image_directory, 'no', image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[-1] == 'jpg':
        image_path = os.path.join(image_directory, 'yes', image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(1)

# print(len(dataset))
# print(len(label))
# needs to be 3000 as it is reading all the images from the dataset

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label,
                                                    test_size=0.2, random_state=0)

# print(x_train.shape)  # (2400, 64, 64, 3) = (number, width, height, number_channels)
# print(y_train.shape)

# print(x_test.shape)  # (600, 64, 64, 3) = (number, width, height, number_channels)
# print(y_test.shape)

# normalising the data
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train = to_categorical(y_train,num_classes=2)
y_test = to_categorical(y_test,num_classes=2)

# Model Building
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation('softmax'))

# Binary Crossentropy = 1 , sigmoid
# Cross Entropy = 2 , softmax

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_split=0.2, shuffle=False)

model.save('BrainTumor10EpochsCategorical.h5')


