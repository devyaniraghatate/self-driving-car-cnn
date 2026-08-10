"""
model.py
--------
NVIDIA End-to-End Self Driving CNN architecture.
Paper: "End to End Learning for Self-Driving Cars" (NVIDIA, 2016)
Input: 66x200x3 YUV image  ->  Output: 1 value (steering angle, regression)
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


def createModel():
    model = Sequential()

    # 5 Convolutional layers (as per NVIDIA paper)
    model.add(Conv2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))

    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))   # output = steering angle (regression, no activation)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    return model