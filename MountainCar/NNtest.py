from keras.models import Sequential
from keras.layers import Dense
import numpy as np


model = Sequential()
model.add(Dense(200, input_dim=2, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=100)