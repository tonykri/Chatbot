import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class MLmodelClass:
    def __init__(self, inputLen, outputLen):
        self.model = Sequential()
        self.model.add(Dense(50, activation='relu', input_dim=inputLen))
        self.model.add(Dense(20, activation='relu'))
        self.model.add(Dense(30, activation='relu'))
        self.model.add(Dense(outputLen, activation='softmax'))

        self.model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    
    def train(self, x, y, epochs=500, verbose=2):
        self.model.fit(x, y, epochs=500, verbose=2)


    def predict(self, x):
        pred = self.model.predict(x)
        return np.argmax(pred)
    