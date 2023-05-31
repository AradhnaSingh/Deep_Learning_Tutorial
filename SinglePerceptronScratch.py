import numpy as np
import numpy as np

class Perceptron:
    def __init__(self,input_size,learning_rate=.01): 
       # self.weight=np.zeros(input_size + 1)
        self.weights=np.random.rand(input_size + 1)
        self.learning_rate=learning_rate 
    
    def predict(self,input):
        summation=np.dot(input,self.weights[1:])+self.weights[0]
        activation=1 if summation>0 else 0
        return activation

    def train(self,input,output):
        error=output - self.predict(input)
        self.weights[1:]+=self.learning_rate*error*input
        self.weights[0]+=self.learning_rate*error

# Creating an instance of the perceptron
perceptron = Perceptron(input_size=2)

# Defining a training dataset (OR gate)
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_outputs = np.array([0, 1, 1, 1])

# Training the perceptron
epochs = 10
for _ in range(epochs):
    for inputs, target in zip(training_inputs, target_outputs):
        perceptron.train(inputs, target)

# Testing the perceptron
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for inputs in test_inputs:
    print(f"Input: {inputs}, Predicted Output: {perceptron.predict(inputs)}")
        
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

# Creating the perceptron model
model = Sequential()
model.add(Dense(units=1, activation='sigmoid', input_shape=(2,)))

# Compiling the model
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Defining a training dataset (OR gate)
training_inputs = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
target_outputs = tf.constant([[0], [1], [1], [1]], dtype=tf.float32)

# Training the perceptron
model.fit(training_inputs, target_outputs, epochs=10)

# Testing the perceptron
test_inputs = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
predictions = model.predict(test_inputs)
for i, inputs in enumerate(test_inputs):
    print(f"Input: {inputs.numpy()}, Predicted Output: {predictions[i][0]}")




