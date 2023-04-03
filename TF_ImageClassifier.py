"""
TensorFlow Image classifier using Fashion MNIST dataset from Keras:
https://keras.io/api/datasets/fashion_mnist/#loaddata-function
Expected outputs:
    0 -> T-shirt/top
    1 -> Trouser
    2 -> Pullover
    3 -> Dress
    4 -> Coat
    5 -> Sandal
    6 -> Shirt
    7 -> Sneaker
    8 -> Bag
    9 -> Ankle boot
"""
import matplotlib.pyplot as plt
# import numpy as np
# import ssl
import random
import tensorflow as tf
from tensorflow import keras


# Load a pre-defined dataset (70k images of 28x28).
fashion_mnist = keras.datasets.fashion_mnist

# Pull out data from dataset (60k images for training, 10k images for testing).
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Show data.
# print(train_images[0])
# print(train_labels[0])
# plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)
# plt.show()

# Define neural net structure.
model = keras.Sequential([
    # Input layer -> is a 28x28 image ("Flatten" means that the 28x28 image is flattened into a single 784x1 input layer).
    keras.layers.Flatten(input_shape=(28,28)),
    
    # Hidden layer -> is 128 deep. "Relu" returns the value, or 0 (Works good enough. Much faster).
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    
    # Hidden layer -> is 128 deep. "Relu" returns the value, or 0 (Works good enough. Much faster).
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    
    # Output layer -> is 0-10 (Depending on the piece of clothing it is). Return maximum.
    keras.layers.Dense(units=10, activation=tf.nn.softmax)])

# Compile model
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model using our training data
model.fit(train_images, train_labels, epochs=10)

# Test model using our testing data
test_loss = model.evaluate(test_images, test_labels)
clothing_option = random.randint(0, 9)
plt.imshow(test_images[clothing_option], cmap='gray', vmin=0, vmax=255)
plt.show()

while True:
    # Make predictions
    predictions = model.predict(test_images)

    # Save prediction
    result = list(predictions[clothing_option]).index(max(predictions[clothing_option]))
    # print(list(predictions[clothing_option]).index(max(predictions[clothing_option])))

    # Print correct answer
    print(f"Output: {result}")
    
    #Asking the user if they would like to continue
    answer = input("Would you like to continue? (Y/N): ")
    
    if answer.upper() == 'Y':
        print("Let's keep predicting!")
        test_loss = model.evaluate(test_images, test_labels)
        clothing_option = random.randint(0, 9)
        plt.imshow(test_images[clothing_option], cmap='gray', vmin=0, vmax=255)
        plt.show()
    else:
        print("See ya!\n")
        break
    
print("Code completed!")
