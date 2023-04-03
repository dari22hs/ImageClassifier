"""
TensorFlow Image classifier version 2 using Fashion MNIST dataset from Keras:
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
import random
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = 28
NUM_CLASSES = 10
EPOCHS = 10
BATCH_SIZE = 32

# Load a pre-defined dataset (70k images of 28x28).
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# Define neural net structure.
model = keras.Sequential([
    # Input layer -> is a 28x28 image ("Flatten" means that the 28x28 image is flattened into a single 784x1 input layer).
    keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)),
    
    # Hidden layer -> is 128 deep. "Relu" returns the value, or 0 (Works good enough. Much faster).
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    
    # Hidden layer -> is 128 deep. "Relu" returns the value, or 0 (Works good enough. Much faster).
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    
    # Output layer -> is 0-10 (Depending on the piece of clothing it is). Return maximum.
    keras.layers.Dense(units=NUM_CLASSES, activation=tf.nn.softmax)])

# Compile model
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model using our training data
model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Test model using our testing data
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Create a dictionary of class labels and corresponding names
results = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Show a random test image and its predicted result
def show_prediction():
    """
    Function that displays a random image from the dataset and prints the true and predicted labels.
    """
    random_index = random.randint(0, len(test_images) -1)
    image = test_images[random_index]
    true_label = test_labels[random_index]
    prediction = model.predict(tf.expand_dims(image, axis=0))[0]
    predicted_label = tf.argmax(prediction).numpy()
    plt.imshow(image, cmap='gray')
    plt.title(f"True Label: {results[true_label]} | Predicted Label: {results[predicted_label]}")
    plt.show()

# Loop for making predictions
while True:
    show_prediction()       
    answer = input("Would you like to continue? (Y/N): ")
    if answer.upper() == 'Y':
        print("Let's keep predicting!")
    else:
        print("See ya!\n")
        break
    
print("Code completed!")
