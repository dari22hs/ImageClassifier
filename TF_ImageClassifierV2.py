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

# Load a pre-defined dataset (70k images of 28x28). / Cargar un conjunto de datos predefinido (70 mil imágenes de 28x28)
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# Define neural net structure.
model = keras.Sequential([
    # Input layer -> is a 28x28 image ("Flatten" means that the 28x28 image is flattened into a single 784x1 input layer). / Capa de entrada -> es una imagen de 28x28 ("Flatten" significa que la imagen de 28x28 se aplanó en una sola capa de entrada de 784x1).
    keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)),
    
    # Hidden layer -> is 128 deep. "Relu" returns the value, or 0 (Works good enough. Much faster). / Capa oculta -> tiene una profundidad de 128. "Relu" devuelve el valor o 0 (funciona lo suficientemente bien y es mucho más rápido).
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    
    # Hidden layer -> is 128 deep. "Relu" returns the value, or 0 (Works good enough. Much faster). / Capa oculta -> tiene una profundidad de 128. "Relu" devuelve el valor o 0 (funciona lo suficientemente bien y es mucho más rápido).
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    
    # Output layer -> is 0-10 (Depending on the piece of clothing it is). Return maximum. / Capa de salida -> es 0-10 (dependiendo de la prenda de vestir). Devuelve el máximo.
    keras.layers.Dense(units=NUM_CLASSES, activation=tf.nn.softmax)])

# Compile model / Compilar el modelo
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model using our training data / Entrenar el modelo usando nuestros datos de entrenamiento.
model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Test model using our testing data / Probar el modelo usando nuestros datos de prueba.
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Create a dictionary of class labels and corresponding names / Crear un diccionario de etiquetas de clases y los nombres correspondientes.
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


# Show a random test image and its predicted result / Mostrar una imagen de prueba aleatoria y su resultado predicho.
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

# Loop for making predictions / Bucle para hacer predicciones
while True:
    show_prediction()       
    answer = input("Would you like to continue? (Y/N): ")
    if answer.upper() == 'Y':
        print("Let's keep predicting!")
    else:
        print("See ya!\n")
        break
    
print("Code completed!")
