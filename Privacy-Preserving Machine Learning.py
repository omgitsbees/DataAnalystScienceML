import tensorflow as tf
import tensorflow_federated as tff

# Define a simple model
def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# Define a function to create the federated learning process
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=sample_batch,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Load a sample dataset (e.g., MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a federated dataset (e.g., split the data by users)
federated_train_data = [tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(20)]

# Build the Federated Averaging process
iterative_process = tff.learning.build_federated_averaging_process(model_fn)

# Initialize the process
state = iterative_process.initialize()

# Run the federated learning process for a few rounds
for round_num in range(1, 11):
    state, metrics = iterative_process.next(state, federated_train_data)
    print(f'Round {round_num}, metrics={metrics}')

# Test the model
model = create_keras_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.set_weights(state.model.weights)
model.evaluate(x_test, y_test)
