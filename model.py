import tensorflow as tf

def create_cnn_model(input_shape):
    """
    Creates, compiles, and returns a CNN model for land cover classification.

    Args:
        input_shape (tuple): The shape of the input data (e.g., (1, 1, n_bands)).
        num_classes (int): The number of output classes.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=1, padding='valid', activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(48, kernel_size=(1, 1), activation='relu', padding='valid'))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, kernel_size=(1, 1), activation='relu', padding='valid'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(80, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    print("Model created and compiled successfully.")
    model.summary()
    return model
