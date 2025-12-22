import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

def create_cnn_model(input_shape):
    """
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



def create_ann_model(input_shape):
    """
    Args:
        input_shape: Tuple like (H, W, C) or (features,).
        num_classes: Number of target classes.
        hidden_units: Tuple of Dense layer sizes.
        dropout: Dropout rate applied after hidden layers.
        learning_rate: Optimizer learning rate.

    Returns:
        tf.keras.Model
    """
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=input_shape))

    model.add(tf.keras.layers.Dense(32, activation='relu'))

    model.add(tf.keras.layers.Dense(7, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
        )

    print("ANN model created and compiled successfully.")
    model.summary()
    return model



def create_rf_model():
    """
    Args:
        n_estimators: Number of trees.
        max_depth: Max depth of each tree.
        random_state: RNG seed.
        n_jobs: Parallel jobs (-1 uses all cores).
        class_weight: Dict or 'balanced' for imbalanced data.

    Returns:
        sklearn.ensemble.RandomForestClassifier
    """
    rf = RandomForestClassifier(n_estimators=100,
                                min_samples_leaf= 3,
                                random_state=42
                                )

    return rf