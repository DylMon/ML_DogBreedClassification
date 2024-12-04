import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Paths to the train and test datasets
TRAIN_DIR = "train"
TEST_DIR = "test"

# Hyperparameters
IMAGE_SIZE = (224, 224)  # Image dimensions for resizing
BATCH_SIZE = 32
EPOCHS = 20

def load_datasets():
    """Load the training and testing datasets."""
    # Load training dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # Load testing dataset
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # Capture class names from the original datasets
    class_names = train_ds.class_names

    # Optimize performance by adding prefetch
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, test_ds, class_names


def create_model(num_classes):
    """Define the CNN model."""
    model = models.Sequential([
        # Input layer (resized images)
        layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),  # Normalize pixel values

        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten and fully connected layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Add dropout for regularization
        layers.Dense(num_classes, activation='softmax')  # Output layer
    ])
    return model

def plot_metrics(history):
    """Plot training metrics: Loss and Accuracy."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

if __name__ == "__main__":
    # Load datasets
    train_ds, test_ds, class_names = load_datasets()

    # Get number of classes
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    # Create and compile the model
    model = create_model(num_classes)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS
    )

    # Save the trained model
    model.save("dog_breed_classifier.h5")
    print("Model saved as dog_breed_classifier.h5")

    # Plot metrics
    plot_metrics(history)

