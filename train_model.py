import os
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflowjs as tfjs  # Import TensorFlow.js converter
import matplotlib.pyplot as plt

# Paths to the train and test datasets
TRAIN_DIR = "train"
TEST_DIR = "test"

# Hyperparameters
IMAGE_SIZE = (224, 224)  # Image dimensions for resizing
BATCH_SIZE = 32
EPOCHS = 20
INITIAL_EPOCHS = 10  # Epochs before fine-tuning
FINE_TUNE_EPOCHS = 10  # Additional epochs for fine-tuning

def load_datasets():
    """Load the training and testing datasets with data augmentation."""
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

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

    # Capture class names BEFORE data augmentation
    class_names = train_ds.class_names

    # Apply data augmentation
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

    # Optimize performance by adding prefetch
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, test_ds, class_names



def create_model(num_classes):
    """Define the model with transfer learning."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,  # Exclude the final classification layer
        weights='imagenet'  # Use pretrained weights
    )
    base_model.trainable = False  # Freeze the base model layers

    # Add classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Output layer
    ])
    return model


def fine_tune_model(model, base_model):
    """Unfreeze some layers of the base model for fine-tuning."""
    base_model.trainable = True
    # Fine-tune from the last 30 layers of the base model
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Recompile with a lower learning rate for fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


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

    # Train the model (initial training phase)
    print("Starting initial training...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=INITIAL_EPOCHS
    )

    # Fine-tune the model
    print("Fine-tuning the model...")
    fine_tune_model(model, model.layers[0])  # Pass the base model
    fine_tune_history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=FINE_TUNE_EPOCHS,
        initial_epoch=INITIAL_EPOCHS
    )

    # Combine histories for plotting
    history.history['accuracy'].extend(fine_tune_history.history['accuracy'])
    history.history['val_accuracy'].extend(fine_tune_history.history['val_accuracy'])
    history.history['loss'].extend(fine_tune_history.history['loss'])
    history.history['val_loss'].extend(fine_tune_history.history['val_loss'])

    # Save the trained model in HDF5 format (optional backup)
    model.save("dog_breed_classifier.h5")
    print("Model saved as dog_breed_classifier.h5")

    # Save the model in TensorFlow.js format for the webpage
    tfjs_output_dir = "public/tfjs_model"
    tfjs.converters.save_keras_model(model, tfjs_output_dir)
    print(f"Model saved in TensorFlow.js format at: {tfjs_output_dir}")

    # Plot metrics
    plot_metrics(history)
