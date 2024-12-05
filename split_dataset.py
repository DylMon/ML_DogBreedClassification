import os
import shutil
from sklearn.model_selection import train_test_split

# Paths for the original dataset and output folders
BASE_DIR = "images"
TRAIN_DIR = "train"
TEST_DIR = "test"

# Percentage of data to reserve for testing
TEST_SPLIT = 0.2


def create_dir(path):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def split_dataset():
    """Splits the dataset into train and test sets."""
    # Create train and test directories
    create_dir(TRAIN_DIR)
    create_dir(TEST_DIR)

    # Iterate through each breed folder in the images directory
    for breed_folder in os.listdir(BASE_DIR):
        breed_path = os.path.join(BASE_DIR, breed_folder)

        # Skip if not a directory
        if not os.path.isdir(breed_path):
            continue

        # Get all images in the current breed folder
        images = [os.path.join(breed_path, img) for img in os.listdir(breed_path) if img.endswith(('.jpg', '.png'))]

        # Skip folders with fewer than 2 images
        if len(images) < 2:
            print(f"Skipping {breed_folder} (not enough images)")
            continue

        # Split images into train and test sets
        train_images, test_images = train_test_split(images, test_size=TEST_SPLIT, random_state=42)

        # Create corresponding breed folders in train/test directories
        train_breed_dir = os.path.join(TRAIN_DIR, breed_folder)
        test_breed_dir = os.path.join(TEST_DIR, breed_folder)
        create_dir(train_breed_dir)
        create_dir(test_breed_dir)

        # Copy images to the respective directories
        for img in train_images:
            shutil.copy(img, train_breed_dir)
        for img in test_images:
            shutil.copy(img, test_breed_dir)

        print(f"Processed {breed_folder}: {len(train_images)} train, {len(test_images)} test")


if __name__ == "__main__":
    split_dataset()
    print("Dataset splitting complete!")
