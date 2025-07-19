import os
import time
import json
from PIL import Image
import numpy as np

# --- Configuration ---
INPUT_FOLDER = "input_images"
PROCESSED_FOLDER = "processed_images"
LOG_FILE = "processing_log.txt" # Optional: for logging events

# Model's expected input dimensions and type (from previous metadata check)
# For the 'cxr-pneumonia' model, it expects [-1, 1, 224, 224] FP32
TARGET_HEIGHT = 224
TARGET_WIDTH = 224
TARGET_CHANNELS = 1 # Grayscale
TARGET_DATATYPE = "FP32"

# --- Helper Functions ---

def ensure_directories_exist():
    """Ensures that the input and processed directories exist."""
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    print(f"Ensured '{INPUT_FOLDER}' and '{PROCESSED_FOLDER}' directories exist.")

def clean_folder(folder_path):
    """Deletes all files within a specified folder."""
    log_message(f"Cleaning contents of folder: {folder_path}")
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                log_message(f"Deleted: {file_path}")
        log_message(f"Finished cleaning folder: {folder_path}")
    except Exception as e:
        log_message(f"Error cleaning folder {folder_path}: {e}")


def log_message(message):
    """Logs a message to the console and an optional log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def process_image_file(image_path):
    """
    Loads, preprocesses, and saves an image for model inference.
    Args:
        image_path (str): The full path to the input JPEG image.
    Returns:
        tuple: (processed_data_path, processed_shape, processed_datatype) if successful,
               None otherwise.
    """
    try:
        log_message(f"Processing image: {image_path}")
        img = Image.open(image_path)
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT)) # Pillow expects (width, height)

        # Convert to grayscale ('L' for Luminance) as the model expects 1 channel
        if img.mode != 'L':
            img = img.convert('L')

        # Convert image to a numpy array
        img_array = np.asarray(img) # Shape will be (HEIGHT, WIDTH) for grayscale

        # Normalize pixel values if the model expects FP32 in a certain range (e.g., 0-1)
        if TARGET_DATATYPE == "FP32":
            img_array = img_array / 255.0 # Normalize to 0-1 range for FP32

        # Add batch dimension and channel dimension, then transpose to [BATCH, CHANNEL, HEIGHT, WIDTH]
        # Current img_array shape for grayscale is (HEIGHT, WIDTH)
        # Add channel dimension (1) at axis 0: (1, HEIGHT, WIDTH)
        # Add batch dimension (1) at axis 0: (1, 1, HEIGHT, WIDTH)
        processed_img_array = img_array[np.newaxis, np.newaxis, :, :]

        # Construct the output filename (e.g., 'image_name.jpeg' -> 'image_name.npy')
        original_filename = os.path.basename(image_path)
        base_filename, _ = os.path.splitext(original_filename)
        processed_filename = f"{base_filename}.npy"
        processed_file_path = os.path.join(PROCESSED_FOLDER, processed_filename)

        # Save the processed NumPy array
        np.save(processed_file_path, processed_img_array)
        log_message(f"Successfully processed and saved to: {processed_file_path}")

        # Return relevant info for the next step (inference)
        return processed_file_path, list(processed_img_array.shape), TARGET_DATATYPE

    except FileNotFoundError:
        log_message(f"Error: Input image file not found at {image_path}")
        return None
    except Exception as e:
        log_message(f"Error processing image {image_path}: {e}")
        return None

def cleanup_original_file(file_path):
    """Deletes the original file after successful processing."""
    try:
        os.remove(file_path)
        log_message(f"Cleaned up original file: {file_path}")
    except Exception as e:
        log_message(f"Error cleaning up file {file_path}: {e}")

# --- Main Job Logic ---

if __name__ == "__main__":
    ensure_directories_exist()
    clean_folder(PROCESSED_FOLDER) # Clean the output folder before starting
    log_message("Image processing job started. Looking for JPEG files...")

    try:
        # Get list of JPEG files in the input folder
        jpeg_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpeg', '.jpg'))]

        if jpeg_files:
            log_message(f"Found {len(jpeg_files)} JPEG file(s) to process.")
            for filename in jpeg_files:
                input_file_path = os.path.join(INPUT_FOLDER, filename)
                result = process_image_file(input_file_path)

                if result:
                    # If processing was successful, delete the original file
                    cleanup_original_file(input_file_path)
                    # In a real application, you might also want to store `result`
                    # in a queue or a database for the next step (inference application)
                    # For this example, we just print it.
                    log_message(f"Processed file details for next step: Path={result[0]}, Shape={result[1]}, Datatype={result[2]}")
                else:
                    log_message(f"Failed to process {input_file_path}. Skipping cleanup.")
        else:
            log_message(f"No new JPEG files found in '{INPUT_FOLDER}'. Exiting.")

    except Exception as e:
        log_message(f"An unexpected error occurred: {e}")

    log_message("Image processing job finished.")
