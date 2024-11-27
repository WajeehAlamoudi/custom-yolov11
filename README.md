
# YOLOv11 Custom Object Detection

This repository provides instructions on how to train the YOLOv11 (You Only Look Once version 11) object detection model on your custom dataset. YOLOv11 is a fast and efficient model for real-time object detection and can be fine-tuned to detect specific objects or classes based on your needs.

# Prerequisites
- Python 3.8+
- Downloading the packages.
- GPU highly recommended.
- if GPU not available you can use colab or any computaional power provider.
# Getting Started
- orgnize your project as followed:
```bash
custom_yolov11/
│
├── data/
│   ├── train/
│   │   ├── images/              # Training images
│   │   └── labels/              # Training label files (.txt for each image)
│   │   └── labels.cache         # Cache for labels in the training set (optional)
│   ├── valid/
│   │   ├── images/              # Validation images
│   │   └── labels/              # Validation label files (.txt for each image)
│   │   └── labels.cache         # Cache for labels in the validation set (optional)
│   ├── test/
│   │   ├── images/              # Test images (no labels.cache file)
│   │   └── labels/              # Test label files (.txt for each image) - optional, if labels are available
│   └── data.yaml                # Data configuration file
│
└── config/
    └── data.yaml             # YOLOv11 configuration for training


```
# Annotate the images for training
- using any annotate program or use:
https://roboflow.com/

# Configuration File
```yaml
train: /path/to/train/images
valid: /path/to/valid/images

nc: 2  # Number of classes
names:
  0: "class1"
  1: "class2"
  # or use ['class1','class2]

  # for further hyperparameter usage
  '''
    batch_size: 16
    epochs: 50
    img_size: 416
    learning_rate: 0.001
    momentum: 0.937
    weight_decay: 0.0005
'''
```
# Note
- you can get ready data or make your custom data.
Below code that retun images from video:

```
import cv2
import os


# Function to extract frames from a video
def video_to_images(video_path, output_folder, frames_per_second=3):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the interval to capture frames
    interval = int(fps / frames_per_second)

    # Initialize frame count
    count = 0
    saved_frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            break

        # Save every 'interval' frame
        if count % interval == 0:
            # Construct the filename for the output image
            image_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")

            # Save the frame as an image
            cv2.imwrite(image_filename, frame)
            print(f"Saved {image_filename}")
            saved_frame_count += 1

        count += 1

    # Release the video capture object
    cap.release()
    print("Finished extracting frames.")


# Example usage
video_path = "path/to/video"
output_folder = 'video_images'  # Output folder to save images
video_to_images(video_path, output_folder)

```
