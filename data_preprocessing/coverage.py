import cv2
import numpy as np
import os
import csv

# Create directory for saving segmented images
segmentation_dir = 'segmented_images_b'
os.makedirs(segmentation_dir, exist_ok=True)

# Create CSV file and write header row
with open('canopy_coverage.csv', 'w', newline='') as csvfile:
    fieldnames = ['folder', 'a_coverage', 'c_coverage', 'e_coverage']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# Iterate through 60 folders
for folder_num in range(1, 61):
    folder_path = f'../resource/rgb/{folder_num}'

    # Create directory for saving segmented images of the corresponding folder
    folder_segment_dir = os.path.join(segmentation_dir, str(folder_num))
    os.makedirs(folder_segment_dir, exist_ok=True)

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist, skipping")
        continue

    # Initialize result dictionary
    results = {'folder': folder_num, 'a_coverage': None, 'c_coverage': None, 'e_coverage': None}

    # Process a.jpg, c.jpg, e.jpg
    for img_name in ['a.jpg', 'c.jpg', 'e.jpg']:
        img_path = os.path.join(folder_path, img_name)

        # Check if the image file exists
        if not os.path.exists(img_path):
            print(f"Image {img_path} does not exist, the corresponding value will be set to None")
            continue

        # Read image
        try:
            # Read the original image
            original = cv2.imread(img_path)

            # Use 2g-r-b method to separate soil and background
            # Convert to float for calculation
            demo1 = np.array(original, dtype=np.float32) / 255.0
            (b, g, r) = cv2.split(demo1)
            gray = 2.4 * g - b - r

            # Get maximum and minimum values
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

            # Convert to uint8 type for Otsu's binarization
            gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
            (thresh, th) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)

            # Create mask, convert binary image to 3 channels for operation with original image
            mask = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR) / 255.0

            # Create pure black background
            black_bg = np.zeros_like(original)

            # Retain original color for vegetation area, set background area to pure black
            # Formula: Result = Vegetation area (original image) + Background area (black)
            result = np.uint8(original * mask + black_bg * (1 - mask))

            # Save the processed image
            segmented_img_name = f"{os.path.splitext(img_name)[0]}_segmented.jpg"
            segmented_img_path = os.path.join(folder_segment_dir, segmented_img_name)
            cv2.imwrite(segmented_img_path, result)
            print(f"Segmented image saved to: {segmented_img_path}")

            # Calculate the ratio of green vegetation to all pixels
            coverage = round(sum(sum(th / 255)) / (len(th[0]) * len(th)) * 100, 2)
            print(f'Canopy coverage of {img_name} in folder {folder_num} is {coverage}%')

            # Store the result
            if img_name == 'a.jpg':
                results['a_coverage'] = coverage
            elif img_name == 'c.jpg':
                results['c_coverage'] = coverage
            elif img_name == 'e.jpg':
                results['e_coverage'] = coverage

        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            continue

    # Write results to CSV file
    with open('canopy_coverage.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(results)