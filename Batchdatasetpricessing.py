#!/usr/bin/env python
# coding: utf-8

# In[38]:


import cv2
import os
import time
import numpy as np
import configparser
import argparse
import json


output_dir = ('H:\dataset\Cancerous+cell+smears+2023\Batch')
input_dir = ('H:\dataset\Cancerous+cell+smears+2023')

def process_image(image_path, setup_file=None):
    """
    Process a single image.
    """
    # Load image
    image = cv2.imread(image_path)

    # Process image using the setup file, if supplied
    if setup_file is not None:
        with open(setup_file, 'r') as f:
            setup = json.load(f)
        # Apply setup parameters to image processing
        # ...

    # Return processed image
    return processed_image

def process_batch(input_dir, output_dir, setup_file=None):
    """
    Process all images in a directory and save the processed images to a separate directory.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all images in the input directory
    for filename in os.listdir(input_dir):
        # Only process image files
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Get the full path to the image
            image_path = os.path.join(input_dir, filename)

            # Process the image
            processed_image = process_image(image_path, setup_file)

            # Save the processed image to the output directory with the same filename
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, processed_image)

if __name__ == '__main__':
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Process images in a batch setting.')
    parser.add_argument('input_dir', type=str, help='Input directory containing images to process.')
    parser.add_argument('output_dir', type=str, help='Output directory to save processed images.')
    parser.add_argument('--setup_file', type=str, default=None, help='Setup initialization file for image processing.')

    # Parse command-line arguments
    args = parser.parse_args()

    # Process the batch of images
    process_batch(args.input_dir, args.output_dir, args.setup_file)


# In[17]:


# specify the directory containing the images
image_dir = ('H:\dataset\Cancerous+cell+smears+2023')

# specify the color to extract (red, green, or blue)
color = "red"

# iterate through all image files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".BMP"):
        # read the image file
        img = cv2.imread(os.path.join(image_dir, filename))

        # extract the specified color channel
        if color == "red":
            img = img[:, :, 2]
        elif color == "green":
            img = img[:, :, 1]
        elif color == "blue":
            img = img[:, :, 0]

        # save the converted image
        cv2.imwrite(os.path.join(image_dir, f"{filename}_{color}.jpg"), img)


# In[16]:


# specify the directory containing the images
image_dir = ('H:\dataset\Cancerous+cell+smears+2023')

# iterate through all image files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".BMP"):
        # read the image file
        img = cv2.imread(os.path.join(image_dir, filename))

        # calculate the histogram
        hist = cv2.calcHist([img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

        # print the histogram for the image
        print(f"Histogram for {filename}: {hist}")


# In[24]:


# specify the directory containing the images
image_dir = ('H:\dataset\Cancerous+cell+smears+2023')

# specify the classes of images
classes = ["class1", "class2", "class3"]

# initialize dictionaries to store the histograms and counts for each class
histograms = {c: np.zeros((256, 256, 256)) for c in classes}
counts = {c: 0 for c in classes}

# iterate through all image files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".BMP"):
        # determine the class of the image
        image_class = None
        for c in classes:
            if c in filename:
                image_class = c
                break

        # read the image file
        img = cv2.imread(os.path.join(image_dir, filename))

        # calculate the histogram
        hist = cv2.calcHist([img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

        # add the histogram to the running total for the class
        if image_class is not None:
            histograms[image_class] += hist
            counts[image_class] += 1

# calculate the average histograms for each class
averages = {c: histograms[c] / counts[c] for c in classes}

# print the average histograms for each class
for c in classes:
    print(f"Average histogram for {c}: {averages[c]}")



# In[29]:


# specify the directory containing the images
image_dir = ('H:\dataset\Cancerous+cell+smears+2023')

# iterate through all image files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".BMP"):
        # read the image file
        img = cv2.imread(os.path.join(image_dir, filename))

        # convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # perform histogram equalization
        equalized = cv2.equalizeHist(gray)
   
        # save the equalized image
        cv2.imwrite(os.path.join(image_dir, f"{filename}_equalized.jpg"), equalized)


# In[30]:


# define a function to add Salt and Pepper noise to an image
def add_salt_and_pepper_noise(image, strength):
    # generate a random mask of Salt and Pepper noise
    mask = np.random.choice([0, 1, 2], size=image.shape[:2], p=[1 - strength, strength/2, strength/2])
    
    # apply the mask to the image
    noise = np.zeros_like(image)
    noise[mask == 1] = [255, 255, 255] # Salt noise
    noise[mask == 2] = [0, 0, 0] # Pepper noise
    noisy_image = cv2.addWeighted(image, 1 - strength, noise, strength, 0)
    
    return noisy_image

# specify the directory containing the images
image_dir = ('H:\dataset\Cancerous+cell+smears+2023')

# specify the strength of the Salt and Pepper noise
strength = 0.1

# iterate through all image files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".BMP"):
        # read the image file
        img = cv2.imread(os.path.join(image_dir, filename))

        # add Salt and Pepper noise to the image
        noisy_image = add_salt_and_pepper_noise(img, strength)

        # save the noisy image to a new file
        noisy_filename = os.path.splitext(filename)[0] + "_noisy.jpg"
        cv2.imwrite(os.path.join(image_dir, noisy_filename), noisy_image)


# In[31]:


# define a function to add Gaussian noise to an image
def add_gaussian_noise(image, mean, std_dev):
    # generate Gaussian noise with mean and standard deviation parameters
    noise = np.zeros_like(image)
    cv2.randn(noise, mean, std_dev)

    # add the noise to the image
    noisy_image = cv2.add(image, noise)

    return noisy_image

# specify the mean and standard deviation of the Gaussian noise
mean = 0
std_dev = 50

# iterate through all image files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".BMP"):
        # read the image file
        img = cv2.imread(os.path.join(image_dir, filename))

        # add Gaussian noise to the image
        noisy_image = add_gaussian_noise(img, mean, std_dev)

        # save the noisy image to a new file
        noisy_filename = os.path.splitext(filename)[0] + "_noisy.jpg"
        cv2.imwrite(os.path.join(image_dir, noisy_filename), noisy_image)


# In[ ]:


# define a function to apply a linear filter to an image
def apply_linear_filter(image, kernel):
    # apply the filter using OpenCV's filter2D function
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image

# define the kernel/mask for the filter
mask_size = 3 # size of the square kernel
pixel_weights = [1, 2, 1, 2, 4, 2, 1, 2, 1] # weights for each pixel in the kernel

# create the kernel from the pixel weights
kernel = np.array(pixel_weights).reshape((mask_size, mask_size))

# iterate through all image files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".BMP"):
        # read the image file
        img = cv2.imread(os.path.join(image_dir, filename))

        # apply the linear filter to the image
        filtered_image = apply_linear_filter(img, kernel)

        # save the filtered image to a new file
        filtered_filename = os.path.splitext(filename)[0] + "_filtered.jpg"
        cv2.imwrite(os.path.join(image_dir, filtered_filename), filtered_image)


# In[ ]:


# define a function to apply a median filter to an image
def apply_median_filter(image, kernel_size):
    # apply the filter using OpenCV's medianBlur function
    filtered_image = cv2.medianBlur(image, kernel_size)

    return filtered_image

# define the kernel/mask size for the filter
kernel_size = 3 # size of the square kernel

# iterate through all image files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".BMP"):
        # read the image file
        img = cv2.imread(os.path.join(image_dir, filename))

        # apply the median filter to the image
        filtered_image = apply_median_filter(img, kernel_size)

        # save the filtered image to a new file
        filtered_filename = os.path.splitext(filename)[0] + "_filtered.jpg"
        cv2.imwrite(os.path.join(image_dir, filtered_filename), filtered_image)


# In[ ]:


# define a function to process the images
def process_images(image_dir):
    start_time = time.time()

    # read in the images
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".BMP"):
            img = cv2.imread(os.path.join(image_dir, filename))
            # process the image here

    end_time = time.time()
    print("Processing time for images in directory {}: {:.2f} seconds".format(image_dir, end_time - start_time))

# process the images
process_images(image_dir)


# In[ ]:


# define a function to process the images
def process_images(image_dir):
    total_time = 0
    num_images = 0

    # read in the images
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".BMP"):
            img = cv2.imread(os.path.join(image_dir, filename))

            start_time = time.time()
            # process the image here
            end_time = time.time()

            total_time += end_time - start_time
            num_images += 1

    avg_time_per_image = total_time / num_images
    print("Average processing time per image for directory {}: {:.2f} seconds".format(image_dir, avg_time_per_image))

# process the images
process_images(image_dir)

