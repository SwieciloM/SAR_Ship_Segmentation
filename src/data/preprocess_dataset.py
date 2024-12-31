import os
import shutil
import re
import itertools
from sklearn.model_selection import train_test_split


# Directory containing the images and masks
image_dir = r'../../data/interim/JPEG_images'
mask_dir = r'../../data/interim/PNG_masks'

# Directories for the split datasets
train_dir = r'../../data/processed/train'
val_dir = r'../../data/processed/val'
test_dir = r'../../data/processed/test'

# Split ratios
train_size = 0.7
val_size = 0.15
# The test size will be computed as (1 - train_size - val_size)

# Seed for reproducibility
random_seed = 42

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'images'))
        os.makedirs(os.path.join(path, 'masks'))

# Create directories
create_dir(train_dir)
create_dir(val_dir)
create_dir(test_dir)

# Get all file names
images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
masks = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]

# Sort the lists to ensure alignment
images.sort(key=lambda name: tuple(map(int, re.findall(r'\d+', name))))
masks.sort(key=lambda name: tuple(map(int, re.findall(r'\d+', name))))

# Splitting the data
train_images, test_images, train_masks, test_masks = train_test_split(
    images, masks, test_size=1-train_size-val_size, random_state=random_seed)
train_images, val_images, train_masks, val_masks = train_test_split(
    train_images, train_masks, test_size=val_size/(train_size+val_size), random_state=random_seed)

def copy_files(src_files, dst_folder):
    for file in src_files:
        shutil.copy(file, os.path.join(dst_folder, os.path.basename(file).replace('_mask', '')))

# Copy files to their respective directories
copy_files(train_images, os.path.join(train_dir, 'images'))
copy_files(train_masks, os.path.join(train_dir, 'masks'))
copy_files(val_images, os.path.join(val_dir, 'images'))
copy_files(val_masks, os.path.join(val_dir, 'masks'))
copy_files(test_images, os.path.join(test_dir, 'images'))
copy_files(test_masks, os.path.join(test_dir, 'masks'))

def summarize_data(directory):
    images = os.listdir(os.path.join(directory, 'images'))
    masks = os.listdir(os.path.join(directory, 'masks'))
    print(f"Total images in {os.path.basename(directory)}: {len(images)}")
    print(f"Total masks in {os.path.basename(directory)}: {len(masks)}")

# Print summaries
print("Summary of the dataset split")
summarize_data(train_dir)
summarize_data(val_dir)
summarize_data(test_dir)
