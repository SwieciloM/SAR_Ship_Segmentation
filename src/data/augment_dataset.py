import os
import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import label, find_objects, binary_dilation
from PIL import Image, ImageDraw
from patchify import patchify, unpatchify

def create_patches(image, patch_size, step):
    patches = patchify(np.array(image), patch_size, step=step)
    return patches

def reconstruct_image(patches, merged_image_size):
    # Reassemble the image
    reconstructed_img = unpatchify(patches, merged_image_size)
    reconstructed_img = Image.fromarray(reconstructed_img.astype(np.uint8))
    return reconstructed_img

def detect_ships(image):
    """
    Detect ships (that are not touching the edge and non-overlapping) bounding boxes 
    """
    image_array = np.array(image)

    # Convert to binary if it's not
    if len(image_array.shape) > 2:
        image_array = image_array[:, :, 0] > 128

    structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 8-connectivity
    labeled_array, num_features = label(image_array, structure=structure)

    shapes = find_objects(labeled_array)
    height, width = image_array.shape
    valid_shapes = []

    def is_overlapping(box1, box2):
        # Returns True if two boxes overlap
        return not (box1[2] <= box2[0] or box1[0] >= box2[2] or
                    box1[3] <= box2[1] or box1[1] >= box2[3])

    for shape in shapes:
        start_x, start_y = shape[1].start, shape[0].start
        end_x, end_y = shape[1].stop, shape[0].stop

        # Check if the shape touches the border
        if start_y > 0 and start_x > 0 and end_y < height and end_x < width:
            current_box = (start_x, start_y, end_x, end_y)
            # Check for overlap with already accepted boxes
            if not any(is_overlapping(current_box, existing_box) for existing_box in valid_shapes):
                valid_shapes.append(current_box)

    return valid_shapes, len(shapes)

def draw_bounding_boxes(image, boxes):
    """
    Draw rectangles on the image based on provided bounding boxes.
    """
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)  # Modify the image directly

    for box in boxes:
        draw.rectangle(box, outline='red')

    return image

def extract_parts(image, valid_shapes):
    """
    Extracts parts of the image corresponding to the bounding boxes in valid_shapes.
    
    :param image: A PIL.Image object of the original image.
    :param valid_shapes: A list of tuples (start_x, start_y, end_x, end_y) representing bounding boxes.
    :return: A list of PIL.Image objects, each one cropped to a bounding box from valid_shapes.
    """
    cropped_images = []
    
    for (start_x, start_y, end_x, end_y) in valid_shapes:
        # Define the bounding box with the correct order of coordinates
        box = (start_x, start_y, end_x, end_y)
        # Crop the image and append to the list of cropped images
        cropped_image = image.crop(box)
        cropped_images.append(cropped_image)
    
    return cropped_images

def average_brightness(image):
    """
    Calculate the average brightness of a grayscale image.

    :param image: A PIL.Image object.
    :return: Average brightness of the image.
    """
     # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Calculate the average brightness
    average = np.mean(image_array)
    
    return average

def distribute_ship_copies(image_patches, mask_patches, image_ships, mask_ships, copy_factor):
    new_ships_num = copy_factor*len(image_ships) - len(image_ships)
    patches_num = len(image_patches)
    final_new_ships_num = 0
    mod_image_patches = []
    mod_mask_patches = []

    if new_ships_num >= patches_num:
        patches_to_edit = [1] * patches_num
    else:
        patches_to_edit = [1] * new_ships_num + [0] * (patches_num - new_ships_num)
        random.shuffle(patches_to_edit)

    ships = list(zip(image_ships, mask_ships))

    for i, should_be_edit in enumerate(patches_to_edit):
        img_patch = Image.fromarray(image_patches[i])
        mask_patch = Image.fromarray(mask_patches[i])

        if should_be_edit:
            img_ship, mask_ship = random.choice(ships)
            angle = random.choice([0, 90, 180, 270])
            img_ship_rot = img_ship.rotate(angle, expand=True)
            mask_ship_rot = mask_ship.rotate(angle, expand=True)
            
            # Random position within the patch
            max_x = img_patch.width - img_ship_rot.width
            max_y = img_patch.height - img_ship_rot.height
            if max_x > 0 and max_y > 0:
                random_x = random.randint(0, max_x)
                random_y = random.randint(0, max_y)

                array_mask_ship_rot = (np.array(mask_ship_rot) == np.max(mask_ship_rot)).astype(int)
                dilated_mask_ship_rot = binary_dilation(array_mask_ship_rot, structure=np.ones((5, 5)), iterations=1)
                dilated_mask_ship_rot = Image.fromarray((dilated_mask_ship_rot * 255).astype(np.uint8))
                
                # Paste the rotated part onto the patch
                img_patch.paste(img_ship_rot, (random_x, random_y), dilated_mask_ship_rot)
                mask_patch.paste(mask_ship_rot, (random_x, random_y), mask_ship_rot)

                final_new_ships_num += 1

        mod_image_patches.append(img_patch)
        mod_mask_patches.append(mask_patch)

    return mod_image_patches, mod_mask_patches, final_new_ships_num

def apply_augmentation(img_patches, mask_patches, image_ships, mask_ships, copy_factor=2):
    img_patches_augmented, mask_patches_augmented = img_patches, mask_patches
    num_row, num_col, *_ = np.shape(img_patches)
    img_patches_editable = []
    mask_patches_editable = []
    editable_patches_indexes = []
    for r in range(num_row):
        for c in range(num_col):
            img_patch = img_patches[r, c]
            mask_patch = mask_patches[r, c]
            avg_bright = average_brightness(img_patch)
            if not (avg_bright > 25 or np.any(mask_patch == 255)):
                img_patches_editable.append(img_patch)
                mask_patches_editable.append(mask_patch)
                editable_patches_indexes.append((r, c))
                # img_patches_augmented[r, c] = img_patch*2
                
    img_patches_mod, mask_patches_mod, new_ships_num = distribute_ship_copies(img_patches_editable, mask_patches_editable, image_ships, mask_ships, copy_factor)

    for idx, img_patche_mod, mask_patche_mod in zip(editable_patches_indexes, img_patches_mod, mask_patches_mod):
        img_patches_augmented[idx[0], idx[1]] = img_patche_mod
        mask_patches_augmented[idx[0], idx[1]] = mask_patche_mod

    return img_patches_augmented, mask_patches_augmented, new_ships_num 

def save_image(image, output_path):
    """
    Save a PIL Image to a file.

    :param image: PIL Image object.
    :param output_path: File path where the image will be saved.
    """
    if image.mode != 'L':
        image = image.convert('L')
    
    image.save(output_path) 

if __name__ == '__main__':
    base_dir = r'C:\Users\micha\Pulpit\Studia\E-learning\Semestr 10\Praca dyplomowa\SAR_Ship_Segmentation\data\processed\train'
    dest_dir = r'C:\Users\micha\Pulpit\Studia\E-learning\Semestr 10\Praca dyplomowa\SAR_Ship_Segmentation\data\augmented_v2\train'

    images_dir = os.path.join(base_dir, 'images')
    masks_dir = os.path.join(base_dir, 'masks')
    images_paths = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir))]
    masks_paths = [os.path.join(masks_dir, f) for f in sorted(os.listdir(masks_dir))]

    num_of_original_images = 0
    num_of_augmented_images = 0
    num_of_original_ships = 0
    num_of_added_ships = 0

    for image_path, mask_path in list(zip(images_paths, masks_paths)):
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        valid_bounding_boxes, all_ships_num = detect_ships(mask)

        if len(valid_bounding_boxes):
            img_ships = extract_parts(image, valid_bounding_boxes)
            mask_ships = extract_parts(mask, valid_bounding_boxes)

            img_patches = create_patches(image, patch_size=(100, 100), step=100)
            mask_patches = create_patches(mask, patch_size=(100, 100), step=100)
            
            img_patches_new, mask_patches_new, added_ships_num = apply_augmentation(img_patches, mask_patches, img_ships, mask_ships, 3)
            re_img = reconstruct_image(img_patches_new, (800, 800))
            re_mask = reconstruct_image(mask_patches_new, (800, 800))

            save_image(re_img, os.path.join(dest_dir, "images", os.path.basename(image_path).replace('.jpg', '_AUG1.jpg')))
            save_image(re_mask, os.path.join(dest_dir, "masks", os.path.basename(mask_path).replace('.png', '_AUG1.png')))

            num_of_augmented_images += 1
            num_of_added_ships += added_ships_num
        
        else:
            save_image(image, os.path.join(dest_dir, "images", os.path.basename(image_path)))
            save_image(mask, os.path.join(dest_dir, "masks", os.path.basename(mask_path)))

        num_of_original_images += 1
        num_of_original_ships += all_ships_num

    print(f"Augmented images : {num_of_augmented_images}")
    print(f"Total proccesed images : {num_of_original_images}")
    print(f"Synthetic ships : {num_of_added_ships}")
    print(f"Original ships : {num_of_original_ships}")
    print(f"Total ships : {num_of_original_ships+num_of_added_ships}")