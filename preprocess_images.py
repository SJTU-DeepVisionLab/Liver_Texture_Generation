import os
import cv2
from PIL import Image
import glob
from tqdm import tqdm
import numpy as np

def invert_masks(mask_image):
    '''
    Invert the mask image, i.e., if the mask is white, then set it to black, and vice versa.
    '''
    mask_image = cv2.bitwise_not(mask_image)
    return mask_image

def invert_masks_from_files(mask_dir, output_dir):
    '''
    Invert the masks in the mask_dir and save the inverted masks to the output_dir.
    '''
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    assert len(mask_files) > 0, "No mask files found in the mask_dir."
    for mask_file in tqdm(mask_files):
        mask_image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        inverted_mask = invert_masks(mask_image)
        output_file = os.path.join(output_dir, os.path.basename(mask_file))
        cv2.imwrite(output_file, inverted_mask)

def create_masked_image(image_path, mask_path, output_path):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
    image_np = np.array(image)
    mask_np = np.array(mask)
    
    white_bg = np.ones_like(image_np) * 255
    mask_np = mask_np[:,:,np.newaxis] / 255.0
    masked_image = image_np * mask_np + white_bg * (1 - mask_np)
    
    masked_image = Image.fromarray(masked_image.astype(np.uint8))
    masked_image.save(output_path)

def generate_masked_images(data_root):
    '''
    Generate masked images from the original images and masks.
    '''
    image_dir = os.path.join(data_root, "1031")
    mask_dir = os.path.join(data_root, "1031-mask")
    masked_dir = os.path.join(data_root, "1031-masked_images")
    os.makedirs(masked_dir, exist_ok=True)
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    
    assert len(image_files) == len(mask_files), "The number of images and masks should be the same."
    for image_file, mask_file in tqdm(zip(image_files, mask_files)):
        output_file = os.path.join(masked_dir, os.path.basename(image_file))
        create_masked_image(image_file, mask_file, output_file)
    print("Masked images have been saved to:", masked_dir)


if __name__ == "__main__":
    mask_dir = "/data_hdd3/users/yangchen/Ruijing_medical_projects/new_data/liver1031/1031-mask"
    output_dir = "/data_hdd3/users/yangchen/Ruijing_medical_projects/new_data/liver1031/1031-inverted-masks"
    os.makedirs(output_dir, exist_ok=True)
    invert_masks_from_files(mask_dir, output_dir)
    print("Inverted masks have been saved to:", output_dir)
    data_root = "/data_hdd3/users/yangchen/Ruijing_medical_projects/new_data/liver1031"
    generate_masked_images(data_root)