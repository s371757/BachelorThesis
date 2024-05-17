import os
import numpy as np
from skimage import io, segmentation
from skimage.util import img_as_float
from concurrent.futures import ThreadPoolExecutor
from skimage.transform import resize
from datastructures import Image, Superpixel

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image_data = img_as_float(io.imread(image_path))
            image_id = os.path.splitext(filename)[0]
            images.append(Image(image_id, image_data))
    print(f"Loaded {len(images)} images from {folder_path}")
    return images

def pad_superpixel(pixel_data, mask, padded_size=(299, 299, 3)):
    # Calculate bounding box of the superpixel
    mask_indices = np.where(mask)
    min_x, min_y = mask_indices[0].min(), mask_indices[1].min()
    max_x, max_y = mask_indices[0].max(), mask_indices[1].max()
    width, height = max_x - min_x + 1, max_y - min_y + 1

    # Create the background with gray scale value 117.5
    gray_value = 117.5 / 255.0
    background_size = max(width, height)
    background = np.full((background_size, background_size, 3), gray_value)

    # Extract the bounding box of the superpixel
    superpixel_data = np.full((width, height, 3), gray_value)
    for i in range(3):
        channel = pixel_data[min_x:max_x+1, min_y:max_y+1, i]
        superpixel_data[:, :, i] = np.where(mask[min_x:max_x+1, min_y:max_y+1], channel, gray_value)
    
    # Place the superpixel in the center of the background
    start_x = (background_size - width) // 2
    start_y = (background_size - height) // 2
    background[start_x:start_x+width, start_y:start_y+height, :] = superpixel_data

    # Resize the background to the padded size
    padded_image = resize(background, padded_size, anti_aliasing=True)

    return padded_image

def create_superpixels(image, n_segments=100):
    segments = segmentation.slic(image.image_data, n_segments=n_segments, compactness=10, start_label=1)
    image.superpixels_mask = segments
    superpixels = []
    
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        pixel_data = image.image_data * mask[:, :, np.newaxis]
        
        padded_data = pad_superpixel(image.image_data, mask)
        superpixel = Superpixel(segment_id, image.image_id, pixel_data, padded_data)
        superpixels.append(superpixel)
    
    return superpixels

def extract_superpixels_from_images(images):
    with ThreadPoolExecutor() as executor:
        all_superpixels_for_class = []
        futures = {executor.submit(create_superpixels, image): image for image in images}
        for future in futures:
            image = futures[future]
            superpixels = future.result()
            for sp in superpixels:
                image.add_superpixel(sp)
                all_superpixels_for_class.append(sp)
    print("Superpixels extracted for all images")
    return all_superpixels_for_class

# Main function
def main(folder_path):
    images = load_images_from_folder(folder_path)
    all_superpixels_for_class = extract_superpixels_from_images(images)
    return images, all_superpixels_for_class

# Example usage
if __name__ == "__main__":
    folder_path = 'imagenet_dataset/anchor'  # Make sure this path is correct
    if not os.path.exists(folder_path):
        print(f"The folder path {folder_path} does not exist.")
    else:
        print(f"Processing images in folder: {folder_path}")
        images = main(folder_path)
        if images:
            for image in images:
                image.display_superpixels()
        else:
            print("No images found to process.")
