import os
import numpy as np
from skimage import io, segmentation
from skimage.util import img_as_float
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

class Superpixel:
    def __init__(self, superpixel_id, image_id, original_data, padded_data):
        self.superpixel_id = superpixel_id
        self.image_id = image_id
        self.original_data = original_data
        self.padded_data = padded_data

class Image:
    def __init__(self, image_id, image_data):
        self.image_id = image_id
        self.image_data = image_data
        self.superpixels = []

    def add_superpixel(self, superpixel):
        self.superpixels.append(superpixel)
    
    def display_superpixels(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display the original image with superpixel boundaries
        image_with_boundaries = segmentation.mark_boundaries(self.image_data, self.superpixels_mask)
        ax[0].imshow(image_with_boundaries)
        ax[0].set_title(f'Original Image with Superpixels - {self.image_id}')
        
        # Display the padded superpixels
        for i, sp in enumerate(self.superpixels):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(sp.padded_data)
            ax.set_title(f'Padded Superpixel {i} - {self.image_id}')
            plt.show()

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

def pad_superpixel(pixel_data, mask, padded_size=(255, 255, 3)):
    padded_image = np.full(padded_size, 0.5)  # Grey background
    mask_indices = np.where(mask)
    min_x, min_y = mask_indices[0].min(), mask_indices[1].min()
    max_x, max_y = mask_indices[0].max(), mask_indices[1].max()
    width, height = max_x - min_x + 1, max_y - min_y + 1

    # Extract the bounding box of the superpixel
    superpixel_data = np.zeros((width, height, 3))
    for i in range(3):
        superpixel_data[:, :, i] = pixel_data[min_x:max_x+1, min_y:max_y+1, i] * mask[min_x:max_x+1, min_y:max_y+1]
    
    start_x = (padded_size[0] - width) // 2
    start_y = (padded_size[1] - height) // 2
    
    # Ensure the dimensions match
    if (start_x + width) <= padded_size[0] and (start_y + height) <= padded_size[1]:
        padded_image[start_x:start_x+width, start_y:start_y+height, :] = superpixel_data
    else:
        print(f"Skipping superpixel due to dimension mismatch: {width}x{height} cannot fit in {padded_size}")
    
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
        futures = {executor.submit(create_superpixels, image): image for image in images}
        for future in futures:
            image = futures[future]
            superpixels = future.result()
            for sp in superpixels:
                image.add_superpixel(sp)
    print("Superpixels extracted for all images")

# Main function
def main(folder_path):
    images = load_images_from_folder(folder_path)
    extract_superpixels_from_images(images)
    return images

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
