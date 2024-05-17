import random
import numpy as np
from skimage import segmentation
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import os

import pickle

# Define a function to save an object to a file
def save_object(obj, filename):
    with open(filename, 'wb') as out_file:
        pickle.dump(obj, out_file, pickle.HIGHEST_PROTOCOL)

# Define a function to load an object from a file
def load_object(filename):
    with open(filename, 'rb') as in_file:
        return pickle.load(in_file)
    
def create_superpixels_directory(directory="superpixels"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

class Image:
    def __init__(self, image_id, image_data):
        self.image_id = image_id
        self.image_data = image_data
        self.superpixels = []

    def add_superpixel(self, superpixel):
        self.superpixels.append(superpixel)
    
    def display_superpixels(self):
        # Display the original image with superpixel boundaries
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        image_with_boundaries = segmentation.mark_boundaries(self.image_data, self.superpixels_mask)
        ax[0].imshow(image_with_boundaries)
        ax[0].set_title(f'Original Image with Superpixels - {self.image_id}')
        
        # Display a random padded superpixel
        if self.superpixels:
            sp = random.choice(self.superpixels)
            ax[1].imshow(sp.padded_data)
            ax[1].set_title(f'Padded Superpixel {sp.superpixel_id} - {self.image_id}')
        
        plt.show()


class Superpixel:
    def __init__(self, superpixel_id, image_id, original_data, padded_data):
        self.superpixel_id = superpixel_id
        self.image_id = image_id
        self.original_data = original_data
        self.padded_data = padded_data
        self.feature_vector = None


class Cluster:
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.superpixels = []
        self.center = None

    def add_superpixel(self, superpixel):
        self.superpixels.append(superpixel)
        print(f"Added superpixel {superpixel.superpixel_id} to cluster {self.cluster_id}")

    def compute_center(self):
        print(f"Computing center for cluster {self.cluster_id}")
        data = np.array([sp.padded_data.flatten() for sp in self.superpixels])
        self.center = np.mean(data, axis=0)
        print(f"Center for cluster {self.cluster_id} computed")

    def filter_superpixels(self, n=40):
        print(f"Filtering superpixels for cluster {self.cluster_id}")
        print("Originally there were: ", len(self.superpixels), "superpixels in this cluster")
        distances = [np.linalg.norm(sp.padded_data.flatten() - self.center) for sp in self.superpixels]
        sorted_superpixels = sorted(zip(distances, self.superpixels))
        self.superpixels = [sp for _, sp in sorted_superpixels[:n]]
        print(f"Filtered superpixels for cluster {self.cluster_id}")

    def remove_unpopular_clusters(self, discovery_images_count):
        print(f"Removing unpopular clusters for cluster {self.cluster_id}")
        image_ids = [sp.image_id for sp in self.superpixels]
        image_count = len(set(image_ids))
        cluster_size = len(self.superpixels)

        high_frequency = image_count >= discovery_images_count // 2
        medium_frequency = image_count >= discovery_images_count // 4 and cluster_size > discovery_images_count
        high_popularity = cluster_size >= 2 * discovery_images_count

        is_popular = high_frequency or medium_frequency or high_popularity
        print(f"Cluster {self.cluster_id} {'is' if is_popular else 'is not'} popular")
        print(f"Image count: {image_count}, Cluster size: {cluster_size}", "Discovery images count:", discovery_images_count)
        return is_popular

    def display_images(self):
        print(f"Displaying images for cluster {self.cluster_id}")
        for sp in self.superpixels:
            plt.figure(figsize=(10, 10))
            plt.title(f"Superpixel ID: {sp.superpixel_id}")
            plt.imshow(sp.padded_data)
            plt.axis('off')
            plt.show()

