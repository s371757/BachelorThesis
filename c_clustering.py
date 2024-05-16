import numpy as np
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb


class Cluster:
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.superpixels = []
        self.center = None

    def add_superpixel(self, superpixel):
        self.superpixels.append(superpixel)
    
    def compute_center(self):
        data = np.array([sp.pixel_data.flatten() for sp in self.superpixels])
        self.center = np.mean(data, axis=0)

    def filter_superpixels(self, n=40):
        distances = [np.linalg.norm(sp.pixel_data.flatten() - self.center) for sp in self.superpixels]
        sorted_superpixels = [sp for _, sp in sorted(zip(distances, self.superpixels))]
        self.superpixels = sorted_superpixels[:n]

    def remove_unpopular_clusters(self, discovery_images_count):
        image_ids = [sp.image_id for sp in self.superpixels]
        image_count = len(set(image_ids))
        cluster_size = len(self.superpixels)

        high_frequency = image_count > discovery_images_count // 2
        medium_frequency = image_count > discovery_images_count // 4 and cluster_size > discovery_images_count
        high_popularity = cluster_size > 2 * discovery_images_count

        return high_frequency or medium_frequency or high_popularity

def prepare_superpixel_data(images):
    superpixel_data = []
    superpixel_objects = []
    for image in images:
        for sp in image.superpixels:
            superpixel_data.append(sp.pixel_data.flatten())
            superpixel_objects.append(sp)
    return np.array(superpixel_data), superpixel_objects

def cluster_superpixels(superpixel_data, n_clusters=25):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(superpixel_data)
    return labels, kmeans.cluster_centers_

def create_clusters(labels, superpixel_objects, n_clusters):
    clusters = [Cluster(cluster_id) for cluster_id in range(n_clusters)]
    for sp, label in zip(superpixel_objects, labels):
        clusters[label].add_superpixel(sp)
    return clusters

def filter_clusters(clusters, discovery_images_count):
    filtered_clusters = []
    for cluster in clusters:
        cluster.compute_center()
        cluster.filter_superpixels()
        if cluster.remove_unpopular_clusters(discovery_images_count):
            filtered_clusters.append(cluster)
    return filtered_clusters

def highlight_superpixels(image, superpixels, color=(255, 0, 0)):
    mask = np.zeros(image.shape[:2], dtype=bool)
    for sp in superpixels:
        x, y, w, h = sp.position
        mask[x:x+w, y:y+h] = True
    highlighted_image = mark_boundaries(image, mask, color=color)
    return highlighted_image

def display_cluster_images(cluster, images):
    for sp in cluster.superpixels:
        image = next(image for image in images if image.image_id == sp.image_id)
        highlighted_image = highlight_superpixels(image.image_data, [sp])
        plt.figure(figsize=(10, 10))
        plt.title(f"Image ID: {image.image_id} - Superpixel ID: {sp.superpixel_id}")
        plt.imshow(highlighted_image)
        plt.axis('off')
        plt.show()

# Main function to cluster superpixels
def main_clustering(images, n_clusters=25):
    superpixel_data, superpixel_objects = prepare_superpixel_data(images)
    labels, _ = cluster_superpixels(superpixel_data, n_clusters)
    clusters = create_clusters(labels, superpixel_objects, n_clusters)
    discovery_images_count = len(images)
    filtered_clusters = filter_clusters(clusters, discovery_images_count)
    return filtered_clusters

# Example usage
# Assuming `images` is a list of Image objects with superpixels already extracted
clusters = main_clustering(images, n_clusters=25)

# Display images of a specific cluster
cluster_id_to_display = 0  # Change this to the cluster you want to display
display_cluster_images(clusters[cluster_id_to_display], images)