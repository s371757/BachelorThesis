import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
from transformers import AutoFeatureExtractor, AutoModel
import slic_segmentation 

from datastructures import Cluster

def prepare_superpixel_data(images):
    print("Preparing superpixel data for clustering")
    superpixel_data = []
    superpixel_objects = []
    for image in images:
        for sp in image.superpixels:
            superpixel_data.append(sp.padded_data.flatten())
            superpixel_objects.append(sp)
    print(f"Prepared superpixel data for {len(superpixel_objects)} superpixels")
    return np.array(superpixel_data), superpixel_objects

def cluster_superpixels(superpixel_data, n_clusters=25):
    print(f"Clustering superpixels into {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(superpixel_data)
    print("Superpixels clustered")
    return labels, kmeans.cluster_centers_

def create_clusters(labels, superpixel_objects, n_clusters):
    print("Creating clusters")
    clusters = [Cluster(cluster_id) for cluster_id in range(n_clusters)]
    for sp, label in zip(superpixel_objects, labels):
        clusters[label].add_superpixel(sp)
    print(f"Created {len(clusters)} clusters")
    return clusters

def filter_clusters(clusters, discovery_images_count):
    print("Filtering clusters")
    filtered_clusters = []
    for cluster in clusters:
        cluster.compute_center()
        cluster.filter_superpixels()
        if cluster.remove_unpopular_clusters(discovery_images_count):
            filtered_clusters.append(cluster)
    print(f"Filtered clusters down to {len(filtered_clusters)}")
    return filtered_clusters

def extract_inception_features(superpixels):
    print("Extracting Inception-V3 features for superpixels using Hugging Face")
    feature_extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
    
    features = []
    for sp in superpixels:
        padded_segment = sp.padded_data
        inputs = feature_extractor(images=padded_segment, return_tensors="pt", do_rescale=False)
        outputs = model(**inputs)
        feature = outputs.last_hidden_state[:, 1, :].detach().numpy().flatten()
        features.append(feature)
        sp.feature_vector = feature
    
    print(f"Extracted features for {len(features)} superpixels")
    return np.array(features)

def main_clustering(superpixels, n_clusters=25):
    print("Starting clustering process")
    superpixel_features = extract_inception_features(superpixels)
    labels, _ = cluster_superpixels(superpixel_features, n_clusters)
    clusters = create_clusters(labels, superpixels, n_clusters)
    discovery_images_count = len(set(sp.image_id for sp in superpixels))
    filtered_clusters = filter_clusters(clusters, discovery_images_count)
    print("Clustering process completed")
    return filtered_clusters

# Example usage
if __name__ == "__main__":
    import slic_segmentation

    folder_path = 'imagenet_dataset/anchor'  # Make sure this path is correct
    images, superpixels = slic_segmentation.main(folder_path)  # Run the segmentation to get the images with superpixels
    
    if images:
        print(f"Loaded {len(images)} images for clustering")
        clusters = main_clustering(superpixels, n_clusters=25)
        print(f"Generated {len(clusters)} clusters")

        # Display images of a specific cluster
        cluster_id_to_display = 0  # Change this to the cluster you want to display
        if clusters:
            clusters[cluster_id_to_display].display_images()
        else:
            print("No clusters to display")
    else:
        print("No images found to process")