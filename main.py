import random
import torch
import os
from transformers import AutoFeatureExtractor, AutoModel
from PIL import Image
import numpy as np
from skimage import io, segmentation, transform
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean


from .slic_segmentation import perform_slic, resize_and_pad_segments, img_as_float

# Define constants
TARGET_SIZE = (299, 299)
PADDING_VALUE = 117.5
N_CLUSTERS = 25
N_CLOSEST = 40

def load_inception_v3_model():
    feature_extractor = AutoFeatureExtractor.from_pretrained("google/inception_v3")
    model = AutoModel.from_pretrained("google/inception_v3", output_hidden_states=True)
    return feature_extractor, model

def load_image(image_path):
    return img_as_float(io.imread(image_path))

def resize_and_pad_segments(image, segments, target_size=TARGET_SIZE, padding_value=PADDING_VALUE):
    unique_segments = np.unique(segments)
    resized_segments = []
    for seg_val in unique_segments:
        mask = segments == seg_val
        segment = image * mask[..., np.newaxis]
        segment_resized = transform.resize(segment, target_size, anti_aliasing=True)
        pad_height = target_size[0] - segment_resized.shape[0]
        pad_width = target_size[1] - segment_resized.shape[1]
        padded_segment = np.pad(segment_resized, ((0, pad_height), (0, pad_width), (0, 0)), 'constant', constant_values=padding_value)
        resized_segments.append(padded_segment)
    return resized_segments

def extract_features(segment, feature_extractor, model):
    segment_image = Image.fromarray((segment * 255).astype(np.uint8))
    inputs = feature_extractor(images=segment_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    features = outputs.hidden_states[-5].mean(dim=[1, 2]).squeeze().numpy()  # "mixed_8" layer
    return features

def get_image_paths_for_class(class_dir, num_images=50):
    all_images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
    return random.sample(all_images, num_images)

def cluster_segments(features, n_clusters=N_CLUSTERS):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    return kmeans

def filter_clusters(kmeans, features, metadata, n_closest=N_CLOSEST):
    filtered_clusters = []
    labels = kmeans.labels_
    for cluster_id in range(kmeans.n_clusters):
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_features = np.array([features[i] for i in cluster_indices])
        cluster_metadata = [metadata[i] for i in cluster_indices]
        distances = np.linalg.norm(cluster_features - kmeans.cluster_centers_[cluster_id], axis=1)
        closest_indices = np.argsort(distances)[:n_closest]
        filtered_clusters.append([cluster_metadata[i] for i in closest_indices])
    return filtered_clusters

def filter_based_on_frequency(filtered_clusters, total_images_per_class):
    final_clusters = []
    for cluster in filtered_clusters:
        image_ids = [item['image_id'] for item in cluster]
        unique_images = len(set(image_ids))
        if unique_images > total_images_per_class / 2:
            final_clusters.append(cluster)
        elif unique_images > total_images_per_class / 4 and len(cluster) > total_images_per_class:
            final_clusters.append(cluster)
        elif len(cluster) > 2 * total_images_per_class:
            final_clusters.append(cluster)
    return final_clusters

# Example usage
def main():
    # Load the Inception-V3 model
    feature_extractor, model = load_inception_v3_model()

    # Assuming we have a structure like: /path/to/imagenet/class_name/image.jpg
    imagenet_dir = '/path/to/imagenet'
    class_dirs = [os.path.join(imagenet_dir, class_dir) for class_dir in os.listdir(imagenet_dir) if os.path.isdir(os.path.join(imagenet_dir, class_dir))]
    selected_classes = random.sample(class_dirs, 100)

    all_segments_features = []
    all_segments_metadata = []

    for class_dir in selected_classes:
        image_paths = get_image_paths_for_class(class_dir)
        for image_id, image_path in enumerate(image_paths):
            image = load_image(image_path)
            for num_segments in [15, 50, 80]:
                segments = perform_slic(image, num_segments)
                padded_segments = resize_and_pad_segments(image, segments)
                for seg_id, segment in enumerate(padded_segments):
                    features = extract_features(segment, feature_extractor, model)
                    all_segments_features.append(features)
                    all_segments_metadata.append({'class_dir': class_dir, 'image_id': image_id, 'seg_id': seg_id})

    # Convert features to numpy array
    all_segments_features = np.array(all_segments_features)

    # Perform clustering
    kmeans = cluster_segments(all_segments_features)

    # Filter clusters
    filtered_clusters = filter_clusters(kmeans, all_segments_features, all_segments_metadata)

    # Filter based on frequency and popularity criteria
    total_images_per_class = 50  # Since we use 50 images per class
    final_clusters = filter_based_on_frequency(filtered_clusters, total_images_per_class)

    # Visualize or further process final_clusters as needed
    for cluster in final_clusters:
        print(f"Cluster of size {len(cluster)}:")
        for segment in cluster:
            print(segment)

if __name__ == "__main__":
    main()
