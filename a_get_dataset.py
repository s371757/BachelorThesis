from datasets import load_dataset
import os
import shutil

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("imagenet-1k")

# Define constant variable for the filename
csv_filename = "imagenet_dataset.csv"

# Export dataset to CSV
dataset.export_to_csv(csv_filename)

# Convert CSV to folder
folder_path = "imagenet_dataset"
os.makedirs(folder_path, exist_ok=True)

# Move CSV files to the folder
shutil.move(csv_filename, os.path.join(folder_path, csv_filename))
# Create README file
readme_text = "This folder contains the ImageNet dataset in CSV format."
with open(os.path.join(folder_path, "README.txt"), "w") as readme_file:
    readme_file.write(readme_text)

print("ImageNet dataset packaged into folder:", folder_path)
