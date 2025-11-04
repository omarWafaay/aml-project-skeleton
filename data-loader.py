import os
import requests
import zipfile
from io import BytesIO

# Define target folder
dataset_folder = "dataset"
zip_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

# Create the folder if it doesn't exist

# Download and extract the dataset
print("Downloading Tiny ImageNet...")
response = requests.get(zip_url)
print("Extracting to:", dataset_folder)

with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(dataset_folder)

print("Done! Dataset is ready in:", dataset_folder)