# Required imports
import numpy as np
import pandas as pd
import rasterio
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load carbon CSV
CSV_PATH = 'carbon_stock_data/Yogyakarta_Carbon_Stock.csv'
FEATURES = ['CA', 'CB', 'CS']
dataset = pd.read_csv(CSV_PATH)
dataset = dataset[FEATURES]

# Load TIF image
IMAGE_PATH = 'saved_data/Output_RF.tif'
image = rasterio.open(IMAGE_PATH)
height = image.height
width = image.width
image_data = image.read()
num_bands = image_data.shape[0]

# Flatten image data to 2D (pixels, bands)
image_data_flat = image_data.reshape((-1, num_bands))

# Exclude NaN pixels
nan_mask = np.isnan(image_data_flat).any(axis=1)
valid_image_data = image_data_flat[~nan_mask]

# Downsample valid TIF pixels to match CSV rows
num_csv_rows = dataset.shape[0]
downsampled_indices = np.linspace(0, valid_image_data.shape[0] - 1, num=num_csv_rows, dtype=int)
downsampled_image_data = valid_image_data[downsampled_indices]

# Normalize both datasets
scaler = StandardScaler()
image_data_normalized = scaler.fit_transform(downsampled_image_data)
csv_features_normalized = scaler.fit_transform(dataset)

# Apply weights
image_weight = 1
csv_weight = 1
weighted_image_data = image_data_normalized * image_weight
weighted_csv_features = csv_features_normalized * csv_weight

# Combine weighted TIF and CSV data
combined_features = np.hstack((weighted_image_data, weighted_csv_features))

# KMeans clustering
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(combined_features)
cluster_labels = kmeans.labels_

# Assign labels to valid pixels in downsampled indices
clustered_full = np.full(valid_image_data.shape[0], np.nan)
for i, idx in enumerate(downsampled_indices):
    clustered_full[idx] = cluster_labels[i]

# Map clustered_full back to full array, including NaN pixels
full_clustered_image = np.full(image_data_flat.shape[0], np.nan)
full_clustered_image[~nan_mask] = clustered_full

# Reshape clustered data to original TIF dimensions
clustered_image = full_clustered_image.reshape(height, width)

# Save clustered image as new TIF file
output_path = 'saved_data/clustered_image_output_weighted.tif'
profile = image.profile
profile.update(dtype=rasterio.float32, count=1)
with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(clustered_image.astype(np.float32), 1)

# Visualize clustered image
plt.figure(figsize=(8, 8))
plt.imshow(clustered_image, cmap='viridis', interpolation='nearest')
plt.title('Clustered Image (Weighted Features)')
plt.colorbar(label='Cluster Labels')
plt.axis('off')
plt.show()

print(f'Clustered image saved to: {output_path}')
