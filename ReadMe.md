# Documentation: Deploying the Random Forest Model in Backend
### See Test_Saved_Model.ipynb for more context

This guide provides instructions on how to load and use the pre-trained Random Forest model for deployment in your backend application. The backend is responsible for retrieving TIF files, processing them, and returning the classification results as TIF files.

## 1. Prerequisites

Ensure the following packages are installed in your backend environment:

- `tensorflow` (compatible with TensorFlow Decision Forests)
- `tensorflow_decision_forests`
- `pandas`
- `numpy`
- `rasterio`
- `earthpy`

You can install them using pip:

```bash
pip install tensorflow tensorflow_decision_forests pandas numpy rasterio earthpy
```

## 2. Model Information

- **Model Path**: `saved_data/NDVI_RF_V1`
- **Input Features**:
  - `B1`, `B2`, `B3`, `B4`, `B5`, `B6`, `B7`
  - `EVI`, `NBR`, `NDMI`, `NDWI`, `NDBI`, `NDBaI`, `NDVI`
  - `elevation`

## 3. Loading the Model

Example of loading the model using TensorFlow Decision Forests:

```python
import tensorflow as tf

# Load the pre-trained model
model = tf.saved_model.load('saved_data/NDVI_RF_V1')
```

## 4. Processing TIF Files

### 4.1. Loading the TIF File

Use `rasterio` to load the TIF file:

```python
import rasterio

# Load image
image = rasterio.open('path_to_input_image.tif')
height = image.height
width = image.width
crs = image.crs
transform = image.transform
shape = (height, width)
```

### 4.2. Preparing the Data

Extract the necessary features from the TIF file:

```python
import pandas as pd
import numpy as np

FEATURES = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 
            'EVI', 'NBR', 'NDMI', 'NDWI', 'NDBI',
            'NDBaI', 'NDVI', 'elevation']

used_image_feature = [i + 1 for i in range(len(FEATURES)) if image.descriptions[i] in FEATURES]
image_data = []
for i in used_image_feature:
    band_data = image.read(i).flatten()
    image_data.append(band_data)

# Create DataFrame for image data
image_df = pd.DataFrame(np.array(image_data).T, columns=FEATURES)
image_df['elevation'] = image_df['elevation'].astype(np.int64)
```

### 4.3. Making Predictions

Use the loaded model to make predictions:

```python
# Use the dictionary as input to the model
prediction = model(dict(image_df))
image_predicted_classes = np.argmax(prediction, axis=1)

# Reshape predictions to image dimensions
image_predicted_classes = image_predicted_classes.reshape(shape)
```

### 4.4. Saving the Prediction as TIF

Save the classified image to a TIF file:

```python
import rasterio

# Define save location
save_location = 'saved_data/Output_RF.tif'

# Save the classified image to a file
with rasterio.open(
    save_location,
    'w',
    driver='GTiff',
    height=image_predicted_classes.shape[0],
    width=image_predicted_classes.shape[1],
    count=1,
    dtype=image_predicted_classes.dtype.name,
    crs=crs,
    transform=transform,
) as dst:
    dst.write(image_predicted_classes, 1)

print('File saved to ' + save_location)
```

## 5. Integrating with Backend

Implement the following workflow in your backend application:

1. **Retrieve the TIF File**: The backend should receive or access the input TIF file that needs classification.

2. **Load the Model**: Use the instructions above to load the pre-trained Random Forest model.

3. **Process the TIF File**:
    - **Load the TIF File**: Use `rasterio` to open and read the TIF file.
    - **Prepare the Data**: Extract the required features and convert them into a `pandas` DataFrame.
    - **Convert to TensorFlow Dataset**: Transform the DataFrame into a TensorFlow dataset compatible with the model.
    - **Make Predictions**: Use the model to predict the class for each pixel.

4. **Save the Prediction**: Save the predicted classes as a new TIF file using `rasterio`.

5. **Return the TIF File**: Provide the processed TIF file as a response or store it in a designated location for retrieval.

## 6. Handling Model Dependencies

Ensure that all necessary model files and dependencies are included when deploying the backend application. This includes:

- The saved model directory (`saved_data/NDVI_RF_V1`)
- All required Python packages as listed in the prerequisites

## 7. Testing the Deployment

After deploying, test the backend by processing sample TIF files to verify that predictions are generated and saved correctly. Ensure that the backend handles file retrieval, processing, and response efficiently and accurately.

## 8. Security and Optimization

- **Security**: Validate and sanitize all input files to prevent security vulnerabilities.
- **Optimization**: Handle large files efficiently and manage server resources to ensure scalability.

## 9. Additional Notes

- The backend processes TIF files directly, reducing the preprocessing burden on the frontend.
- Ensure the backend server has sufficient permissions to read and write files in the specified directories.
- Customize the backend application as needed to fit the specific requirements of your deployment environment.
- The file is saved as a GeoTIFF, which means it contains geospatial data that can be imported into Google Earth Engine and displayed on an Android app using the Google Earth API.