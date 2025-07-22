# Install some packages (uncomment if needed)
# pip install rasterio
# pip install earthpy

# Imports
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import pandas as pd
import numpy as np
import rasterio
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
import earthpy.plot as ep
import os

# Suppress TensorFlow INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Parameter
ALL_FEATURES = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'EVI', 'NBR', 'NDMI', 'NDWI', 'NDBI', 'NDBaI', 'NDVI', 'elevation']
FEATURES = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'EVI', 'NBR', 'NDMI', 'NDWI', 'NDBI', 'NDBaI', 'NDVI', 'elevation']
LABEL = ['classvalue']
CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
PALETTE = ['#F08080', '#D2B48C', '#87CEFA', '#008080', '#90EE90', '#228B22', '#808000', '#006400', '#FF8C00']
SAMPLE_PATH = 'ndvi_data/Samples_LC_JambiFixedWatersAllFeatures_2023.csv'
IMAGE_PATH = 'ndvi_data/Landsat_Semarang_2023.tif'

# Load image
image = rasterio.open(IMAGE_PATH)
height = image.height
width = image.width
shape = (height, width)

# (Optional) Visualize the image
image_vis = []
for x in [6, 5, 4]:
    image_vis.append(image.read(x))
image_vis = np.stack(image_vis)
ep.plot_rgb(image_vis, figsize=(8, 8), stretch=True)
plt.show()

# Read samples
samples = pd.read_csv(SAMPLE_PATH)
samples = samples.sample(frac=1)
# Ensure label is integer type
samples['classvalue'] = samples['classvalue'].astype('int32')
print(samples)

# Split into train and test based on 'sample' column
train = samples[samples['sample'] == 'train']
test = samples[samples['sample'] == 'test']

# Prepare data for TFDF
train_data = train[FEATURES + LABEL]
test_data = test[FEATURES + LABEL]

filtered_train = train[FEATURES + LABEL]
Label1 = filtered_train[filtered_train['classvalue'] == 1]
Label2 = filtered_train[filtered_train['classvalue'] == 2]
Label3 = filtered_train[filtered_train['classvalue'] == 3]
Label4 = filtered_train[filtered_train['classvalue'] == 4]
Label5 = filtered_train[filtered_train['classvalue'] == 5]
Label6 = filtered_train[filtered_train['classvalue'] == 6]
Label7 = filtered_train[filtered_train['classvalue'] == 7]
Label8 = filtered_train[filtered_train['classvalue'] == 8]
Label9 = filtered_train[filtered_train['classvalue'] == 9]

print("Label 1", Label1["classvalue"].count())
print("Label 2", Label2["classvalue"].count())
print("Label 3", Label3["classvalue"].count())
print("Label 4", Label4["classvalue"].count())
print("Label 5", Label5["classvalue"].count())
print("Label 6", Label6["classvalue"].count())
print("Label 7", Label7["classvalue"].count())
print("Label 8", Label8["classvalue"].count())
print("Label 9", Label9["classvalue"].count())

# Convert to TensorFlow datasets
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_data, label=LABEL[0])
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_data, label=LABEL[0])

best_num_trees = 50
best_max_depth = 10
best_min_examples = 2

# Define and train the best model
rf_model = tfdf.keras.RandomForestModel(
    task=tfdf.keras.Task.CLASSIFICATION,
    num_trees=best_num_trees,
    max_depth=best_max_depth,
    min_examples=best_min_examples
)

rf_model.fit(train_ds)

# Evaluate the best model
best_evaluation = rf_model.evaluate(test_ds, return_dict=True)
print(f"Best Model Evaluation: {best_evaluation}")

# Evaluate the model
evaluation = rf_model.evaluate(test_ds)
print(f"Model evaluation: {evaluation}")

# Predictions on test data
test_predictions = rf_model.predict(test_ds)
predicted_classes = np.argmax(test_predictions, axis=1)
true_classes = test_data[LABEL[0]].values

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes, labels=CLASSES, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot()
plt.show()

# Classification report
print(classification_report(true_classes, predicted_classes))

nodata_value = 0  # Replace with the identified value

# Read image features for prediction
image_data = []
used_image_features = [
    i + 1 for i in range(0, len(ALL_FEATURES)) if image.descriptions[i] in FEATURES
]
for i in used_image_features:
    band_data = image.read(i).flatten()
    image_data.append(band_data)

# Create DataFrame for image data
image_df = pd.DataFrame(np.array(image_data).T, columns=FEATURES)

# Automatically set invalid (no-data) pixels to NaN
image_df.replace(to_replace=[nodata_value, None], value=np.nan, inplace=True)

# Skip NaN rows during prediction
valid_pixels = ~image_df.isnull().any(axis=1)
valid_data = image_df[valid_pixels]

# Convert only valid data to TensorFlow dataset
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_data)

# Predict on valid data
valid_predictions = rf_model.predict(valid_ds)
valid_predicted_classes = np.argmax(valid_predictions, axis=1)

# Create an empty array for the output and fill in only valid predictions
image_predicted_classes = np.full(image_df.shape[0], np.nan)  # Initialize with NaN
image_predicted_classes[valid_pixels] = valid_predicted_classes

# Reshape predictions back to the original image dimensions
image_predicted_classes = image_predicted_classes.reshape((image.height, image.width))

# Visualize the predictions
cmap, norm = from_levels_and_colors(CLASSES, PALETTE, extend='max')
ep.plot_bands(image_predicted_classes, cmap=cmap, norm=norm, figsize=(8, 8))
plt.show()

# Saving the Model
MODEL_SAVED_PATH = 'saved_data/'
MODEL_NAME = 'NDVI_RF_V1'
rf_model.save(MODEL_SAVED_PATH + MODEL_NAME)

# import pickle
# with open(MODEL_SAVED_PATH + MODEL_NAME + '.pkl', 'wb') as model_file:
#     pickle.dump(rf_model, model_file)
