import random
import numpy as np
from pyrsgis import raster
from sklearn.utils import resample
from sklearn.feature_extraction import image

def imageChipsFromSingleBandArray(data_arr, y_size=5, x_size=5):
    # Pad using reflect mode and extract patches
    padded = np.pad(data_arr, ((y_size // 2,), (x_size // 2,)), mode='reflect')
    return image.extract_patches_2d(padded, (y_size, x_size))


def imageChipsFromArray_update(data_array, x_size=5, y_size=5):
    if data_array.ndim == 2:  # Single-band image
        return imageChipsFromSingleBandArray(data_array, y_size=y_size, x_size=x_size)

    elif data_array.ndim > 2:  # Multi-band image
        data_array = np.moveaxis(data_array, 0, -1)  # Shape: (H, W, Bands)
        chips = [imageChipsFromSingleBandArray(data_array[..., b], y_size=y_size, x_size=x_size)
                 for b in range(data_array.shape[-1])]
        return np.stack(chips, axis=-1)  # Shape: (num_patches, y_size, x_size, Bands)

    else:
        raise ValueError("Input array must be 2D or 3D.")


def load_and_prepare_data(feature_file, label_file):
    """
    Loads satellite imagery and labels, normalizes features,
    creates image chips, and balances classes.
    """
    print("Loading data...")

    # Read data
    ds_features, feature_arr = raster.read(feature_file, bands='all')
    ds_labels, label_arr = raster.read(label_file, bands='all')

    # Normalize features (vectorized)
    band_min = feature_arr.min(axis=(1, 2), keepdims=True)
    band_max = feature_arr.max(axis=(1, 2), keepdims=True)
    arrFeatures_fuzzy = (feature_arr - band_min) / (band_max - band_min + 1e-8)

    # Create chips
    features_chips = imageChipsFromArray_update(arrFeatures_fuzzy, x_size=9, y_size=9)

    # Adjust labels
    arrPositiveLabels = label_arr + 1
    n_class_pos = len(np.unique(arrPositiveLabels))

    # Find zero-label positions
    zero_positions = np.argwhere(arrPositiveLabels == 0)
    non_zero_count = np.count_nonzero(arrPositiveLabels)

    # Random negative sampling
    num_samples = int(non_zero_count / (n_class_pos - 1))
    sampled_idx = random.sample(range(zero_positions.shape[0]), num_samples)

    arrNegativeLabels = np.zeros_like(arrPositiveLabels)
    arrNegativeLabels[tuple(zero_positions[sampled_idx].T)] = n_class_pos

    # Flatten
    arrPositiveLabels_flat = arrPositiveLabels.ravel()
    positive_mask = arrPositiveLabels_flat != 0

    # Select positive samples
    features = features_chips[positive_mask]
    labels = arrPositiveLabels_flat[positive_mask]


    # Coastal upsampling
    coastal_mask = arrPositiveLabels_flat == 6
    temp = np.count_nonzero(arrPositiveLabels_flat == 1) - np.count_nonzero(coastal_mask)
    print('temp',temp)
    if temp > 0:
        coastal_upsample = resample(features_chips[coastal_mask],
                                    replace=True,
                                    n_samples=temp,
                                    random_state=42)
        features = np.vstack([features, coastal_upsample])
        labels = np.hstack([labels, [6] * temp])

    labels = labels - 1
    print(f"Data prepared. Total samples: {len(labels)}")

    return features, labels


def load(feature_file):
    """
    Loads satellite imagery and labels, normalizes features,
    creates image chips, and balances classes.
    """
    print("Loading data...")

    # Read data
    ds_features, feature_arr = raster.read(feature_file, bands='all')

    min_x = ds_features.bbox[0][0]
    max_x = ds_features.bbox[1][0]
    min_y = ds_features.bbox[0][1]
    max_y = ds_features.bbox[1][1]

    extent = [min_x, max_x, min_y, max_y]

    # Normalize features (vectorized)
    band_min = feature_arr.min(axis=(1, 2), keepdims=True)
    band_max = feature_arr.max(axis=(1, 2), keepdims=True)
    arrFeatures_fuzzy = (feature_arr - band_min) / (band_max - band_min + 1e-8)

    # Create chips
    features_chips = imageChipsFromArray_update(arrFeatures_fuzzy, x_size=9, y_size=9)
    return features_chips, extent