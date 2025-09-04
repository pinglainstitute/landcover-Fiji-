import os
import numpy as np
from tensorflow.keras.models import load_model
from data_loader import load, plot_Classified_img

def predict_land_cover(feature_arr, extent, year):
    """
    Predicts land cover for a given satellite image using a trained model.

    Args:
        model_path (str): Path to the trained .h5 model file.
        input_image_path (str): Path to the input GeoTIFF image.
        output_image_path (str): Path to save the classified GeoTIFF image.
    """
    # print(f"Loading model from: {model_path}")
    RESULTS_DIR = 'results'
    model = load_model(os.path.join(RESULTS_DIR, 'best_land_cover_model.h5'))
    
    # print(f"Reading input image: {input_image_path}")
    n_class_pos = 7

    print("Predicting land cover classes...")
    predictions = model.predict(feature_arr)
    predicted_labels = np.argmax(predictions, axis=1)

    NAME = 'classified_'+ str(year) + '.png'

    plot_Classified_img(predicted_labels, extent, n_class_pos, os.path.join(RESULTS_DIR, NAME))

    
    # print(f"Saving classified image to: {output_image_path}")



if __name__ == "__main__":
    for i in range(2013,2025):
        year = str(i)
        feature_arr, extent = load('/data/'+year+'_original.tif')

        predict_land_cover(feature_arr, extent, i)