CNN for Remote Sensing Land Cover ClassificationThis project uses a Convolutional Neural Network (CNN) with TensorFlow/Keras to perform supervised land cover classification on remote sensing imagery. The training process is validated using K-Fold cross-validation to ensure the model's robustness.Project Structure.
├── data/                  # Folder for input GeoTIFF files
├── results/               # Output folder for plots and the saved model
├── data_loader.py         # Script to load and preprocess data
├── model.py               # Contains the CNN model architecture
├── plotting.py            # Functions for generating plots (metrics, ROC, etc.)
├── train.py               # Main script to execute model training and validation
├── predict.py             # Example script to run inference with the trained model
├── requirements.txt       # Required Python packages
└── README.md              # This file
How to Run1. SetupFirst, install the required Python packages:pip install -r requirements.txt
2. Train the ModelRun the main training script. This will perform 10-fold cross-validation, save the best model, and generate evaluation plots in the results/ folder.python train.py
4. Make a PredictionTo use the trained model for prediction on a new image, you can adapt the predict.py script.
This will output a classified land cover map in the results/ directory.