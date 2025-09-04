import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import joblib

# Import our custom modules
from data_loader import load_and_prepare_data
from model import create_cnn_model, create_ann_model, create_rf_model
from plotting import plot_training_history, plot_confusion_matrix, plot_roc_curves

# --- Configuration ---
# File paths
FEATURE_FILE = '/data/2013_original.tif'
LABEL_FILE = '/data/Label_2013_Nadi_1750_samples.tif'
RESULTS_DIR = 'results'

# Model & Training parameters
N_SPLITS = 10
EPOCHS = 150
PATIENCE = 15
LABELS_NAME = ['Urban', 'Grass/Crop', 'Forest', 'Bare Soil', 
               'Water', 'Coastal', 'Alluvium'] # Adjust to your classes

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Train land-cover classifier")
    parser.add_argument(
        "model",
        nargs='?',
        default='cnn',
        choices=['cnn', 'ann', 'rf'],
        help="Model type to train: cnn, ann, or rf"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- 1. Load Data ---
    try:
        features, labels = load_and_prepare_data(FEATURE_FILE, LABEL_FILE)
    except FileNotFoundError:
        print(f"Error: Make sure your data files exist at '{FEATURE_FILE}' and '{LABEL_FILE}'")
        return

    # --- 2. K-Fold Cross-Validation ---
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_metrics = []
    best_val_accuracy = -1
    best_model = None
    best_history = None
    best_test_x = None
    best_test_y = None

    print(f"\n--- Starting {N_SPLITS}-Fold Cross-Validation ({args.model.upper()}) ---")
    for fold, (train_index, test_index) in enumerate(kf.split(features, labels)):
        print(f"\n** Fold {fold + 1}/{N_SPLITS} **")
        # print(model)

        # Split data for this fold
        train_x, test_x = features[train_index], features[test_index]
        train_y, test_y = labels[train_index], labels[test_index]

        if args.model in ['cnn', 'ann']:
            # Build model
            if args.model == 'cnn':
                model = create_cnn_model(train_x.shape[1:])
            else:
                model = create_ann_model(train_x.shape[1:])

            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)

            # Train
            history = model.fit(
                train_x, train_y,
                epochs=EPOCHS,
                validation_data=(test_x, test_y),
                callbacks=[early_stopping],
                verbose=1
            )

            # Evaluate
            loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
            print(f"Fold Accuracy: {accuracy:.4f}")

            if accuracy > best_val_accuracy:
                best_val_accuracy = accuracy
                best_model = model
                best_history = history
                best_test_x, best_test_y = test_x, test_y
                print(f"New best model found in fold {fold + 1}")

            fold_metrics.append({'Fold': fold + 1, 'Accuracy': accuracy, 'Loss': loss})

        else:  # rf
            # Flatten patches to 2D features dynamically
            train_x_2d = train_x.reshape((train_x.shape[0], -1))
            test_x_2d = test_x.reshape((test_x.shape[0], -1))

            rf = create_rf_model()
            rf.fit(train_x_2d, train_y)
            preds = rf.predict(test_x_2d)
            probs = rf.predict_proba(test_x_2d)

            accuracy = accuracy_score(test_y, preds)
            print(f"Fold Accuracy: {accuracy:.4f}")

            if accuracy > best_val_accuracy:
                best_val_accuracy = accuracy
                best_model = rf
                best_history = None
                best_test_x, best_test_y = test_x_2d, test_y
                best_probs = probs
                print(f"New best model found in fold {fold + 1}")

            fold_metrics.append({'Fold': fold + 1, 'Accuracy': accuracy, 'Loss': np.nan})

    print("\n--- Cross-Validation Complete ---")
    metrics_df = pd.DataFrame(fold_metrics)
    print("Average Metrics Across All Folds:")
    print(metrics_df.mean(numeric_only=True))

    # --- 3. Final Evaluation and Visualization ---
    if best_model is not None:
        print("\n--- Analyzing the Best Performing Model ---")

        if args.model in ['cnn', 'ann']:
            # Save Keras model
            model_save_path = os.path.join(RESULTS_DIR, f'best_land_cover_model_{args.model}.h5')
            best_model.save(model_save_path)
            print(f"Best model saved to {model_save_path}")

            # Predictions for plots
            test_z_probs = best_model.predict(best_test_x)
            test_z_class = np.argmax(test_z_probs, axis=1)

            # Plots
            plot_training_history(best_history, os.path.join(RESULTS_DIR, f'training_history_{args.model}.png'))
            plot_confusion_matrix(best_test_y, test_z_class, LABELS_NAME, os.path.join(RESULTS_DIR, f'confusion_matrix_{args.model}.png'))
            plot_roc_curves(best_test_y, test_z_probs, LABELS_NAME, os.path.join(RESULTS_DIR, f'roc_curves_{args.model}.png'))
        else:
            # Save RF model
            model_save_path = os.path.join(RESULTS_DIR, 'best_land_cover_model_rf.joblib')
            joblib.dump(best_model, model_save_path)
            print(f"Best RF model saved to {model_save_path}")

            # We saved best_probs from the best fold; regenerate for best_test_x if needed
            if 'best_probs' not in locals():
                best_probs = best_model.predict_proba(best_test_x)
            test_z_class = np.argmax(best_probs, axis=1)

            plot_confusion_matrix(best_test_y, test_z_class, LABELS_NAME, os.path.join(RESULTS_DIR, 'confusion_matrix_rf.png'))
            plot_roc_curves(best_test_y, best_probs, LABELS_NAME, os.path.join(RESULTS_DIR, 'roc_curves_rf.png'))
    else:
        print("Training failed. No best model was found.")

if __name__ == "__main__":
    main()