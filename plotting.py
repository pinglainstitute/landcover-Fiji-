import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

def plot_training_history(history, save_path):
    """Plots and saves the training & validation accuracy and loss."""
    pd.DataFrame(history.history).plot(figsize=(10, 6))
    plt.grid(True)
    plt.gca().set_ylim(0, 1.2) # Set a consistent y-axis range
    plt.title("Model Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.show()

def plot_roc_curves(y_true, y_pred_probs, class_names, save_path):
    """Plots and saves the ROC curves for each class."""
    n_classes = len(class_names)
    y_true_dummies = pd.get_dummies(y_true, drop_first=False).values

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_dummies[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    
    macro_roc_auc = roc_auc_score(y_true, y_pred_probs, average='macro', multi_class='ovr')
    print(f"Macro-average ROC AUC score: {macro_roc_auc:.4f}")

def plot_Classified_img(predicted_labels, extent, n_class_pos, save_path):
    """Plots and saves the Classified image."""

    features_chips_prediction_arr_Nadi_2013 = predicted_labels.reshape((780, 818))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.yticks(rotation=90, va='center')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cmap = mpl.colors.ListedColormap(['red','lightgreen','darkgreen','yellow','blue','lightyellow','gray'])
    image = plt.imshow(features_chips_prediction_arr_Nadi_2013, cmap=cmap, extent=extent, interpolation='nearest')
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size='5%', pad=0.5, pack_start=True)
    fig.add_axes(cax)
    cbar = plt.colorbar(image, orientation='horizontal', label='Alteration Type', cax=cax)
    tick_locs = (np.arange(n_class_pos) +0.5)*(n_class_pos-1)/n_class_pos
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(['Urban','Crop/Grass','Trees','Bare','Water','Coastal','Alluvium'])
    plt.grid(False)
    plt.savefig(save_path)
