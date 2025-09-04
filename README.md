# Land Cover Mapping – Fiji 🌍

This project focuses on **land cover and land use change detection in Nadi, Fiji** (2013–2019) using satellite imagery and machine learning. Fiji is undergoing rapid urbanisation, yet most municipal councils lack technical support for urban planning. Our work applies **Convolutional Neural Networks (CNNs)** and compares them with other supervised/unsupervised learning methods to generate accurate land cover maps.

---

## ✨ Key Highlights
- **Data Source**: Landsat-8 OLI (30m resolution) satellite imagery.
- **Labels**: Custom annotated dataset created for supervised learning.
- **Algorithms Evaluated**:
  - Artificial Neural Network (ANN) → **94.86% accuracy**
  - Random Forest (RF) → **96.95% accuracy**
  - Convolutional Neural Network (CNN) → **99.05% accuracy**
  - K-Means (unsupervised, baseline)
- **Best Model**: CNN was chosen for classifying land cover types for remaining years.
- **Change Detection**: Visualisation of urban area growth and land cover transitions over time.

---

## 📊 Project Structure
