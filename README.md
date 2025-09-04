# Land Cover Mapping – Fiji 🌍

This project focuses on **land cover and land use change detection in Nadi, Fiji** (2013–2019) using satellite imagery and machine learning. Fiji is undergoing rapid urbanisation, yet most municipal[...]


## 🛠️ Reproducibility: Commands

**1. Install dependencies (example for Python):**
```bash
pip install -r requirements.txt
```


**2. Prepare the dataset:**
- Download Landsat-8 OLI imagery for Nadi, Fiji (2013–2019).
- Place raw data in the `data/` directory.


**3. Train models:**
- For Random Forest:
    ```bash
    python ".\train.py" rf
    ```
- For Artificial Neural Network:
    ```bash
    python ".\train.py" ann
    ```
- For Convolutional Neural Network:
    ```bash
    python ".\train.py" cnn
    ```


**6. Visualize change detection:**
```bash
python ".\predict.py"
```# Land Cover Mapping – Fiji 🌍

This project focuses on **land cover and land use change detection in Nadi, Fiji** (2013–2019) using satellite imagery and machine learning. Fiji is undergoing rapid urbanisation, yet most municipal[...]


## 🛠️ Reproducibility: Commands

**1. Install dependencies (example for Python):**
```bash
pip install -r requirements.txt
```


**2. Prepare the dataset:**
- Download Landsat-8 OLI imagery for Nadi, Fiji (2013–2019).
- Place raw data in the `data/` directory.


**3. Train models:**
- For Random Forest:
    ```bash
    python ".\train.py" rf
    ```
- For Artificial Neural Network:
    ```bash
    python ".\train.py" ann
    ```
- For Convolutional Neural Network:
    ```bash
    python ".\train.py" cnn
    ```


**6. Visualize change detection:**
```bash
python ".\predict.py"
```
