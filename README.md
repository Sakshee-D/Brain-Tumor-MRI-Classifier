# Brain Tumor MRI Classification App

A Streamlit-based web application for classifying brain tumors from MRI scans using a Convolutional Neural Network (CNN). The application enables model training, evaluation, and real-time prediction through an interactive web interface.

> Student project developed to explore deep learning techniques for medical image classification.

---

## Features

- MRI brain tumor classification using CNNs
- Interactive Streamlit web application
- Train models directly from uploaded datasets
- Automatic class detection from dataset folders
- Save and reload trained models
- Real-time image prediction
- Training history visualization and performance tracking

---

## Supported Classes

The model classifies MRI scans into the following categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Matplotlib
- Pillow

---

## Project Structure

```text
BrainTumorMRI/
│
├── app.py
├── requirements.txt
│
├── dataset/
│   ├── Training/
│   └── Testing/
│
├── models/
│   ├── tumor_cnn.keras
│   ├── class_names.json
│   └── training_history.json
│
└── pages/
    ├── Train.py
    └── Predict.py
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Dataset Format

The application expects the following folder structure:

```text
Training/
│
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

Testing/
│
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

You can either:

- Upload the original dataset ZIP files through the Train page, or
- Place the dataset folders inside the `dataset/` directory.

---

## Model Training

Navigate to the **Train** page and:

1. Upload the training and testing datasets, or
2. Use datasets stored locally in the project directory.

During training, the application:

- Preprocesses MRI images
- Trains a CNN-based classification model
- Evaluates performance on the testing dataset
- Saves trained model artifacts automatically

---

## Saved Outputs

After successful training, the following files are generated:

```text
models/
├── tumor_cnn.keras
├── class_names.json
└── training_history.json
```

### File Description

| File | Purpose |
|--------|---------|
| tumor_cnn.keras | Trained model weights and architecture |
| class_names.json | Mapping of output classes |
| training_history.json | Training and validation metrics |

---

## Prediction

Upload an MRI scan through the prediction interface to:

- Predict the tumor category
- View confidence scores
- Generate predictions in real time

---

## Future Improvements

- Transfer learning using EfficientNet and ResNet
- Model explainability with Grad-CAM
- Advanced medical image preprocessing
- Cloud deployment enhancements
- Improved visualization of prediction confidence

---

## Disclaimer

This project was developed for educational purposes.