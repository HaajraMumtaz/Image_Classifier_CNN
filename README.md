# Image_Classifier_CNN

 # Handwritten Digit Recognition using CNN

A deep learning project that uses a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** to classify handwritten digits from the **MNIST dataset** with approximately **94% test accuracy**.

## Features

* CNN-based image classification
* Automatic MNIST dataset loading
* Data normalization and preprocessing
* Training and validation monitoring
* Accuracy and loss visualization
* High-performance digit recognition

## Dataset

* **Training Images:** 60,000
* **Testing Images:** 10,000
* **Classes:** 10 (Digits 0–9)
* **Image Size:** 28 × 28 pixels
* **Channels:** Grayscale (1)

## Model Architecture

```text
Input (28×28×1)
│
├── Conv2D (32, 3×3, ReLU)
├── MaxPooling2D (2×2)
├── Conv2D (64, 3×3, ReLU)
├── MaxPooling2D (2×2)
├── Flatten
├── Dense (128, ReLU)
├── Dropout (0.3)
└── Dense (10, Softmax)
```

## Training Configuration

| Parameter        | Value                           |
| ---------------- | ------------------------------- |
| Optimizer        | Adam                            |
| Loss             | Sparse Categorical Crossentropy |
| Batch Size       | 128                             |
| Epochs           | 10                              |
| Validation Split | 10%                             |

## Installation

```bash
git clone https://github.com/yourusername/MNIST-CNN.git
cd MNIST-CNN
pip install tensorflow numpy matplotlib
```

## Run

```bash
python mnist_cnn.py
```

## Results

```text
Test Accuracy : ~94%
Test Loss     : ~0.20
```

The model automatically generates:

```text
cnn_training_curves.png
```

which contains:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

## Project Structure

```text
MNIST-CNN/
│
├── mnist_cnn.py
├── cnn_training_curves.png
├── README.md
└── requirements.txt
```

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib

## Future Improvements

* Batch Normalization
* Early Stopping
* Data Augmentation
* Hyperparameter Tuning
* Model Deployment with Flask or Streamlit
* TensorFlow Lite Conversion

## Conclusion

This project demonstrates how Convolutional Neural Networks can automatically learn image features and achieve excellent performance on handwritten digit recognition tasks. Despite its simple architecture, the model reaches around **99% accuracy**, making it a strong introduction to **Deep Learning** and **Computer Vision**.
