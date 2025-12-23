# Fashion-MNIST Image Classification using Deep Convolutional Neural Networks

## Project Overview
This repository presents a **research-grade deep learning implementation** for **image classification on the Fashion-MNIST dataset** using a **custom-designed Convolutional Neural Network (CNN)**. The project demonstrates end-to-end model development, from data loading and visualization to network design, training, evaluation, and prediction. The architecture and methodology are suitable for **academic portfolios, MSc-level coursework, and applied deep learning research**.

---

## Dataset Description
- **Dataset:** Fashion-MNIST
- **Source:** Keras built-in datasets
- **Classes:** 10 clothing categories (e.g., T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Image size:** 28 × 28 pixels
- **Color format:** Grayscale
- **Training samples:** 60,000
- **Test samples:** 10,000
- **Labels:** Integer-encoded (0–9)

---

## Model Architecture
The network is a **deep CNN inspired by VGG-style architectures**, emphasizing hierarchical feature extraction.

### Convolutional Feature Extractor
- Initial wide-kernel convolution for low-level feature capture
- Progressive depth increase: 64 → 128 → 256 filters
- ReLU activation for non-linearity
- SAME padding to preserve spatial resolution
- MaxPooling layers for spatial downsampling and translation invariance

### Fully Connected Classifier
- Flattening layer to convert feature maps into vectors
- Dense layers for high-level representation learning
- Dropout regularization (50%) to prevent overfitting
- Softmax output layer for multi-class classification

---

## Training Configuration
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Evaluation Metric:** Accuracy
- **Batching:** Implicit via Keras
- **Epochs:** Up to 10 (Early Stopping enabled)
- **Validation Split:** 20% of training data
- **Early Stopping:** Monitors training accuracy to prevent overfitting

---

## Model Visualization
- Architecture diagram generated using `keras.utils.plot_model`
- Training curves plotted for:
  - Accuracy
  - Loss
- These visualizations support **model interpretability and learning diagnostics**

---

## Evaluation and Results
- Model evaluated on unseen Fashion-MNIST test data
- Predictions generated for sample test images
- Class probabilities computed using Softmax
- Final predictions obtained via argmax over class probabilities

---

## Key Learning Outcomes
- Practical implementation of deep CNNs using TensorFlow/Keras
- Understanding of convolutional feature hierarchies
- Effective use of dropout and early stopping for regularization
- Visualization-driven model debugging and performance analysis

---

## Technologies Used
- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib

---

## Research and Academic Relevance
This project reflects **standard deep learning practices used in computer vision research**, making it appropriate for:
- Graduate-level AI / ML coursework
- Portfolio submission for AI-focused programs
- Foundations for extending into transfer learning or real-world image datasets

---

## Future Extensions
- Batch normalization for faster convergence
- Data augmentation for robustness
- Transfer learning using pretrained CNN backbones
- Confusion matrix and class-wise performance analysis

---

## Author Notes
This repository is intended as a **clean, reproducible, and academically presentable deep learning experiment**, emphasizing clarity, correctness, and extensibility.
