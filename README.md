# Census Income Classification Project

This project implements machine learning models to predict whether an individual's income exceeds a specific threshold (binary classification) based on census data. It explores two distinct approaches: a Deep Learning model using **Multi-Layer Perceptrons (MLP)** with TensorFlow/Keras, and a probabilistic **Gaussian Naive Bayes** classifier implemented from scratch.

## 📂 Project Structure

* **`fp_mlp.py`**: The main training script for the Neural Network. It handles data preprocessing, model architecture definition, training, and visualization of metrics (Loss, Precision, Recall, F1-Score).
* **`f_project.py`**: A standalone implementation of a Gaussian Naive Bayes classifier. It calculates class probabilities manually using mean and standard deviation without relying on high-level ML libraries like Scikit-Learn.
* **`mlp_load.py`**: An inference script that reconstructs the Keras model, loads pre-trained weights (`model_weights.h5`), and runs predictions on a test dataset (`dataset_t.csv`).
* **`dataset.csv`**: The primary dataset used for training and validation.
* **`dataset_t.csv`**: A separate dataset used for testing predictions after the model is trained.

## 🧠 Methodology

### 1. Deep Learning Approach (MLP)
The project utilizes TensorFlow and Keras to build a neural network capable of handling mixed data types.

* **Data Preprocessing**:
    * **Categorical Data**: Features like `workclass`, `education`, and `marital-status` are encoded using `StringLookup` or `IntegerLookup` layers.
    * **Numerical Data**: Features like `age` and `capital-gain` are normalized using Keras `Normalization` layers to standard deviation.
* **Model Architecture**:
    * **Input**: Concatenated numerical and encoded categorical features.
    * **Hidden Layers**: Two Dense layers (64 and 16 neurons) with ReLU activation.
    * **Regularization**: Dropout layers (rate 0.4) are applied after dense layers to prevent overfitting.
    * **Output**: A single neuron with Sigmoid activation for binary probability.
* **Training Configuration**:
    * **Optimizer**: Adam (learning rate = 0.005).
    * **Loss Function**: Binary Crossentropy.
    * **Custom Metrics**: Implements Precision, Recall, and F1-Score using Keras backend operations.

### 2. Probabilistic Approach (Naive Bayes)
A custom implementation of the Naive Bayes algorithm to demonstrate understanding of probabilistic modeling.

* **Logic**: Calculates the Gaussian probability distribution function (PDF) for continuous variables.
* **Training**: Summarizes the dataset by calculating the mean and standard deviation for each feature per class.
* **Prediction**: Calculates the joint probability of a new record belonging to a specific class based on prior probability and likelihood.

## 🚀 How to Run

### Prerequisites
Install the required dependencies:
```bash
pip install tensorflow pandas numpy matplotlib
