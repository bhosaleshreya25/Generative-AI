# Generative AI Lab – Practice Lab Assignment 1

## Neural Network Implementation from Scratch – Fashion-MNIST Image Classification

### Student Information

| Field          | Details                 |
| -------------- | ----------------------- |
| **Name**       | Shreya Dattaram Bhosale |
| **PRN**        | 202502110010            |
| **Class**      | T.Y. B.Tech             |
| **Course**     | Generative AI Lab       |
| **Department** | CSE (AI & ML)           |
| **Batch**      | A3                      |
| **Date**       | 15/08/2026              |

---

## Objective

The objective of this practical is to understand the working of a basic **feedforward neural network** by implementing it from scratch using **Python and NumPy**.

The implementation covers:

* Forward Propagation
* Backpropagation
* ReLU Activation
* Softmax Activation
* Cross-Entropy Loss
* He Weight Initialization
* Mini-Batch Gradient Descent
* Model Evaluation
* Performance Visualization

The neural network is implemented **without using TensorFlow, Keras, or PyTorch**.

---

## Dataset

### Fashion-MNIST

Fashion-MNIST is a grayscale image classification dataset containing images of clothing and footwear.

Each image has a resolution of **28 × 28 pixels**, resulting in **784 input features** after flattening.

For this practical, a **stratified subset of 20,000 images** is used.

| Dataset Split | Images | Purpose          |
| ------------- | -----: | ---------------- |
| Training      | 16,000 | Model Training   |
| Validation    |  2,000 | Model Validation |
| Testing       |  2,000 | Final Evaluation |

### Dataset Source

**OpenML Fashion-MNIST**

Pixel values are normalized from **0–255 to 0–1** before training.

---

## Classes

| Label | Class       |
| ----: | ----------- |
|     0 | T-shirt/top |
|     1 | Trouser     |
|     2 | Pullover    |
|     3 | Dress       |
|     4 | Coat        |
|     5 | Sandal      |
|     6 | Shirt       |
|     7 | Sneaker     |
|     8 | Bag         |
|     9 | Ankle boot  |

---

## Neural Network Architecture

```text
784 Input Neurons
        │
        ▼
128 Hidden Neurons
        │
        ▼
      ReLU
        │
        ▼
10 Output Neurons
        │
        ▼
     Softmax
```

### Model Details

| Component             | Configuration               |
| --------------------- | --------------------------- |
| Input Layer           | 784 neurons                 |
| Hidden Layer          | 128 neurons                 |
| Output Layer          | 10 neurons                  |
| Hidden Activation     | ReLU                        |
| Output Activation     | Softmax                     |
| Weight Initialization | He Initialization           |
| Loss Function         | Cross-Entropy               |
| Optimizer             | Mini-Batch Gradient Descent |

---

## Training

The model is trained using **Mini-Batch Gradient Descent**.

| Parameter     |                       Value |
| ------------- | --------------------------: |
| Learning Rate |                         0.1 |
| Batch Size    |                         128 |
| Epochs        |                          15 |
| Optimizer     | Mini-Batch Gradient Descent |

For each mini-batch, the model performs the following steps:

1. Forward propagation
2. Loss calculation
3. Backpropagation
4. Gradient calculation
5. Weight and bias updates

---

## Results

The trained neural network achieved the following performance:

| Metric              |     Result |
| ------------------- | ---------: |
| Training Accuracy   | **88.33%** |
| Validation Accuracy | **85.40%** |
| Test Accuracy       | **85.60%** |
| Precision           | **85.64%** |
| Recall              | **85.60%** |
| F1-Score            | **85.57%** |

### Final Loss

* **Training Loss:** 0.3319
* **Validation Loss:** 0.4055

---

## Performance Visualization


<img width="783" height="791" alt="image" src="https://github.com/user-attachments/assets/63dbc0c0-21fd-4622-8419-b10e18ee3b0e" />
<img width="960" height="876" alt="image" src="https://github.com/user-attachments/assets/411ed8f9-590e-44a7-8934-6e6d16b940c0" />
<img width="825" height="487" alt="image" src="https://github.com/user-attachments/assets/c4e84535-992f-4c8f-8099-981194326823" />
<img width="798" height="483" alt="image" src="https://github.com/user-attachments/assets/a13bef9f-d0cd-4415-b9ee-6511edefe45d" />


## Analysis

The implemented feedforward neural network achieved a **test accuracy of 85.60%**, demonstrating good performance on unseen Fashion-MNIST images.

The training accuracy of **88.33%** and validation accuracy of **85.40%** indicate that the model successfully learned useful patterns while maintaining reasonable generalization.

Some classification errors occur between visually similar classes, particularly:

* Shirt
* T-shirt/top
* Pullover
* Coat

These classes have similar visual characteristics, making them more challenging for a basic fully connected neural network.

A **Convolutional Neural Network (CNN)** could potentially improve classification performance because CNNs are designed to learn spatial and local features from images.

This practical provides a fundamental understanding of how neural networks work internally, including forward propagation, backpropagation, loss computation, gradient descent, and parameter updates.

---

## Technologies Used

* **Python**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **Jupyter Notebook / Google Colab**

### Library Usage

| Technology                      | Purpose                                                 |
| ------------------------------- | ------------------------------------------------------- |
| Python                          | Programming Language                                    |
| NumPy                           | Neural Network implementation and numerical operations  |
| Matplotlib                      | Performance and result visualization                    |
| Scikit-learn                    | Dataset loading, data splitting, and evaluation metrics |
| Jupyter Notebook / Google Colab | Development and execution environment                   |

> **Note:** The neural network itself is implemented using **NumPy** without TensorFlow, Keras, or PyTorch.

---

## How to Run

### 1. Install Required Libraries

```bash
pip install numpy matplotlib scikit-learn jupyter
```

### 2. Start Jupyter Notebook

```bash
jupyter notebook
```

### 3. Open the Notebook

```text
Shreya_Bhosale_Generative-AI_Lab_Assignment_1.ipynb
```

### 4. Run the Notebook

Run all cells sequentially.

The Fashion-MNIST dataset will be downloaded automatically from **OpenML** when the dataset-loading section is executed.

---

## Project Structure

```text
Generative AI/
└── Assignment-1/
    ├── Shreya_Bhosale_Generative-AI_Lab_Assignment_1.ipynb
    ├── README.md
    ├── training.JPG
    ├── classification.JPG
    ├── confusion_matrix.JPG
    └── model_comparison.JPG
```

---

## Key Learning Outcomes

Through this practical, the following concepts were implemented and understood:

* Neural network architecture
* Forward propagation
* Backpropagation
* Activation functions
* ReLU
* Softmax
* Cross-Entropy Loss
* He Initialization
* Gradient Descent
* Mini-Batch Training
* Model Evaluation
* Confusion Matrix
* Classification Metrics
* Overfitting and Generalization

---

## Conclusion

A basic **feedforward neural network was successfully implemented from scratch using Python and NumPy** for Fashion-MNIST image classification.

The model achieved:

* **85.60% Test Accuracy**
* **85.57% F1-Score**
* **88.33% Training Accuracy**

The practical provided hands-on understanding of **forward propagation, backpropagation, ReLU, Softmax, cross-entropy loss, He initialization, mini-batch gradient descent, and model evaluation** without relying on built-in deep-learning frameworks.

This implementation provides a strong foundation for understanding more advanced deep-learning architectures such as **Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformer-based models**.

---

## Author

**Shreya Dattaram Bhosale**
T.Y. B.Tech – CSE (AI & ML)
Generative AI Lab

