Generative-AI
Practice Lab Assignment 1 - Neural Network Implementation from Scratch
Fashion-MNIST Image Classification
Student Information
Name: Shreya Dattaram Bhosale
PRN: 202502110010
Class: T.Y.B.Tech
Course: Generative AI Lab
Department: CSE (AI & ML)
Batch: A3
Date: 15/08/2026
Objective
The objective of this practical is to understand the working of a basic feedforward neural network by implementing it from scratch using Python and NumPy.

The implementation covers:

Forward propagation
Backpropagation
ReLU and Softmax activation
Cross-Entropy Loss
He Initialization
Mini-Batch Gradient Descent
Model evaluation
The neural network is implemented without using TensorFlow, Keras or PyTorch.

Dataset
Fashion-MNIST
Fashion-MNIST is a grayscale image classification dataset containing images of clothing and footwear.

Each image is 28 × 28 pixels, resulting in 784 input features after flattening.

A stratified subset of 20,000 images is used for this practical.

Data	Images	Purpose
Training	16,000	Model training
Validation	2,000	Model validation
Testing	2,000	Final evaluation
Classes
Label	Class
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
Dataset Source: OpenML Fashion-MNIST

Pixel values are normalized from 0–255 to 0–1 before training.

Neural Network Architecture
784 Input Neurons
        ↓
128 Hidden Neurons
        ↓
      ReLU
        ↓
10 Output Neurons
        ↓
     Softmax
Model Details
Input Layer: 784 neurons
Hidden Layer: 128 neurons
Output Layer: 10 neurons
Hidden Activation: ReLU
Output Activation: Softmax
Weight Initialization: He Initialization
Loss Function: Cross-Entropy
Training

The model is trained using Mini-Batch Gradient Descent.

Parameter	Value
Learning Rate	0.1
Batch Size	128
Epochs	15
Optimizer	Mini-Batch Gradient Descent

For each batch, the model performs forward propagation, calculates the loss, performs backpropagation and updates the weights and biases.

Results
Metric	Result
Training Accuracy	88.33%
Validation Accuracy	85.40%
Test Accuracy	85.60%
Precision	85.64%
Recall	85.60%
F1-Score	85.57%

Final training loss: 0.3319

Final validation loss: 0.4055

Performance Visualization
Training Performance

Classification Report

Confusion Matrix

Model Comparison

Analysis

The model achieved 85.60% test accuracy, showing good performance on unseen Fashion-MNIST images.

The training accuracy of 88.33% and validation accuracy of 85.40% indicate that the model learned useful patterns while maintaining reasonable generalization.

Some errors occur between visually similar classes such as Shirt, T-shirt/top, Pullover and Coat.

A feedforward neural network is useful for understanding the fundamentals of neural networks. A CNN could further improve image classification by learning spatial features from images.

Technologies Used
Python
NumPy
Matplotlib
Scikit-learn
Jupyter Notebook / Google Colab

The neural network is implemented using NumPy. Scikit-learn is used for dataset loading, data splitting and evaluation metrics.

How to Run

Install the required libraries:

pip install numpy matplotlib scikit-learn jupyter

Start Jupyter Notebook:

jupyter notebook

Open:

Practice_Lab_1_MNIST_From_Scratch.ipynb

Run the cells sequentially. The Fashion-MNIST dataset will be downloaded automatically from OpenML.

Project Structure
Generative AI/
├── Assignment-1/
   ├── Shreya_Bhosale_Generative-AI_Lab_Assignment_1.ipynb
   ├── README.md
   ├── training.JPG
   ├── classification.JPG
   ├── confusion_matrix.JPG
   └── model_comparison.JPG


A feedforward neural network was successfully implemented from scratch using Python and NumPy for Fashion-MNIST classification.
The model achieved 85.60% test accuracy with an F1-score of 85.57%.
This practical provided an understanding of forward propagation, backpropagation, loss calculation and gradient descent without using built-in deep-learning frameworks.

Visualizations:

<img width="783" height="791" alt="image" src="https://github.com/user-attachments/assets/63dbc0c0-21fd-4622-8419-b10e18ee3b0e" />
<img width="960" height="876" alt="image" src="https://github.com/user-attachments/assets/411ed8f9-590e-44a7-8934-6e6d16b940c0" />
<img width="825" height="487" alt="image" src="https://github.com/user-attachments/assets/c4e84535-992f-4c8f-8099-981194326823" />
<img width="798" height="483" alt="image" src="https://github.com/user-attachments/assets/a13bef9f-d0cd-4415-b9ee-6511edefe45d" />
