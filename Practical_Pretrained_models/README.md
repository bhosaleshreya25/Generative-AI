Sure. Here is the **complete edited README content** ready to copy-paste into your GitHub `README.md`.

````markdown
# Generative AI Lab – Pretrained Models Practical

## Bird Species Image Classification Using Deep Learning

### Comparative Analysis of Custom CNN, MobileNetV2 and ResNet50

### Student Information

| Field | Details |
|---|---|
| **Students** | Shreya Bhosale – 202502110010 |
|              | Sakshi Walunj – 202502110007 |
|              | Dhanashri Shah – 202502110008 |
| **Class** | T.Y. B.Tech |
| **Course** | Generative AI Lab |
| **Department** | CSE (AI & ML) |
| **Dataset** | CUB-200-2011 |

---

## Aim

The aim of this practical is to study fine-grained bird species image classification using deep learning.

A Custom CNN is implemented from scratch as a baseline model. Two pretrained models, **MobileNetV2 and ResNet50**, are then implemented using ImageNet pretrained weights.

The practical focuses on transfer learning, fine-tuning, model evaluation and comparison of the three approaches.

The supplied research paper is used as a reference for understanding transfer learning and fine-grained bird species classification. The paper evaluates MobileNet and InceptionV3, while ResNet50 is added to this practical as an additional pretrained comparison model.

---

## Objective

The objectives of this practical are:

* Study the supplied research paper on bird species classification.
* Download and prepare the CUB-200-2011 dataset.
* Understand the challenges of fine-grained image classification.
* Implement a Custom CNN from scratch.
* Implement MobileNetV2 using ImageNet pretrained weights.
* Implement ResNet50 using ImageNet pretrained weights.
* Apply transfer learning to the pretrained models.
* Fine-tune the final layers using a smaller learning rate.
* Visualize training and validation performance.
* Visualize intermediate feature maps.
* Study the effect of learning rate.
* Evaluate the three models using multiple metrics.
* Generate confusion matrices.
* Generate sample predictions.
* Compare the MobileNetV2 result with the supplied research paper.
* Analyze computational requirements and possible improvements.

---

## Dataset

### CUB-200-2011

The project uses the **CUB-200-2011 (Caltech-UCSD Birds-200-2011)** dataset for fine-grained bird species classification.

The dataset contains images belonging to **200 bird species**.

| Dataset Property | Value |
|---|---:|
| Total Images | 11,788 |
| Number of Classes | 200 |
| Official Training Images | 5,994 |
| Official Test Images | 5,794 |
| Input Size | 224 × 224 × 3 |
| Image Type | RGB |

### Dataset Split

The official CUB training and test partition is preserved.

The official training set is further divided into:

| Dataset Split | Images | Purpose |
|---|---:|---|
| Training | 5,094 | Model Training |
| Validation | 900 | Model Validation |
| Testing | 5,794 | Final Evaluation |

The official test set is kept completely separate from training and validation.

### Dataset Source

**CUB-200-2011 Dataset – Kaggle**

https://www.kaggle.com/datasets/wenewone/cub2002011

---

## Research Paper

### Selected Research Paper

**Title:** Enhanced Bird Species Image Recognition and Classification using MobileNet and InceptionV3 Transfer learning Architectures

**Authors:** Sakthi Priya G., Vignesh Saravanan K., and Dheetchana K.

**Journal:** Electronic Letters on Computer Vision and Image Analysis

**Volume:** 24, Issue 1

**Pages:** 118–133

**Year:** 2025

**DOI:** 10.5565/rev/elcvia.2020

### Research Paper Links

**Official Journal Page:**  
https://elcvia.cvc.uab.cat/article/view/2020

**DOI:**  
https://doi.org/10.5565/rev/elcvia.2020

**Official PDF:**  
https://ddd.uab.cat/pub/elcvia/elcvia_a2025v24n1/elcvia_a2025v24n1p118.pdf

The research paper studies fine-grained bird species classification using the CUB-200-2011 dataset and focuses on MobileNet and InceptionV3 transfer learning architectures.

The paper reports:

| Metric | Published MobileNet Result |
|---|---:|
| Accuracy | 74.60% |
| Loss | 0.8685 |

These values are used only as reference values from the paper. They are not treated as the results of this implementation.

---

## Neural Network Models

Three models are implemented in this practical.

### 1. Custom CNN

A CNN is developed from scratch and used as the baseline model.

The model learns image features directly from the CUB-200-2011 dataset without using pretrained ImageNet weights.

The Custom CNN is used to establish a baseline against which the pretrained models can be compared.

---

### 2. MobileNetV2

MobileNetV2 is used as the first pretrained model.

The model uses **ImageNet pretrained weights**.

The pretrained feature extractor is initially frozen and a new classification head is trained for the 200 bird species.

After the initial training stage, selected final layers are unfrozen and fine-tuned using a smaller learning rate.

MobileNetV2 provides a relatively lightweight architecture and is useful for studying the trade-off between classification performance and computational requirements.

---

### 3. ResNet50

ResNet50 is used as the second pretrained model.

The model uses **ImageNet pretrained weights**.

The pretrained backbone is initially frozen while the new classification head is trained.

The final layers are then fine-tuned using a smaller learning rate.

ResNet50 provides a deeper architecture for comparison with MobileNetV2.

> **Note:** ResNet50 is an additional model implemented in this practical. The supplied research paper compares MobileNet and InceptionV3, not ResNet50.

---

## Model Comparison

```text
                    CUB-200-2011
                          |
                          v
                Image Preprocessing
                          |
                          v
                  Data Augmentation
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
      Custom CNN      MobileNetV2      ResNet50
      From Scratch     Pretrained      Pretrained
          |               |               |
          |               v               v
          |        Transfer Learning
          |               |               |
          |               v               v
          |          Fine-Tuning
          |               |               |
          +---------------+---------------+
                          |
                          v
                  Model Evaluation
                          |
                          v
                  Model Comparison
````

---

## Image Preprocessing

All input images are resized to:

```text
224 × 224 × 3
```

The training data is augmented to provide variation during training.

The augmentation methods include:

* Rotation
* Horizontal flipping
* Scaling/zooming
* Translation

ImageNet-specific preprocessing is applied to the pretrained architectures.

---

## Transfer Learning

Transfer learning is applied to MobileNetV2 and ResNet50.

The training process is divided into two stages.

### Stage 1 – Feature Extraction

The pretrained backbone is frozen.

Only the newly added classification head is trained for the bird classification task.

### Stage 2 – Fine-Tuning

The final layers of the pretrained network are unfrozen.

A smaller learning rate is used during fine-tuning.

Batch Normalization layers are kept frozen during fine-tuning to improve training stability.

---

## Training Configuration

| Parameter                  | Custom CNN | MobileNetV2 | ResNet50  |
| -------------------------- | ---------- | ----------- | --------- |
| Input Size                 | 224 × 224  | 224 × 224   | 224 × 224 |
| Batch Size                 | 16         | 16          | 16        |
| Optimizer                  | Adam       | Adam        | Adam      |
| Head Learning Rate         | 3e-4       | 1e-4        | 1e-4      |
| Fine-Tuning Learning Rate  | —          | 1e-5        | 1e-5      |
| Maximum Head Epochs        | 30         | 30          | 30        |
| Maximum Fine-Tuning Epochs | —          | 15          | 15        |
| Pretrained Weights         | No         | ImageNet    | ImageNet  |
| Data Augmentation          | Yes        | Yes         | Yes       |
| Early Stopping             | Yes        | Yes         | Yes       |
| Regularization             | Dropout    | Dropout     | Dropout   |

---

## Results

The model results obtained from the current execution are:

### Test Performance

| Model       | Accuracy | Precision | Recall | F1-Score |
| ----------- | -------: | --------: | -----: | -------: |
| Custom CNN  |   12.89% |    13.37% | 12.89% |   10.99% |
| MobileNetV2 |   57.58% |    60.19% | 57.58% |   57.19% |
| ResNet50    |    1.85% |     1.04% |  1.85% |    0.82% |

### Test Loss and Computational Comparison

| Metric           | Custom CNN | MobileNetV2 |   ResNet50 |
| ---------------- | ---------: | ----------: | ---------: |
| Test Accuracy    |     12.89% |      57.58% |      1.85% |
| Test Loss        |     3.7995 |      1.5031 |     5.1883 |
| Training Time    |  26.67 min |   19.92 min |  15.14 min |
| Total Parameters |  1,293,288 |   2,637,320 | 24,163,660 |

The results above are based on the current execution of the notebook.

---

## Performance Visualization

The notebook includes visual analysis of the trained models, including:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss
* Model comparison
* Feature maps
* Confusion matrices
* Sample predictions

Recommended screenshots for the GitHub README are:

1. Dataset verification
2. Model architecture
3. Training and validation performance
4. Final model comparison
5. Sample predictions or confusion matrix

---

## Confusion Matrix

A confusion matrix is generated for each model to analyze classification performance across the 200 bird species.

Since the dataset contains 200 classes, the confusion matrix represents the classification relationship between all 200 species.

The confusion matrices help identify classes that are frequently confused with each other.

---

## Feature Map Visualization

Intermediate feature maps are visualized to understand the features learned by the models.

The visualization provides an internal view of how image information is represented as it passes through the network.

This helps in understanding the difference between features learned by the Custom CNN and representations obtained from pretrained architectures.

---

## Sample Predictions

Sample predictions are generated using the test dataset.

The predictions can be used to compare:

* Actual bird species
* Predicted bird species
* Model confidence

This provides a direct view of how the trained models behave on unseen images.

---

## Hyperparameter Experiment

Learning rate is selected as an important hyperparameter for experimentation.

The notebook evaluates the effect of changing the learning rate and compares the resulting validation performance.

This experiment helps demonstrate how the learning rate affects model training and convergence.

---

## Analysis

The Custom CNN provides a baseline because its features are learned from scratch.

MobileNetV2 performs substantially better than the Custom CNN in the current execution and provides a useful balance between performance and model size.

ResNet50 has significantly more parameters than the other two models. In the current execution, however, its classification performance is considerably lower than MobileNetV2.

The results show that a pretrained model does not automatically guarantee better performance. Model preprocessing, training configuration, fine-tuning and architecture-specific requirements all affect the final result.

The comparison also considers training time and parameter count rather than focusing only on accuracy.

---

## Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **KaggleHub**
* **Google Colab**

### Library Usage

| Technology         | Purpose                                  |
| ------------------ | ---------------------------------------- |
| Python             | Programming language                     |
| TensorFlow / Keras | Deep learning model implementation       |
| NumPy              | Numerical operations                     |
| Pandas             | Data processing and result comparison    |
| Scikit-learn       | Dataset splitting and evaluation metrics |
| Matplotlib         | Performance visualization                |
| Seaborn            | Confusion matrix visualization           |
| KaggleHub          | Dataset download                         |
| Google Colab       | Development and execution environment    |

---

## How to Run

### 1. Install Required Libraries

```bash
pip install -U kagglehub seaborn scikit-learn
```

### 2. Open the Notebook

Open:

```text
Bird_Species_Pretrained_Models_Final(1).ipynb
```

using Google Colab.

### 3. Enable GPU

In Google Colab:

```text
Runtime → Change runtime type → GPU
```

A GPU runtime is recommended for training the models.

### 4. Run the Notebook

Run the cells sequentially.

The notebook will:

1. Set up the environment.
2. Download the CUB-200-2011 dataset.
3. Locate and verify the dataset metadata.
4. Create the training, validation and test sets.
5. Apply image preprocessing and augmentation.
6. Train the Custom CNN.
7. Train MobileNetV2.
8. Fine-tune MobileNetV2.
9. Train ResNet50.
10. Fine-tune ResNet50.
11. Evaluate all three models.
12. Generate confusion matrices.
13. Generate sample predictions.
14. Compare model performance.
15. Compare the MobileNetV2 result with the supplied research paper.

The notebook uses KaggleHub for dataset download.

---

## Project Structure

```text
Bird-Species-Classification/
│
├── Bird_Species_Pretrained_Models_Final(1).ipynb
├── README.md
│
├── research_paper/
│   └── research_paper.pdf
│
├── results/
│   ├── model_comparison.csv
│   ├── confusion_matrices/
│   └── figures/
│
└── requirements.txt
```

The CUB-200-2011 dataset is not included in the repository. It can be downloaded from the Kaggle source provided above.

---

## Key Learning Outcomes

Through this practical, the following concepts were implemented and studied:

* Fine-grained image classification
* CNN-based image classification
* Image preprocessing
* Data augmentation
* Transfer learning
* Fine-tuning
* ImageNet pretrained models
* MobileNetV2
* ResNet50
* Accuracy
* Precision
* Recall
* F1-Score
* Classification loss
* Confusion matrix
* Feature-map visualization
* Hyperparameter tuning
* Model comparison
* Training-time analysis
* Parameter-count comparison

---

## Limitations

The current implementation has some limitations:

* The CUB-200-2011 dataset contains visually similar bird species.
* Similar colours, shapes and poses can make classification difficult.
* Background and image quality can affect predictions.
* ImageNet pretrained features are not specifically learned for bird species.
* ResNet50 has a considerably larger parameter count than MobileNetV2.
* Model performance depends on training configuration and fine-tuning.
* The current ResNet50 execution gives lower classification performance than MobileNetV2.

---

## Future Scope

The project can be extended in the following ways:

* Apply Grad-CAM for model explainability.
* Perform more extensive hyperparameter optimization.
* Compare additional pretrained architectures.
* Use ensemble methods with multiple models.
* Test the models on real-world bird photographs.
* Convert the best model to TensorFlow Lite.
* Deploy the model on an edge device.
* Compare CPU and GPU inference time.
* Analyze memory consumption during inference.
* Evaluate cross-dataset generalization.

---

## Conclusion

A complete fine-grained bird species image-classification pipeline was implemented using the CUB-200-2011 dataset.

A Custom CNN was trained from scratch as the baseline, while MobileNetV2 and ResNet50 were implemented using ImageNet pretrained weights followed by transfer learning and fine-tuning.

The models were evaluated using accuracy, precision, recall, F1-score and test loss. Additional analysis included confusion matrices, feature-map visualization, sample predictions, learning-rate experimentation, training time and parameter comparison.

In the current execution, MobileNetV2 achieved the highest test accuracy among the three implemented models.

The practical provided hands-on experience with CNNs, transfer learning, pretrained models, fine-tuning and evaluation of fine-grained image classification systems.

---

## References

1. Sakthi Priya G., Vignesh Saravanan K., and Dheetchana K., "Enhanced Bird Species Image Recognition and Classification using MobileNet and InceptionV3 Transfer learning Architectures," *Electronic Letters on Computer Vision and Image Analysis*, vol. 24, no. 1, pp. 118–133, 2025.

2. CUB-200-2011 Dataset:
   [https://www.kaggle.com/datasets/wenewone/cub2002011](https://www.kaggle.com/datasets/wenewone/cub2002011)

3. Research Paper – Official Journal Page:
   [https://elcvia.cvc.uab.cat/article/view/2020](https://elcvia.cvc.uab.cat/article/view/2020)

4. Research Paper – DOI:
   [https://doi.org/10.5565/rev/elcvia.2020](https://doi.org/10.5565/rev/elcvia.2020)

5. Research Paper – Official PDF:
   [https://ddd.uab.cat/pub/elcvia/elcvia_a2025v24n1/elcvia_a2025v24n1p118.pdf](https://ddd.uab.cat/pub/elcvia/elcvia_a2025v24n1/elcvia_a2025v24n1p118.pdf)

6. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016.

7. A. G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," 2017.

---

## Author

**Shreya Dattaram Bhosale**
PRN: 202502110010

**Sakshi Walunj**
PRN: 202502110007

**Dhanashri Shah**
PRN: 202502110008

T.Y. B.Tech – CSE (AI & ML)

Generative AI Lab

