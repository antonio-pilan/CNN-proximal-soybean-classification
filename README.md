# Deep Learning for Soybean Classification: A Binary Classification of Healthy vs. Unhealthy Grains

**Authors:** Antonio Augusto Pilan dos Santos & William de Lima Anselmo

## Introduction

This project is an introductory exploration into the application of deep learning for the binary classification of healthy and unhealthy soybean grains, a task of considerable relevance to agricultural efficiency in Brazil, one of the world's largest soybean producers. The primary objective was to understand the practical implementation of a Convolutional Neural Network (CNN) on a public soybean image dataset, navigate the challenges of class imbalance, and evaluate model generalization.

This work is aimed at beginner students in graduate and undergraduate programs and serves as a step-by-step guide to a machine learning process, considering potential issues and building a model capable of generalizing on an imbalanced dataset.

## Problem Context

Brazil is responsible for 40% of the soybean produced worldwide, making it one of the most important agricultural commodities for animal feed, human consumption, and industrial applications. However, soybean production faces significant challenges, with losses exceeding 20% due to diseases, pests, and climate change impacts.

Efficient crop monitoring is essential, but manual monitoring is unfeasible at a large scale. This project explores the potential of computer vision and deep learning to automate the classification of healthy vs. unhealthy soybean grains from proximal images, addressing the research question: **Can deep learning effectively classify healthy vs. unhealthy soybean grains from proximal images?** 

## Dataset

The project utilizes a public dataset of proximal soybean grain images published by Lin et al. (2023). The images were captured using an industrial camera and underwent an automated segmentation process to isolate individual grains. For this project, the dataset, originally with 5 labels, was simplified into a binary classification task: healthy vs. unhealthy grains.

A key characteristic of this dataset is the significant class imbalance:
* **Healthy:** 1,201 samples (22%) 
* **Unhealthy:** 4,312 samples (78%) 

This imbalance requires the use of metrics like Precision, Recall, and F1-score for a meaningful evaluation of the model's performance.

## Methodology

### Algorithm: Convolutional Neural Network (CNN)

A Convolutional Neural Network (CNN) was chosen due to its ability to handle the non-linearity of raw image data and automate feature engineering. The model architecture is as follows:

```python
model = tf.keras.models.Sequential([
    Conv2D(128, (3,3), activation='relu', input_shape=(224,224,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # Binary classification
])
```

**Design Choices:**
* Grayscale input (single channel) was used to consider shape information, though it discards color information.
* Progressive feature map reduction with `Conv2D` layers.
* `MaxPooling2D` for translation invariance and overfitting control.
* A `sigmoid` activation function in the final layer to output a probability.

### Technology Stack

The project was implemented in Python using the following main libraries:
* **TensorFlow & Keras:** For building and training the deep learning model.
* **Scikit-learn:** For data splitting, cross-validation, and performance metrics.
* **Pandas & NumPy:** For data manipulation.
* **Matplotlib & Seaborn:** For data visualization.
* **Pillow (PIL):** For image processing.

### Training Strategy

The model was trained with the following approach:
* **Hyperparameters:**
    * **Optimizer:** Adam 
    * **Learning Rate:** 0.00001 (a conservative rate for stable learning) 
    * **Batch Size:** 128 
* **Class Imbalance Handling:**
    * **Class Weighting:** Applied during training to penalize misclassifications of the minority class (healthy) more heavily.
    * **Threshold Optimization:** An optimal decision boundary was determined using the Precision-Recall curve to maximize the F1-score.
* **Evaluation:**
    * **Data Split:** 40% for Training and 60% for Testing (due to computational constraints).
    * **5-fold Cross-Validation:** Used on the training set to get a statistically robust evaluation of the model's performance and to determine the optimal number of training epochs.
    * **Early Stopping:** To prevent overfitting by stopping the training when the validation loss stops improving.

## Results

After a 5-fold cross-validation process, the optimal training parameters were determined to be **185 epochs** with a decision threshold of **0.6696**. The final model was then trained on the entire training set with these parameters and evaluated on the unseen test data.

The final model's performance on the test data is as follows:

| Metric         | Score   |
| -------------- | ------- |
| Test Accuracy  | 83.71%  |
| Test Precision | 59.74%  |
| Test Recall    | 77.39%  |
| Test F1 Score  | 0.6743  |

The confusion matrix for the test data shows that the model has a good recall for the positive class (healthy seeds), correctly identifying a majority of them. However, the precision indicates that a significant number of unhealthy seeds are still being misclassified as healthy.

## Critical Analysis & Future Work

This project served as a successful introduction to applying deep learning for an agricultural classification task. While the model performs better than random chance, there is significant room for improvement. The current precision of 59.74% is not sufficient for a production-ready system, as it would lead to a notable amount of unhealthy grains being accepted.

Future work should focus on the following areas to improve model performance:
* **Incorporate Color Information:** Transition from grayscale to RGB images to leverage color features, which could help differentiate between healthy and immature grains.
* **Data Augmentation:** Implement techniques like rotation, flipping, and brightness adjustments to artificially increase the size and diversity of the training set.
* **Advanced Imbalance Handling:** Explore more sophisticated techniques like Focal Loss or explicit oversampling (e.g., SMOTE).
* **Hyperparameter Optimization:** Conduct a more systematic search for optimal learning rates, batch sizes, and optimizer parameters.
* **Architecture Exploration:** Experiment with more advanced and established CNN architectures.

## How to Run the Project

1.  **Clone the repository.**
2.  **Upload the `project_intro_ml.ipynb` notebook to Google Colab.**
3.  **Ensure your dataset is in your Google Drive** and update the `image_dir` path in the notebook to point to the correct location.
4.  **Run the cells sequentially.** The notebook will mount your Google Drive, load and preprocess the images, train the model using 5-fold cross-validation, and evaluate the final model.

## References
[1] Wei Lin, Youhao Fu, Peiquan Xu, Shuo Liu, Daoyi Ma, Zitian Jiang, Siyang Zang, Heyang Yao, and Qin Su. *Soybean image dataset for classification*. Elsevier, 2023. 
[2] François Chollet. *Deep Learning with Python*. Manning, second edition, 2021. 
[3] Jayme Garcia Arnal Barbedo. *Deep learning for soybean monitoring and management*. Seeds, 2023.
