# Lab 5: Decision Tree Classifier & Model Evaluation

## Problem Statement
Implement a Machine Learning classification model using a Decision Tree. The objective is to train the model on a dataset, extract the logical rules it learns, and evaluate its predictive performance using standard classification metrics.

## Approach & Theoretical Concepts
A **Decision Tree Classifier** is a supervised machine learning algorithm that recursively splits the data into subsets based on the most significant attribute at each node.

1. **Splitting Criterion:** We use **Gini Impurity** to measure the quality of a split. The algorithm selects the feature that best separates the classes (minimizes impurity).
2. **Train/Test Split:** The dataset is divided into a training set (to teach the model) and a testing set (unseen data to evaluate real-world performance).

### Model Evaluation Metrics
* **Accuracy:** The ratio of correctly predicted observations to the total observations.
* **Confusion Matrix:** A table showing True Positives, True Negatives, False Positives, and False Negatives.
* **Precision:** How many of the positively classified samples were actually positive.
* **Recall (Sensitivity):** How many of the actual positive samples were correctly classified.
* **F1-Score:** The harmonic mean of Precision and Recall.

## Expected Output
```text
--- 1. Loading Dataset ---
Features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Classes: ['setosa', 'versicolor', 'virginica']
Dataset shape: (150, 4)

--- 4. Learned Tree Rules ---
|--- petal length (cm) <= 2.45
|   |--- class: 0
|--- petal length (cm) >  2.45
|   |--- petal length (cm) <= 4.75
|   |   |--- petal width (cm) <= 1.65
|   |   |   |--- class: 1
...

--- 5. Model Evaluation ---
Accuracy Score: 100.00%