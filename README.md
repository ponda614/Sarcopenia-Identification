# Sarcopenia Detection 

## Table of Contents
- [Project Overview](#project-overview)
- [Libraries](#libraries)
- [Data Processing](#data-processing)
- [Model Development](#model-development)
- [Results](#results)

- [Conclusion](#conclusion)

## Project Overview
This project is centered around the development of a machine learning model to detect Sarcopenia from clinical data. The main goal is to automate the process of identifying Sarcopenia, which is characterized by the loss of skeletal muscle mass and function, particularly in older adults.

## Libraries
Key libraries used in this project include:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations and handling arrays.
- **scikit-learn**: For various machine learning tasks including model training and evaluation.
- **XGBoost**: For feature selection and boosting model performance.
- **TensorFlow/Keras**: For building and training the deep learning models.
- **SMOTE**: From the imbalanced-learn library, to address the issue of class imbalance.
- **Matplotlib & Seaborn**: For data visualization and graphical representation.
- 
## Data Processing
In this project, I handled extensive patient data, performing critical tasks such as:
- Cleaning and standardizing data fields like medication names and patient statuses.
- Appropriately dealing with missing values to maintain the integrity of the data.
- Applying one-hot encoding to transform categorical data into a suitable format for machine learning models.

## Model Development
My contributions to the model development include:
- Segmenting the data into training, validation, and test sets.
- Using XGBoost for feature selection to identify key variables.
- Implementing a deep learning model with Keras, focusing on accurately predicting cases of Sarcopenia.
- Employing SMOTE for addressing class imbalance and PCA for reducing data dimensionality.

## Results
The model demonstrated strong performance metrics on the test dataset:
- **Accuracy**: 92%
- **Precision and Recall**:
  - Class 0 (No Sarcopenia): Precision - 95%, Recall - 93%
  - Class 1 (Sarcopenia): Precision - 80%, Recall - 86%
- **F1-Score**:
  - Class 0: 0.94
  - Class 1: 0.83
- **Macro and Weighted Averages**:
  - Macro Avg: Precision - 88%, Recall - 89%, F1-Score - 89%
  - Weighted Avg: Precision - 92%, Recall - 92%, F1-Score - 92%

In addition to the above metrics, the model's predictive capability was further evidenced by:
- True Positives (TP): 202
- False Positives (FP): 51
- False Negatives (FN): 34
- True Negatives (TN): 717

These results highlight the model's ability to effectively distinguish between Sarcopenia and non-Sarcopenia cases with high accuracy and balanced precision and recall.

## Conclusion
This project represents a significant step towards leveraging machine learning in the early detection of Sarcopenia. The model's potential to improve diagnosis and treatment strategies in a clinical setting is promising.
