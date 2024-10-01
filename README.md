# Deep Learning Classification Model

This project demonstrates the process of building and training a deep learning classification model using TensorFlow and Keras. The model can be applied to binary or multi-class classification tasks by adjusting the output layer. The code allows users to input their dataset and customize the features and target columns.

## Features

- **Data Preprocessing**: 
  - Handles missing values using median (for numeric columns) and mode (for categorical columns).
  - Converts categorical columns into numerical format using one-hot encoding.
  - Scales the features using `StandardScaler`.
  
- **Model Architecture**: 
  - A fully connected feedforward neural network using ReLU activation.
  - Dropout layers to reduce overfitting.
  
- **Training**: 
  - Option to customize training/validation split.
  - Plots training history (accuracy over epochs).
  
- **Evaluation**: 
  - Outputs classification metrics such as accuracy, confusion matrix, and classification report.
  - Saves the trained model for future use.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/heytanix/PyDeepModel.git
cd <your-repo-name>
pip install -r requirements.txt
