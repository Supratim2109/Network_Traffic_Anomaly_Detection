# Anomaly Detection in Network Traffic

## Project Overview
This project implements an anomaly detection system using **One-Class SVM** to identify unusual patterns in network traffic. The model is trained to recognize normal behavior, and deviations from these patterns are flagged as anomalies. The dataset used is `KDDTrain.csv`, a subset of the widely-used KDD Cup dataset for intrusion detection.

### Objectives:
- Build a model to detect anomalies in network traffic.
- Train the model using only normal data to differentiate anomalous behavior.
- Evaluate the model using various classification metrics.

## Requirements

### Libraries Used:
- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone (https://github.com/Supratim2109/Network_Traffic_Anomaly_Detection.git)
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then install the required libraries:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

3. **Prepare the Dataset**:
   - Place the `KDDTrain.csv` file in the same directory as the script.
   - The dataset should include the following columns:
     - Features of the network traffic.
     - A `class` column indicating normal or anomalous behavior.

4. **Run the Notebook**:
   Open the Jupyter Notebook (`Detection_model.ipynb`) and execute the cells step-by-step.
   ```bash
   jupyter notebook Detection_model.ipynb
   ```

## Steps in the Code

1. **Data Loading and Preprocessing**:
   - Load the dataset using Pandas.
   - Normalize the feature values using `StandardScaler` to bring them to a common scale.

2. **Train-Test Split**:
   - Split the data into training and testing sets.
   - Use only normal data for training.

3. **Model Training**:
   - Train a **One-Class SVM** with an RBF kernel to learn the patterns of normal network traffic.

4. **Prediction and Evaluation**:
   - Predict anomalies on the test set.
   - Convert predictions to binary format for consistency.
   - Evaluate the model using:
     - **Classification Report** (Precision, Recall, F1-Score).
     - **Confusion Matrix**.
     - **Accuracy Score**.

5. **Visualization**:
   - Plot the decision scores of the test samples.
   - Highlight the decision boundary to visualize normal vs. anomalous points.

## Results

- The model outputs a classification report summarizing its performance.
- A confusion matrix provides insights into true positives, false positives, and false negatives.
- The decision scores distribution plot aids in understanding the model's behavior and thresholds.

