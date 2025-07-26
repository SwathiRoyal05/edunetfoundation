# Smart Irrigation System

This repository contains a Jupyter Notebook (`Smart_trrigation_updated.ipynb`) that implements a machine learning solution for smart irrigation. The system predicts the irrigation needs of multiple land parcels based on sensor data, aiming to optimize water usage.

## Table of Contents

  - [Smart Irrigation System](https://www.google.com/search?q=%23smart-irrigation-system)
      - [Table of Contents](https://www.google.com/search?q=%23table-of-contents)
      - [Project Overview](https://www.google.com/search?q=%23project-overview)
      - [Dataset](https://www.google.com/search?q=%23dataset)
      - [Features and Target Variables](https://www.google.com/search?q=%23features-and-target-variables)
      - [Data Exploration and Visualization](https://www.google.com/search?q=%23data-exploration-and-visualization)
      - [Machine Learning Model](https://www.google.com/search?q=%23machine-learning-model)
      - [Model Evaluation](https://www.google.com/search?q=%23model-evaluation)
      - [Feature Importance](https://www.google.com/search?q=%23feature-importance)
      - [Files in this Repository](https://www.google.com/search?q=%23files-in-this-repository)
      - [How to Run the Notebook](https://www.google.com/search?q=%23how-to-run-the-notebook)
      - [Dependencies](https://www.google.com/search?q=%23dependencies)
      - [Output Files](https://www.google.com/search?q=%23output-files)

## Project Overview

The goal of this project is to develop a predictive model that can determine whether three different land parcels (`parcel_0`, `parcel_1`, `parcel_2`) require irrigation based on various sensor readings. By automating and optimizing irrigation, the system can help conserve water and ensure efficient agricultural practices.

## Dataset

The project utilizes the `irrigation_machine.csv` dataset. This dataset contains sensor readings and the corresponding irrigation status for different parcels.

## Features and Target Variables

  * **Features (X)**: These are the input sensor readings, named `sensor_0` through `sensor_19`.
  * **Target Variables (y)**: These are the output variables indicating the irrigation status of three parcels: `parcel_0`, `parcel_1`, and `parcel_2`. Each parcel's status is a binary classification (e.g., irrigated or not irrigated).

## Data Exploration and Visualization

The notebook includes a comprehensive data exploration section with visualizations to understand the dataset:

  * **Distribution of Target Variables**: Bar plots show the count of irrigated vs. non-irrigated instances for each parcel.
  * **Sensor Data Distributions**: Histograms with Kernel Density Estimates (KDE) for each sensor reveal their individual distributions, ranges, and patterns.
  * **Correlation Heatmap**: A heatmap visualizes the correlation matrix between all sensor readings and parcel irrigation statuses. This helps identify relationships and dependencies between sensor data and irrigation needs.

## Machine Learning Model

  * **Preprocessing**:
      * **Feature Scaling**: `MinMaxScaler` is used to scale the sensor data. This ensures that features with larger numerical ranges do not dominate the model training process.
  * **Model Architecture**:
      * A `MultiOutputClassifier` is employed, as the problem involves predicting multiple binary target variables (one for each parcel) simultaneously.
      * The base estimator for the `MultiOutputClassifier` is a `RandomForestClassifier`. Random Forests are robust ensemble learning methods suitable for classification tasks and are known for handling complex relationships and high-dimensional data.
  * **Training**: The model is trained on a split of the data, with 80% used for training and 20% for testing.

## Model Evaluation

The model's performance is evaluated for each individual parcel using `classification_report`. This report provides key metrics such as:

  * **Precision**: The proportion of true positive predictions among all positive predictions.
  * **Recall**: The proportion of true positive predictions among all actual positives.
  * **F1-score**: The harmonic mean of precision and recall, providing a single metric that balances both.
  * **Support**: The number of actual occurrences of each class in the specified dataset.

## Feature Importance

After training, the notebook visualizes the feature importance for each parcel. This helps in understanding which sensor readings are most influential in predicting the irrigation status of `parcel_0`, `parcel_1`, and `parcel_2` individually. The top 10 most important features for each parcel are displayed.

## Files in this Repository

  * `Smart_trrigation_updated.ipynb`: The main Jupyter Notebook containing the data analysis, model training, and evaluation code.
  * `irrigation_machine.csv`: The dataset used for training and testing the model.

## Dependencies

The following Python libraries are required to run the notebook:

  * `pandas`
  * `matplotlib`
  * `seaborn`
  * `scikit-learn`
  * `joblib`
  * `numpy`

## Output Files

Upon successful execution, the notebook will save the following files:

  * `multi_output_irrigation_model.joblib`: The trained `MultiOutputClassifier` model.
  * `min_max_scaler.joblib`: The fitted `MinMaxScaler` object, necessary for scaling new data before making predictions with the saved model.
