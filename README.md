# Titanic-Survival-Prediction
Machine Learning model to predict Titanic passenger survival.

#  Titanic Survival Prediction  

##  Overview  
This project builds a **Machine Learning model** to predict whether a passenger survived the **Titanic disaster**.  
The dataset includes passenger details such as **age, gender, ticket class, fare, cabin information, etc.**  
We preprocess the data, train a classification model, and evaluate its performance using accuracy, precision, recall, and F1-score.  

---

##  Task Objectives  
- Develop a **classification model** for Titanic survival prediction.  
- Handle **missing values** effectively.  
- Perform **categorical encoding** and **data normalization**.  
- Train and evaluate a **Random Forest Classifier**.  
- Provide **visualizations** for model insights.  
- Save and submit model predictions.  

---

##  Project Structure  
Titanic-Survival-Prediction/
│── data/                  # Folder containing the dataset files
│   ├── train.csv          # Training dataset (if applicable)
│   ├── test.csv           # Testing dataset (if applicable)
│── notebooks/             # Jupyter Notebooks for exploratory data analysis (EDA) and model development
│   ├── Titanic_EDA.ipynb  # Notebook for data exploration and visualization
│   ├── Model_Training.ipynb  # Notebook for training and evaluation
│── models/                # Folder for saving trained models
│   ├── random_forest.pkl  # Saved Random Forest model
│── src/                   # Source code directory containing scripts
│   ├── preprocess.py      # Data preprocessing script (handling missing values, encoding, scaling)
│   ├── train.py           # Model training script
│   ├── evaluate.py        # Script for evaluating the trained model
│── submission.csv         # Model predictions stored in a CSV file
│── requirements.txt       # List of required Python libraries
│── README.md              # Project documentation
│── main.py                # Main script to run the entire pipeline


---

## Dataset Information  
- **Source**: Downloaded using Kaggle API.  
- **Target Variable**: `Survived` (1 = Survived, 0 = Not Survived).  
- **Features**:
  - `Pclass` - Ticket class (1st, 2nd, 3rd).  
  - `Sex` - Gender (Male/Female).  
  - `Age` - Age in years.  
  - `Fare` - Passenger fare.  
  - `Embarked` - Port of embarkation.  
  - `Cabin`, `SibSp`, `Parch`, etc.
 
  ---  
 Why Random Forest?
1. Handles Missing Values Efficiently
The Titanic dataset contains missing values in Age, Cabin, and Embarked columns.

Random Forest can handle missing values by averaging predictions from multiple decision trees.

2. Works Well with Categorical Data
The dataset includes categorical features like Sex (Male/Female), Pclass (1st, 2nd, 3rd), and Embarked (C/Q/S).

Random Forest can handle categorical data without needing extensive preprocessing.

3. Reduces Overfitting Compared to Decision Trees
A single Decision Tree can overfit, capturing noise instead of patterns.

Random Forest uses multiple trees and averages their predictions, reducing overfitting.

4. Handles Non-Linear Relationships
Some features like Fare vs. Survival or Age vs. Survival have non-linear relationships.

Random Forest is flexible and captures such patterns better than linear models.

5. Feature Importance Analysis
Random Forest provides a feature importance ranking, helping us understand which features influence survival.

Example: Sex and Pclass are usually the most important predictors.

6. Robust and High Accuracy
In previous Titanic Kaggle competitions, Random Forest performed better than simpler models like Logistic Regression.

It achieves an accuracy of around 80-85% on Titanic datasets.




