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



