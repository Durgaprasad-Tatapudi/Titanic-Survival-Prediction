# Import necessary libraries
import kagglehub
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Step 1: Download dataset from Kaggle using the correct path
dataset_path = kagglehub.dataset_download("brendan45774/test-file")

# Ensure the dataset path is valid
if not dataset_path:
    raise FileNotFoundError("Dataset download failed. Check your Kaggle API credentials and dataset name.")

print("Path to dataset files:", dataset_path)

# Step 2: List available files
try:
    available_files = os.listdir(dataset_path)
    print("Available files in dataset:", available_files)
except FileNotFoundError:
    raise FileNotFoundError("The dataset folder was not found. Verify the path and dataset.")

# Step 3: Identify correct dataset file
file_name = "tested.csv"  # Ensure correct file name
data_path = os.path.join(dataset_path, file_name)

# Step 4: Load the dataset
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file '{file_name}' not found in {dataset_path}")
    return pd.read_csv(file_path)

df = load_data(data_path)
print("Dataset loaded successfully")
print(df.head())

# Step 5: Identify target column
if "Survived" in df.columns:
    target_column = "Survived"
else:
    print("Warning: 'Survived' column not found. Using the last column as target")
    target_column = df.columns[-1]

# Step 6: Handle missing values
df.fillna(df.select_dtypes(include=[np.number]).median(), inplace=True)

# Step 7: Encode categorical variables
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Step 8: Split features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Step 9: Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Normalize numerical data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Step 11: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 12: Make predictions
y_pred = model.predict(X_val)

# Step 13: Evaluate model
accuracy = accuracy_score(y_val, y_pred)
print("\nValidation Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Step 14: Save model predictions
submission = pd.DataFrame({"PassengerId": df.index[:len(y_val)], "Survived": y_pred})
submission_path = os.path.join(dataset_path, "submission.csv")
submission.to_csv(submission_path, index=False)
print("Predictions saved at:", submission_path)

# Step 15: Plot class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette="coolwarm")
plt.title("Class Distribution of Survived vs Not Survived")
plt.xlabel("Survived (1) or Not (0)")
plt.ylabel("Count")
plt.show()

# Step 16: Plot feature importance
feature_importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names, palette="viridis")
plt.title("Feature Importance in RandomForest Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.show()

# Step 17: Plot confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", 
            xticklabels=["Not Survived", "Survived"], 
            yticklabels=["Not Survived", "Survived"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
