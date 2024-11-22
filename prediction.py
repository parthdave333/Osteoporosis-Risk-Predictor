# importing required libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Load and read the dataset
df = pd.read_csv('osteoporosis.csv')

# Splitting the data into features and target
X = df.drop(['Osteoporosis', 'Id'], axis=1, errors='ignore')
y = df['Osteoporosis']

# Define which columns are numeric and which are categorical
numeric_features = ['Age']
categorical_features = ['Gender', 'Hormonal Changes', 'Family History', 'Race/Ethnicity',
                        'Body Weight', 'Calcium Intake', 'Vitamin D Intake', 'Physical Activity',
                        'Smoking', 'Alcohol Consumption', 'Medical Conditions', 'Medications',
                        'Prior Fractures']

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create a preprocessing and model pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('knn', KNeighborsClassifier(n_neighbors=3))  # KNN classifier
])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2%}')
print('Classification Report:\n', classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)

# Save the model pipeline to a .pkl file
filename = 'osteoporosis-prediction-knn-model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(pipeline, file)
