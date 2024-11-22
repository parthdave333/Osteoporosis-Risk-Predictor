from sklearn.preprocessing import OneHotEncoder
import joblib
import pandas as pd

# Load your training data
X_train = pd.read_csv('osteoporosis.csv')

categorical_columns = ['Gender', 'Hormonal Changes', 'Family History', 'Race/Ethnicity', 'Body Weight', 'Calcium Intake', 'Vitamin D Intake', 'Physical Activity', 'Smoking', 'Alcohol Consumption', 'Medical Conditions', 'Medications', 'Prior Fractures']

# Initialize OneHotEncoder with 'handle_unknown' parameter
encoder = OneHotEncoder(handle_unknown='ignore')

# Fit the encoder on your training data
encoder.fit(X_train[categorical_columns])

# Save the encoder
joblib.dump(encoder, 'one_hot_encoder.pkl')