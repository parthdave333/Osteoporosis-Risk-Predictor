from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Sample data
data = {
    "Age": 50,
    "Gender": "Female",
    "Hormonal Changes": "Yes",
    "Family History": "Yes",
    "Race/Ethnicity": "White",
    "Body Weight": "Normal",
    "Calcium Intake": "High",
    "Vitamin D Intake": "Low",
    "Physical Activity": "Moderate",
    "Smoking": "No",
    "Alcohol Consumption": "Occasionally",
    "Medical Conditions": "No",
    "Medications": "No",
    "Prior Fractures": "No"
}

# Convert the dictionary to a DataFrame
data_df = pd.DataFrame([data])

# Assuming the columns are:
categorical_columns = ['Gender', 'Hormonal Changes', 'Family History', 'Race/Ethnicity', 'Body Weight', 'Calcium Intake', 'Vitamin D Intake', 'Physical Activity', 'Smoking', 'Alcohol Consumption', 'Medical Conditions', 'Medications', 'Prior Fractures']

# Initialize OneHotEncoder with 'handle_unknown' parameter
encoder = OneHotEncoder(handle_unknown='ignore')

# Fit the encoder on your training data
# Assuming X_train is your training DataFrame
# encoder.fit(X_train[categorical_columns])

# Transform the test data
try:
    transformed_data = encoder.fit_transform(data_df[categorical_columns])
    print(transformed_data)  # Print the transformed data
    print(transformed_data.toarray())  # Print the transformed data
except Exception as e:
    print(f'Error: {e}')
