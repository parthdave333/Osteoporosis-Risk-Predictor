from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Load the trained model
with open('osteoporosis-prediction-knn-model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the encoder
#encoder = joblib.load('one_hot_encoder.pkl')
encoder = OneHotEncoder(handle_unknown = 'ignore')
logging.basicConfig(level=logging.DEBUG)
categorical_columns = ['Gender', 'Hormonal Changes', 'Family History', 'Race/Ethnicity', 'Body Weight', 'Calcium Intake', 'Vitamin D Intake', 'Physical Activity', 'Smoking', 'Alcohol Consumption', 'Medical Conditions', 'Medications', 'Prior Fractures']



@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['POST'])
def predict():

    data = {
    'Age': request.form['Age'],
    'Gender': request.form['Gender'],
    'Hormonal Changes': request.form['Hormonal_Changes'],
    'Family History': request.form['Family_History'],
    'Race/Ethnicity': request.form['Race_Ethnicity'],
    'Body Weight': request.form['Body_Weight'],
    'Calcium Intake': request.form['Calcium_Intake'],
    'Vitamin D Intake': request.form['Vitamin_D_Intake'],
    'Physical Activity': request.form['Physical_Activity'],
    'Smoking': request.form['Smoking'],
    'Alcohol Consumption': request.form['Alcohol_Consumption'],
    'Medical Conditions': request.form['Medical_Conditions'],
    'Medications': request.form['Medications'],
    'Prior Fractures': request.form['Prior_Fractures']
    }

    app.logger.debug(f'Received data: {data}')

    input_features = ['Age', 'Gender', 'Hormonal Changes', 'Family History', 'Race/Ethnicity',
                    'Body Weight', 'Calcium Intake', 'Vitamin D Intake', 'Physical Activity',
                    'Smoking', 'Alcohol Consumption', 'Medical Conditions', 'Medications',
                    'Prior Fractures']

    data_df = pd.DataFrame([data])
    data_df['Age'] = data_df['Age'].astype(float)  # Convert Age to float

    # Ensure the categories match the training data
    for col in categorical_columns:
        if col in data_df.columns:
            data_df[col] = data_df[col].astype(str)

    # Encode categorical features
    encoded_features = encoder.fit_transform(data_df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features.toarray(),
                            columns=encoder.get_feature_names_out(categorical_columns))

    #data_df = data_df.drop(columns=categorical_columns)
    data_df = pd.concat([data_df, encoded_df], axis=1)

    app.logger.debug(f'Encoded data: {data_df}')

    # Predict
    my_prediction = model.predict(data_df)
    result = 1 if my_prediction[0] == 1 else 0

    return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
