# Osteoporosis-Risk-Predictor

## Project Overview
This project predicts the risk of osteoporosis based on patient data using a K-Nearest Neighbors (KNN) machine learning model. The tool provides a web-based interface to input health and lifestyle details, encode categorical data, and deliver a prediction of osteoporosis risk.

## Key Features
- **Risk Prediction**: Uses a trained KNN model for classifying osteoporosis risk.
- **Data Preprocessing**: Categorical data is encoded with a one-hot encoder for model compatibility.
- **Web Application**: Flask-based interface for user input and displaying prediction results.
- **Customizable**: Easy to extend with additional features or new datasets.

## Files
### Python Scripts:
- `encode_categorical_features.py`: Prepares and saves a one-hot encoder for categorical data.
- `data_training.py`: Trains the KNN model and saves it as a `.pkl` file.
- `app.py`: Flask app to interact with the user and make predictions.
- `prediction.py`: Contains the prediction pipeline and model evaluation logic.
- `one_hot_encoder_example.py`: Demonstrates the one-hot encoding process.

### Additional Files:
- `osteoporosis-prediction-knn-model.pkl`: The trained KNN model.
- `one_hot_encoder.pkl`: Saved one-hot encoder.
- `osteoporosis.csv`: Dataset used for training and testing.
- `example_input.json`: Sample input for prediction.

## Technologies Used
- **Python**: Core programming language.
- **Flask**: Web framework for building the interface.
- **Scikit-learn**: For data preprocessing and KNN model training.
- **Pandas**: Data handling and manipulation.
- **Joblib/Pickle**: For saving and loading models and encoders.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/osteoporosis-risk-predictor.git

2. Run the Flask app:
   ```bash
   python app.py

3. Open your browser and navigate to http://127.0.0.1:5000/.
4. Input patient details and view the prediction result.

## Dataset
The dataset (osteoporosis.csv) contains patient health and lifestyle data with features like age, gender, medical history, and habits. It serves as the foundation for training the KNN model.

## Results
The model predicts osteoporosis risk with a high degree of accuracy, validated through metrics like accuracy score and confusion matrix.

## Future Enhancements
Integration with additional machine learning models for comparison.
Expanding the dataset with real-world medical data.
Deploying the application on a cloud platform for wider accessibility.

## Contributing
Contributions are welcome! Feel free to fork the repository, improve the code, and submit a pull request.
