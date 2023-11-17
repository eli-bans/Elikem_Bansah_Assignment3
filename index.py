# import necessary dependencies
from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('../best_model.h5')

from sklearn.preprocessing import LabelEncoder, StandardScaler


from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_user_input(user_input):
    # Assuming user_input is a dictionary with feature names as keys and corresponding values

    # Categorical features
    categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                            'Contract', 'PaperlessBilling', 'PaymentMethod']

    # Numerical features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Extract numerical values
    numerical_values = [user_input[feature] for feature in numerical_features]

    # Scale numerical features
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform([numerical_values])

    # Label encode categorical features
    label_encoder = LabelEncoder()
    categorical_encoded = [label_encoder.fit_transform([user_input[feature]])[0] for feature in categorical_features]

    # Concatenate encoded categorical features and scaled numerical features
    preprocessed_input = np.concatenate([categorical_encoded, numerical_scaled.flatten()], axis=0)

    # Reshape to (1, total_features) to match the expected input shape
    preprocessed_input = preprocessed_input.reshape(1, -1)

    return preprocessed_input




@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_text_input = request.form['user_input']
        user_input = preprocess_user_input(user_text_input)
        prediction = model.predict(user_input)

        # Assuming 'prediction' is a numerical value between 0 and 1
        # Convert it to a percentage for display in the template
        prediction_percentage = round(prediction * 100, 2)

        return render_template('index.html', prediction_result=prediction_percentage)

    # Render the template without prediction result for the initial visit
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        gender = request.form['gender']
        senior_citizen = request.form['senior_citizen']
        Partner = request.form['Partner']
        Dependents = request.form['Dependents']
        PhoneService = request.form['PhoneService']
        MultipleLines = request.form['MultipleLines']
        InternetService = request.form['InternetService']
        OnlineSecurity = request.form['OnlineSecurity']
        OnlineBackup = request.form['OnlineBackup']
        DeviceProtection = request.form['DeviceProtection']
        TechSupport = request.form['TechSupport']
        StreamingTV = request.form['StreamingTV']
        StreamingMovies = request.form['StreamingMovies']
        Contract = request.form['Contract']
        PaperlessBilling = request.form['PaperlessBilling']
        PaymentMethod = request.form['PaymentMethod']
        tenure = request.form['tenure']
        monthly_charges = request.form['monthly_charges']
        total_charges = request.form['total_charges']

        # Preprocess user input
        user_input = preprocess_user_input({
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        })

        # Make predictions using the model
        prediction = model.predict(user_input)

        # Redirect to the homepage with the prediction result as a query parameter
        return redirect(url_for('home', prediction_result=float(prediction[0])))

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)