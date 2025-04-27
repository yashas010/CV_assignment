import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

UPLOAD_FOLDER = 'static/uploads'

app = Flask(__name__)
app.secret_key = 'super secret key'  # Required for flash messages

# Load and prepare the Iris dataset
def prepare_model():
    # Load the Iris dataset from scikit-learn
    from sklearn.datasets import load_iris
    from sklearn.metrics import classification_report
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Generate classification report
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, target_names=['Setosa', 'Versicolor', 'Virginica'])
    
    # Save the model and scaler
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    return model.score(X_test_scaled, y_test), report

# Load or train the model at startup
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    accuracy, report = prepare_model()
except:
    accuracy, report = prepare_model()
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    measurement_accuracy = round(accuracy * 100, 2)
    # Create a plot for feature importance or decision boundaries
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'], y=model.feature_importances_)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.savefig('static/feature_importance.png')
    plt.close()
    informative_guide = "The Iris dataset uses four parameters: sepal length, sepal width, petal length, and petal width. These features are crucial in distinguishing between the three Iris species. The visual guide below shows the importance of each feature in the model's decision-making process."
    return render_template('index.html', 
                         accuracy=measurement_accuracy,
                         feature_importance_url='static/feature_importance.png',
                         informative_guide=informative_guide)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = [float(request.form[f]) for f in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        
        # Scale the features
        features_scaled = scaler.transform([features])
        
        # Make prediction and get probabilities
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get the class name and confidence
        iris_classes = ['Setosa', 'Versicolor', 'Virginica']
        result = iris_classes[prediction]
        confidence = round(probabilities[prediction] * 100, 2)
        
        return render_template('index.html', 
                             prediction=result, 
                             confidence=confidence,
                             error=None)
    except Exception as e:
        return render_template('index.html', prediction=None, error='Invalid input. Please enter valid numeric values.')

if __name__ == '__main__':
    app.run(debug=True)
