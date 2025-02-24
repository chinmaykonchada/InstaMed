# retrain_models.py
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle

def retrain_models():
    # Load the training data
    training_data = pd.read_csv("dataset/Training.csv")
    
    # Get all columns except 'prognosis'
    feature_columns = [col for col in training_data.columns if col != 'prognosis']
    
    # Separate features and target
    X = training_data[feature_columns]  # Include all columns except prognosis
    y = training_data['prognosis']
    
    # Create and train the SVC model
    svc_model = SVC(kernel='linear', probability=True)
    svc_model.fit(X, y)
    
    # Create and fit the label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    
    # Save the models
    with open('SVC.pkl', 'wb') as f:
        pickle.dump(svc_model, f)
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save the current feature columns for future reference
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print("Models retrained and saved successfully!")
    print(f"Number of features in trained model: {X.shape[1]}")
    print(f"Unique diseases: {len(label_encoder.classes_)}")
    print(f"Disease labels: {sorted(label_encoder.classes_)}")

if __name__ == "__main__":
    retrain_models()