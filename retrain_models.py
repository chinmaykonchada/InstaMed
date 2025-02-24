# retrain_models.py
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle

def retrain_models():
    # Load and prepare the training data
    training_data = pd.read_csv("dataset/Training.csv")
    
    # Get feature columns (all except prognosis)
    feature_columns = [col for col in training_data.columns if col != 'prognosis']
    
    # Prepare X (features) and y (labels)
    X = training_data[feature_columns]
    y = training_data['prognosis'].values
    
    # Create and fit label encoder first
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Create and train the SVC model with feature names
    svc_model = SVC(kernel='linear', probability=True)
    svc_model.fit(X, y_encoded)
    
    # Save everything
    with open('SVC.pkl', 'wb') as f:
        pickle.dump(svc_model, f)
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
        
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print("Models retrained and saved successfully!")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Available disease labels: {sorted(label_encoder.classes_)}")

if __name__ == "__main__":
    retrain_models()