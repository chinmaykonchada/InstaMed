# ml_model.py
import pickle
import numpy as np
import pandas as pd

class DiseasePredictor:
    def __init__(self):
        # Load all saved components
        with open('SVC.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        with open('label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        with open('feature_columns.pkl', 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        # Load reference data
        self.description = pd.read_csv("dataset/description.csv")
        self.precautions = pd.read_csv("dataset/precautions_df.csv")
        self.medications = pd.read_csv("dataset/medications.csv")
        self.diets = pd.read_csv("dataset/diets.csv")
        self.workout = pd.read_csv("dataset/workout_df.csv")

    def get_all_symptoms(self):
        """Return list of all symptoms for front-end"""
        return sorted(self.feature_columns)

    def preprocess_symptoms(self, symptoms):
        """Convert symptoms list to model input vector"""
        # Create a DataFrame with the correct feature names
        input_data = pd.DataFrame(0, index=[0], columns=self.feature_columns)
        
        for symptom in symptoms:
            symptom_key = symptom.lower().replace(' ', '_')
            if symptom_key in self.feature_columns:
                input_data.loc[0, symptom_key] = 1
                
        return input_data

    def predict_disease(self, symptoms):
        """Predict disease based on symptoms"""
        try:
            # Preprocess symptoms into correct format
            input_vector = self.preprocess_symptoms(symptoms)
            
            # Get prediction
            prediction_encoded = self.model.predict(input_vector)[0]
            
            # Decode prediction
            disease = self.label_encoder.inverse_transform([prediction_encoded])[0]
            return disease
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise

    def get_disease_details(self, disease):
        """Get all details for a predicted disease"""
        try:
            details = {
                'disease': disease,
                'description': "No description available",
                'precautions': ["No specific precautions available"],
                'medications': ["No specific medications available"],
                'diets': ["No specific diet recommendations available"],
                'workouts': ["No specific workout recommendations available"]
            }
            
            # Update with available information
            if disease in self.description['Disease'].values:
                details['description'] = self.description[
                    self.description['Disease'] == disease]['Description'].iloc[0]
            
            if disease in self.precautions['Disease'].values:
                details['precautions'] = self.precautions[
                    self.precautions['Disease'] == disease].iloc[0][1:].tolist()
            
            if disease in self.medications['Disease'].values:
                details['medications'] = self.medications[
                    self.medications['Disease'] == disease]['Medication'].tolist()
            
            if disease in self.diets['Disease'].values:
                details['diets'] = self.diets[
                    self.diets['Disease'] == disease]['Diet'].tolist()
            
            if disease in self.workout['disease'].values:
                details['workouts'] = self.workout[
                    self.workout['disease'] == disease]['workout'].tolist()
            
            return details
            
        except Exception as e:
            print(f"Error getting disease details: {str(e)}")
            raise

    def process_symptoms(self, symptoms):
        """Main function to process symptoms and return all disease information"""
        disease = self.predict_disease(symptoms)
        return self.get_disease_details(disease)