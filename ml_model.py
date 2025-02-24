# ml_model.py
import pickle
import numpy as np
import pandas as pd

class DiseasePredictor:
    def __init__(self):
        # Load the model and necessary data
        self.model = pickle.load(open('SVC.pkl', 'rb'))
        self.training_data = pd.read_csv("dataset/Training.csv")
        
        # Load feature columns
        with open('feature_columns.pkl', 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        # Create symptoms dictionary dynamically
        self.symptoms_dict = {col: idx for idx, col in enumerate(self.feature_columns)}
        
        # Load disease encoder
        self.label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        
        # Load additional data
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
        input_vector = np.zeros(len(self.feature_columns))
        for symptom in symptoms:
            symptom_key = symptom.lower().replace(' ', '_')
            if symptom_key in self.symptoms_dict:
                input_vector[self.symptoms_dict[symptom_key]] = 1
        return input_vector

    def predict_disease(self, symptoms):
        """Predict disease based on symptoms"""
        input_vector = self.preprocess_symptoms(symptoms)
        prediction = self.model.predict([input_vector])[0]
        disease = self.label_encoder.inverse_transform([prediction])[0]
        return disease

    def get_disease_details(self, disease):
        """Get all details for a predicted disease"""
        if disease not in self.description['Disease'].values:
            return {
                'disease': disease,
                'description': "Description not available",
                'precautions': ["No specific precautions available"],
                'medications': ["No specific medications available"],
                'diets': ["No specific diet recommendations available"],
                'workouts': ["No specific workout recommendations available"]
            }
            
        description = self.description[self.description['Disease'] == disease]['Description'].iloc[0]
        
        precautions = (self.precautions[self.precautions['Disease'] == disease].iloc[0][1:].tolist() 
                      if disease in self.precautions['Disease'].values else ["No specific precautions available"])
        
        medications = (self.medications[self.medications['Disease'] == disease]['Medication'].tolist()
                      if disease in self.medications['Disease'].values else ["No specific medications available"])
        
        diets = (self.diets[self.diets['Disease'] == disease]['Diet'].tolist()
                if disease in self.diets['Disease'].values else ["No specific diet recommendations available"])
        
        workouts = (self.workout[self.workout['disease'] == disease]['workout'].tolist()
                   if disease in self.workout['disease'].values else ["No specific workout recommendations available"])

        return {
            'disease': disease,
            'description': description,
            'precautions': precautions,
            'medications': medications,
            'diets': diets,
            'workouts': workouts
        }

    def process_symptoms(self, symptoms):
        """Main function to process symptoms and return all disease information"""
        predicted_disease = self.predict_disease(symptoms)
        return self.get_disease_details(predicted_disease)