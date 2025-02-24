from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from datetime import datetime
from models import Admin, Doctor, NewDisease, db, User
from config import Config
from flask import jsonify
from ml_model import DiseasePredictor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import pickle
from sklearn.svm import SVC



app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize the predictor
disease_predictor = DiseasePredictor()

@login_manager.user_loader
def load_user(user_id):
    # Check if user is an admin
    admin = db.session.get(Admin, int(user_id))
    if admin:
        return admin

    # Check if user is a doctor
    doctor = Doctor.query.get(int(user_id))
    if doctor:
        return doctor

    # Check if user is a regular user
    return User.query.get(int(user_id))


@app.route('/')
def index():
    return render_template('landingpage.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            # Get form data
            full_name = request.form.get('name')
            email = request.form.get('email')
            phone = request.form.get('phone')
            gender = request.form.get('gender')
            dob_str = request.form.get('dob')
            password = request.form.get('password')

            # Validate required fields
            if not all([full_name, email, phone, gender, dob_str, password]):
                flash('All fields are required')
                return redirect(url_for('signup'))

            try:
                dob = datetime.strptime(dob_str, '%Y-%m-%d')
            except ValueError:
                flash('Invalid date format')
                return redirect(url_for('signup'))

            # Check if user already exists
            if User.query.filter_by(email=email).first():
                flash('Email already exists')
                return redirect(url_for('signup'))

            if User.query.filter_by(phone=phone).first():
                flash('Phone number already exists')
                return redirect(url_for('signup'))

            # Create new user
            user = User(
                full_name=full_name,
                email=email,
                phone=phone,
                gender=gender,
                dob=dob
            )
            user.set_password(password)
            
            db.session.add(user)
            db.session.commit()
            
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
            
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration')
            print(f"Error: {str(e)}")  # For debugging
            return redirect(url_for('signup'))
        
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_input = request.form.get('loginInput')
        password = request.form.get('password')
        
        # Check if login input is email or phone
        user = User.query.filter((User.email == login_input) | 
                               (User.phone == login_input)).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('symptoms'))
            
        flash('Invalid credentials')
        return redirect(url_for('login'))
        
    return render_template('login.html')

# app.py routes

@app.route('/symptoms')
@login_required
def symptoms():
    all_symptoms = disease_predictor.get_all_symptoms()
    return render_template('symptoms.html', symptoms=all_symptoms)

@app.route('/get_symptoms')
def get_symptoms():
    """API endpoint to get all symptoms"""
    all_symptoms = disease_predictor.get_all_symptoms()
    return jsonify(all_symptoms)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms', '').split(',')
        if not symptoms or symptoms[0] == '':
            return jsonify({'error': 'No symptoms provided'}), 400
        
        try:
            # Clean symptoms
            symptoms = [s.strip().lower().replace(' ', '_') for s in symptoms if s.strip()]
            
            # Get prediction and render template
            result = disease_predictor.process_symptoms(symptoms)
            
            # Only return error response if there's actually an error
            if isinstance(result, dict) and 'error' in result:
                return jsonify(result), 500
                
            return render_template(
                'prediction.html',
                result=result
            )
            
        except Exception as e:
            # Log the error for debugging
            print(f"Prediction error: {str(e)}")
            return jsonify({'error': 'An error occurred during prediction. Please try again.'}), 500
            
# Add these routes to app.py
# Doctor
from flask import flash, redirect, url_for, request, render_template
from functools import wraps

def doctor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not isinstance(current_user, Doctor):
            flash('Please log in as a doctor to access this page.')
            return redirect(url_for('doctor_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/doctor/login', methods=['GET', 'POST'])
def doctor_login():
    if current_user.is_authenticated:
        if isinstance(current_user, Doctor):
            return redirect(url_for('doctor_dashboard'))
        logout_user()
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        doctor = Doctor.query.filter_by(email=email).first()
        if doctor and doctor.check_password(password):
            if not doctor.is_verified:
                flash('Your account is pending verification.')
                return redirect(url_for('doctor_login'))
            login_user(doctor)
            return redirect(url_for('doctor_dashboard'))
        
        flash('Invalid credentials')
    return render_template('doctor/login.html')

@app.route('/doctor/signup', methods=['GET', 'POST'])
def doctor_signup():
    if request.method == 'POST':
        try:
            doctor = Doctor(
                full_name=request.form.get('name'),
                email=request.form.get('email'),
                phone=request.form.get('phone'),
                registration_number=request.form.get('registration_number'),
                specialization=request.form.get('specialization')
            )
            doctor.set_password(request.form.get('password'))
            
            db.session.add(doctor)
            db.session.commit()
            
            flash('Registration successful! Please wait for account verification.')
            return redirect(url_for('doctor_login'))
        except Exception as e:
            print(str(e))  # For debugging
            db.session.rollback()
            flash('Registration failed. Please try again.')
            
    return render_template('doctor/signup.html')

@app.route('/doctor/dashboard')
@doctor_required
def doctor_dashboard():
    diseases = NewDisease.query.filter_by(doctor_id=current_user.id).all()
    return render_template('doctor/dashboard.html', diseases=diseases)

@app.route('/doctor/add-disease', methods=['GET', 'POST'])
@doctor_required
def add_disease():
    if request.method == 'POST':
        try:
            disease = NewDisease(
                name=request.form.get('name'),
                description=request.form.get('description'),
                symptoms=request.form.get('symptoms'),
                medications=request.form.get('medications'),
                precautions=request.form.get('precautions'),
                diet=request.form.get('diet'),
                workout=request.form.get('workout'),
                doctor_id=current_user.id
            )
            db.session.add(disease)
            db.session.commit()
            flash('Disease information submitted successfully!')
            return redirect(url_for('doctor_dashboard'))
        except Exception as e:
            db.session.rollback()
            flash('Error submitting disease information.')
    
    return render_template('doctor/add_disease.html')
# Add these routes to app.py

@app.route('/doctor/disease/<int:disease_id>')
@doctor_required
def view_disease(disease_id):
    disease = NewDisease.query.get_or_404(disease_id)
    if disease.doctor_id != current_user.id:
        flash('You do not have permission to view this disease.')
        return redirect(url_for('doctor_dashboard'))
    return render_template('doctor/view_disease.html', disease=disease)

@app.route('/doctor/disease/edit/<int:disease_id>', methods=['GET', 'POST'])
@doctor_required
def edit_disease(disease_id):
    disease = NewDisease.query.get_or_404(disease_id)
    if disease.doctor_id != current_user.id:
        flash('You do not have permission to edit this disease.')
        return redirect(url_for('doctor_dashboard'))
    
    if request.method == 'POST':
        try:
            disease.name = request.form.get('name')
            disease.description = request.form.get('description')
            disease.symptoms = request.form.get('symptoms')
            disease.medications = request.form.get('medications')
            disease.precautions = request.form.get('precautions')
            disease.diet = request.form.get('diet')
            disease.workout = request.form.get('workout')
            
            db.session.commit()
            flash('Disease information updated successfully!')
            return redirect(url_for('doctor_dashboard'))
        except Exception as e:
            db.session.rollback()
            flash('Error updating disease information.')
    
    return render_template('doctor/edit_disease.html', disease=disease)

@app.route('/doctor/retrain-model')
@doctor_required
def retrain_model():
    try:
        # Get all approved diseases
        approved_diseases = NewDisease.query.filter_by(status='approved').all()
        
        # Update the training data
        if update_training_data(approved_diseases):
            # Validate the data
            if validate_training_data():
                # Retrain the model
                retrain_ml_model()
                flash('Model retrained successfully with new disease data!')
            else:
                flash('Error validating training data.')
        else:
            flash('Error updating training data.')
    except Exception as e:
        flash(f'Error retraining model: {str(e)}')
    
    return redirect(url_for('doctor_dashboard'))

def update_training_data(new_diseases):
    """
    Update the training dataset with new disease information
    """
    try:
        # Load existing training data
        data = pd.read_csv("dataset/Training.csv")
        
        # Create list of all symptom columns (excluding special columns)
        symptom_columns = [col for col in data.columns if col not in ['prognosis', 'a', 'test1', 'test3']]
        
        new_rows = []
        for disease in new_diseases:
            # Initialize all symptoms to 0
            new_row = dict.fromkeys(data.columns, 0)
            
            # Set symptoms from disease to 1
            symptoms_list = [s.strip().lower().replace(' ', '_') for s in disease.symptoms.split(',')]
            for symptom in symptoms_list:
                if symptom in symptom_columns:
                    new_row[symptom] = 1.0  # Use 1.0 for consistency
            
            # Set disease name
            new_row['prognosis'] = disease.name
            
            new_rows.append(new_row)
        
        # Convert new rows to DataFrame
        new_data = pd.DataFrame(new_rows)
        
        # Ensure all columns match original data
        new_data = new_data[data.columns]
        
        # Concatenate and fill any remaining NaN with 0
        data = pd.concat([data, new_data], ignore_index=True)
        data = data.fillna(0)
        
        # Convert all numeric columns to float
        for col in data.columns:
            if col != 'prognosis':
                data[col] = data[col].astype(float)
        
        # Save the cleaned dataset
        data.to_csv("dataset/Training.csv", index=False)
        print(f"Successfully added {len(new_diseases)} new disease entries to training data")
        return True
        
    except Exception as e:
        print(f"Error updating training data: {str(e)}")
        return False
    
def validate_training_data():
    """
    Validate the training data format and content
    """
    try:
        data = pd.read_csv("dataset/Training.csv")
        
        # Check for empty columns
        empty_cols = data.columns[data.isna().all()].tolist()
        if empty_cols:
            print(f"Warning: Found completely empty columns: {empty_cols}")
        
        # Check for rows with all zeros
        zero_rows = (data.select_dtypes(include=[np.number]) == 0).all(axis=1).sum()
        if zero_rows > 0:
            print(f"Warning: Found {zero_rows} rows with all zeros")
        
        # Verify data types
        for col in data.columns:
            if col != 'prognosis':
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Save cleaned data
        data.to_csv("dataset/Training.csv", index=False)
        print("Training data validated and cleaned")
        
        return True
    except Exception as e:
        print(f"Error validating training data: {str(e)}")
        return False

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def retrain_ml_model():
    """
    Retrain the ML model with updated dataset
    """
    try:
        # Load data
        data = pd.read_csv("dataset/Training.csv")
        
        # Clean data
        for col in data.columns:
            if col != 'prognosis':
                # Convert to numeric and fill NaN with 0
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(data['prognosis'])
        X = data.drop(['prognosis'], axis=1)
        
        # Create pipeline
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='linear'))
        ])
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        pipeline.fit(X_train, y_train)
        
        # Save model
        with open('SVC.pkl', 'wb') as f:
            pickle.dump(pipeline, f)
            
        print("Model retrained successfully")
        return True
        
    except Exception as e:
        print(f"Error in retraining model: {str(e)}")
        return False
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.')
    return redirect(url_for('index'))

@app.route('/doctor/logout')
@doctor_required
def doctor_logout():
    logout_user()
    flash('You have been logged out successfully.')
    return redirect(url_for('doctor_login'))

# Add these imports at the top of your app.py
from flask_login import login_user, logout_user, login_required, current_user
from functools import wraps

# Add this decorator for admin authentication
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not isinstance(current_user, Admin):
            flash('Please log in as an admin to access this page.')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# Update or add these routes
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated and isinstance(current_user, Admin):
        return redirect(url_for('admin_dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        admin = Admin.query.filter_by(username=username).first()
        
        if admin and admin.check_password(password):
            login_user(admin)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid username or password', 'error')
            
    return render_template('admin/login.html')

@app.route('/admin/signup', methods=['GET', 'POST'])
def admin_signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if Admin.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('admin_signup'))
            
        if Admin.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('admin_signup'))
            
        admin = Admin(
            username=username,
            email=email
        )
        admin.set_password(password)
        
        try:
            db.session.add(admin)
            db.session.commit()
            flash('Admin account created successfully!', 'success')
            return redirect(url_for('admin_login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration', 'error')
            print(f"Error: {str(e)}")
            
    return render_template('admin/signup.html')

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    # Fetch pending doctor verifications
    pending_doctors = Doctor.query.filter_by(is_verified=False).all()
    
    # Fetch pending disease reviews
    pending_diseases = NewDisease.query.filter_by(status='pending').all()
    
    return render_template('admin/dashboard.html', 
                          pending_doctors=pending_doctors,
                          pending_diseases=pending_diseases)

@app.route('/admin/logout')
@login_required
def admin_logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('admin_login'))
@app.route('/admin/verify-doctor/<int:doctor_id>/<action>')
@admin_required
def verify_doctor(doctor_id, action):
    doctor = Doctor.query.get_or_404(doctor_id)
    
    if action == 'approve':
        doctor.is_verified = True
        db.session.commit()
        # Send email notification to doctor
        flash(f'Doctor {doctor.full_name} has been verified.')
    elif action == 'reject':
        db.session.delete(doctor)
        db.session.commit()
        flash(f'Doctor {doctor.full_name} has been rejected.')
        
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/review-disease/<int:disease_id>/<action>')
@admin_required
def review_disease(disease_id, action):
    disease = NewDisease.query.get_or_404(disease_id)
    
    if action == 'approve':
        disease.status = 'approved'
        # Here you would add code to update your ML model and datasets
        update_ml_model_with_new_disease(disease)
        flash(f'Disease {disease.name} has been approved and added to the system.')
    elif action == 'reject':
        disease.status = 'rejected'
        flash(f'Disease {disease.name} has been rejected.')
    
    db.session.commit()
    return redirect(url_for('admin_dashboard'))

def update_ml_model_with_new_disease(disease):
    """Update the ML model with new disease data, including handling new symptoms"""
    try:
        # Load existing datasets
        symptoms_df = pd.read_csv("dataset/Training.csv")
        desc_df = pd.read_csv("dataset/description.csv")
        medications_df = pd.read_csv("dataset/medications.csv")
        precautions_df = pd.read_csv("dataset/precautions_df.csv")
        diets_df = pd.read_csv("dataset/diets.csv")
        
        # Get current symptoms (excluding 'prognosis')
        existing_symptoms = [col for col in symptoms_df.columns if col != 'prognosis']
        
        # Process new symptoms list
        new_symptoms_list = [s.strip() for s in disease.symptoms.split(',') if s.strip()]
        
        # Identify truly new symptoms (not in existing columns)
        new_symptoms = [symptom for symptom in new_symptoms_list 
                       if symptom not in existing_symptoms]
        
        # Add new symptom columns if any
        for new_symptom in new_symptoms:
            if new_symptom not in symptoms_df.columns:
                symptoms_df[new_symptom] = 0  # Initialize with zeros for all existing rows
                print(f"Added new symptom column: {new_symptom}")
        
        # Get updated list of all symptoms
        all_symptoms = [col for col in symptoms_df.columns if col != 'prognosis']
        
        # Create new row with all symptoms initialized to 0
        new_symptom_row = {symptom: 0 for symptom in all_symptoms}
        
        # Set 1 for present symptoms
        for symptom in new_symptoms_list:
            if symptom in new_symptom_row:
                new_symptom_row[symptom] = 1
            else:
                print(f"Warning: Symptom '{symptom}' not found in symptoms list")
        
        # Add prognosis (disease name)
        new_symptom_row['prognosis'] = disease.name
        
        # Add to symptoms dataset
        new_row_df = pd.DataFrame([new_symptom_row])
        # Ensure column order matches
        new_row_df = new_row_df[symptoms_df.columns]
        symptoms_df = pd.concat([symptoms_df, new_row_df], ignore_index=True)
        
        # Add to description dataset if not exists
        if not desc_df['Disease'].str.contains(disease.name).any():
            desc_df = pd.concat([desc_df, pd.DataFrame([{
                'Disease': disease.name,
                'Description': disease.description
            }])], ignore_index=True)
        
        # Add to medications dataset if not exists
        if not medications_df['Disease'].str.contains(disease.name).any():
            medications_df = pd.concat([medications_df, pd.DataFrame([{
                'Disease': disease.name,
                'Medication': disease.medications
            }])], ignore_index=True)
        
        # Add to precautions dataset if not exists
        if not precautions_df['Disease'].str.contains(disease.name).any():
            precautions_list = [p.strip() for p in disease.precautions.split(',') if p.strip()]
            precautions_df = pd.concat([precautions_df, pd.DataFrame([{
                'Disease': disease.name,
                'Precaution_1': precautions_list[0] if len(precautions_list) > 0 else 'None',
                'Precaution_2': precautions_list[1] if len(precautions_list) > 1 else 'None',
                'Precaution_3': precautions_list[2] if len(precautions_list) > 2 else 'None',
                'Precaution_4': precautions_list[3] if len(precautions_list) > 3 else 'None'
            }])], ignore_index=True)
        
        # Add to diets dataset if not exists
        if not diets_df['Disease'].str.contains(disease.name).any():
            diets_df = pd.concat([diets_df, pd.DataFrame([{
                'Disease': disease.name,
                'Diet': disease.diet if disease.diet else 'No specific diet recommendations'
            }])], ignore_index=True)
        
        # Save updated datasets
        symptoms_df.to_csv("dataset/Training.csv", index=False)
        desc_df.to_csv("dataset/description.csv", index=False)
        medications_df.to_csv("dataset/medications.csv", index=False)
        precautions_df.to_csv("dataset/precautions_df.csv", index=False)
        diets_df.to_csv("dataset/diets.csv", index=False)
        
        # Retrain the model
        retrain_model()
        
    except Exception as e:
        print(f"Error updating ML model: {str(e)}")
        raise

def retrain_model():
    """Retrain the ML model with updated data"""
    try:
        # Load the updated training data
        data = pd.read_csv("dataset/Training.csv")
        
        # Ensure no missing values
        data = data.fillna(0)  # Fill NaN values with 0
        
        # Preprocess the data
        le = LabelEncoder()
        y = le.fit_transform(data['prognosis'])
        X = data.drop(['prognosis'], axis=1)
        
        # Handle any remaining missing values
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train the model
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        
        # Save the updated model and label encoder
        with open('SVC.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)
            
    except Exception as e:
        print(f"Error retraining model: {str(e)}")
        raise

def get_all_symptoms():
    """Helper function to get list of all symptoms from the training data"""
    try:
        symptoms_df = pd.read_csv("dataset/Training.csv")
        return [col for col in symptoms_df.columns if col != 'prognosis']
    except Exception as e:
        print(f"Error getting symptoms list: {str(e)}")
        return []
    
def validate_new_symptom(symptom):
    """Validate a new symptom before adding it to the dataset"""
    # Remove leading/trailing whitespace
    symptom = symptom.strip()
    
    # Check if symptom name is not empty
    if not symptom:
        return False, "Symptom name cannot be empty"
    
    # Check if symptom name contains only valid characters
    if not symptom.replace(' ', '_').replace('-', '_').isalnum():
        return False, "Symptom name can only contain letters, numbers, spaces, and hyphens"
    
    # Check length
    if len(symptom) > 50:
        return False, "Symptom name is too long (max 50 characters)"
    
    return True, "Valid symptom name"

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=True, reloader_type='stat')
    