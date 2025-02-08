from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from datetime import datetime
from models import db, User
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
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

@app.route('/symptoms')
@login_required
def symptoms():
    return render_template('symptoms.html')

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True)
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=True, reloader_type='stat')
    