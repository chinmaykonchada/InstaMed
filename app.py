from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_migrate import Migrate

app = Flask(__name__)

# Database Configuration
import urllib.parse
password = urllib.parse.quote("chinmay@123")
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://postgres:{password}@localhost/InstaMed'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# User Model
class User(db.Model, UserMixin):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    phone = db.Column(db.String(20), unique=True, nullable=True)
    password = db.Column(db.String(150), nullable=False)

# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        login_input = request.form["loginInput"]
        password = request.form["password"]
        
        # Check if input is email or phone
        user = User.query.filter((User.email == login_input) | (User.phone == login_input)).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash("Login Successful!", "success")
            return redirect(url_for("symptoms"))
        else:
            flash("Invalid credentials, please try again.", "danger")

    return render_template("login.html")

@app.route("/symptoms")
@login_required
def symptoms():
    return render_template("symptoms.html", user=current_user)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
