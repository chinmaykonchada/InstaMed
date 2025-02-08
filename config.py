# config.py
from dotenv import load_dotenv
import os
from urllib.parse import quote_plus

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DB_USER = 'postgres'
    DB_PASSWORD = quote_plus('chinmay@123')  # URL encode the password
    DB_HOST = 'localhost'
    DB_PORT = '5432'
    DB_NAME = 'instamed'
    
    SQLALCHEMY_DATABASE_URI = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False