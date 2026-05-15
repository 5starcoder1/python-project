# ==========================================
# PORTFOLIO WEBSITE BACKEND USING FLASK
# ==========================================

# FEATURES:
# ✅ Contact Form API
# ✅ Project API
# ✅ Skills API
# ✅ Download Resume Route
# ✅ SQLite Database
# ✅ JSON APIs
# ✅ Beginner Friendly

# ------------------------------------------
# INSTALL REQUIRED LIBRARIES
# ------------------------------------------
# pip install flask flask_sqlalchemy flask_cors

# ------------------------------------------
# IMPORTS
# ------------------------------------------

from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime

# ------------------------------------------
# APP CONFIG
# ------------------------------------------

app = Flask(__name__)

CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///portfolio.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ------------------------------------------
# DATABASE MODELS
# ------------------------------------------

class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100))
    description = db.Column(db.Text)
    github = db.Column(db.String(200))

# ------------------------------------------
# HOME ROUTE
# ------------------------------------------

@app.route('/')
def home():
    return jsonify({
        "message": "Portfolio Backend Running 🚀"
    })

# ------------------------------------------
# CONTACT FORM API
# ------------------------------------------

@app.route('/contact', methods=['POST'])
def contact():

    data = request.json

    new_contact = Contact(
        name=data['name'],
        email=data['email'],
        message=data['message']
    )

    db.session.add(new_contact)
    db.session.commit()

    return jsonify({
        "success": True,
        "message": "Message Sent Successfully ✅"
    })

# ------------------------------------------
# GET ALL CONTACTS
# ------------------------------------------

@app.route('/contacts', methods=['GET'])
def get_contacts():

    contacts = Contact.query.all()

    output = []

    for contact in contacts:

        output.append({
            "id": contact.id,
            "name": contact.name,
            "email": contact.email,
            "message": contact.message,
            "created_at": contact.created_at
        })

    return jsonify(output)

# ------------------------------------------
# PROJECT API
# ------------------------------------------

@app.route('/projects', methods=['GET'])
def projects():

    demo_projects = [
        {
            "title": "AI Chat App",
            "description": "AI Powered Chat Application",
            "github": "https://github.com/yourgithub"
        },
        {
            "title": "Portfolio Website",
            "description": "Modern Developer Portfolio",
            "github": "https://github.com/yourgithub"
        },
        {
            "title": "Expense Tracker",
            "description": "Track Daily Expenses",
            "github": "https://github.com/yourgithub"
        }
    ]

    return jsonify(demo_projects)

# ------------------------------------------
# SKILLS API
# ------------------------------------------

@app.route('/skills', methods=['GET'])
def skills():

    skill_list = [
        "Python",
        "Flask",
        "HTML",
        "CSS",
        "JavaScript",
        "React",
        "SQL",
        "GitHub"
    ]

    return jsonify(skill_list)

# ------------------------------------------
# RESUME DOWNLOAD ROUTE
# ------------------------------------------

@app.route('/resume', methods=['GET'])
def resume():

    return send_file(
        'resume.pdf',
        as_attachment=True
    )

# ------------------------------------------
# CREATE DATABASE
# ------------------------------------------

with app.app_context():
    db.create_all()

# ------------------------------------------
# RUN SERVER
# ------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)

"""
    Project Structure
portfolio-backend/
│
├── app.py
├── resume.pdf
└── portfolio.db

"""
