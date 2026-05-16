# =======================================
# BLOG CMS USING FLASK + SQLITE
# =======================================

# FEATURES:
# ✅ Create Blog
# ✅ View All Blogs
# ✅ Single Blog Page
# ✅ Delete Blog
# ✅ SQLite Database
# ✅ REST API
# ✅ Beginner Friendly

# ---------------------------------------
# INSTALL REQUIRED LIBRARIES
# ---------------------------------------
# pip install flask flask_sqlalchemy flask_cors

# ---------------------------------------
# IMPORTS
# ---------------------------------------

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime

# ---------------------------------------
# APP CONFIG
# ---------------------------------------

app = Flask(__name__)

CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------------------------------------
# DATABASE MODEL
# ---------------------------------------

class Blog(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    title = db.Column(db.String(200), nullable=False)

    content = db.Column(db.Text, nullable=False)

    author = db.Column(db.String(100), nullable=False)

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

# ---------------------------------------
# HOME ROUTE
# ---------------------------------------

@app.route('/')
def home():

    return jsonify({
        "message": "🚀 Blog CMS Backend Running"
    })

# ---------------------------------------
# CREATE BLOG
# ---------------------------------------

@app.route('/create-blog', methods=['POST'])
def create_blog():

    data = request.json

    new_blog = Blog(
        title=data['title'],
        content=data['content'],
        author=data['author']
    )

    db.session.add(new_blog)
    db.session.commit()

    return jsonify({
        "success": True,
        "message": "✅ Blog Created Successfully"
    })

# ---------------------------------------
# GET ALL BLOGS
# ---------------------------------------

@app.route('/blogs', methods=['GET'])
def get_blogs():

    blogs = Blog.query.order_by(
        Blog.created_at.desc()
    ).all()

    output = []

    for blog in blogs:

        output.append({
            "id": blog.id,
            "title": blog.title,
            "content": blog.content,
            "author": blog.author,
            "created_at": blog.created_at
        })

    return jsonify(output)

# ---------------------------------------
# GET SINGLE BLOG
# ---------------------------------------

@app.route('/blog/<int:id>', methods=['GET'])
def single_blog(id):

    blog = Blog.query.get_or_404(id)

    return jsonify({
        "id": blog.id,
        "title": blog.title,
        "content": blog.content,
        "author": blog.author,
        "created_at": blog.created_at
    })

# ---------------------------------------
# DELETE BLOG
# ---------------------------------------

@app.route('/delete-blog/<int:id>', methods=['DELETE'])
def delete_blog(id):

    blog = Blog.query.get_or_404(id)

    db.session.delete(blog)

    db.session.commit()

    return jsonify({
        "success": True,
        "message": "❌ Blog Deleted Successfully"
    })

# ---------------------------------------
# UPDATE BLOG
# ---------------------------------------

@app.route('/update-blog/<int:id>', methods=['PUT'])
def update_blog(id):

    blog = Blog.query.get_or_404(id)

    data = request.json

    blog.title = data['title']
    blog.content = data['content']
    blog.author = data['author']

    db.session.commit()

    return jsonify({
        "success": True,
        "message": "✏ Blog Updated Successfully"
    })

# ---------------------------------------
# CREATE DATABASE
# ---------------------------------------

with app.app_context():
    db.create_all()

# ---------------------------------------
# RUN SERVER
# ---------------------------------------

if __name__ == '__main__':
    app.run(debug=True)


"""
blog-cms/
│
├── app.py
└── blog.db

"""
