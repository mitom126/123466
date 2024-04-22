from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import urllib.request
import os
from werkzeug.utils import secure_filename
import re
import subprocess
from src.DB import Database
from src.color import Color
import sys
from src.infer import infer
app = Flask(__name__)

UPLOAD_FOLDER = 'query'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        file.filename='0.jpg'
        filename = secure_filename(file.filename)
        
        querypath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
        print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        depth = 5
        d_type = 'd1'
        query_idx = 0
        query = {'img': None, 'cls': None, 'hist': None}
        print(querypath)
        s1 =  querypath.split('/')
        cls = 'test'
        img = querypath
        print(img)
        print(cls)
        
        method = Color()
        query['img'] = img
        query['cls'] = cls
        query['hist']  = method.histogram(querypath, type='region', n_bin=12, n_slice=3)
        print(query['hist'])

        
        depth = 5
        d_type = 'd1'
        query_idx = 0
        db = Database()
        samples = method.make_samples(db)
        _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
        print(result)
        images = []
        for kq in result:
            image_path = kq['img']
            images.append(image_path)

        return render_template('index.html', filename=filename, query=querypath ,
                            list_img = images)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('database', filename='beaches/' + filename), code=301)

@app.route('/database/<path:filename>')
def database(filename):
    return send_from_directory('database', filename)

@app.route('/query/<path:filename>')
def query(filename):
    return send_from_directory('query', filename)


if __name__ == "__main__":
    app.run()