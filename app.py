from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import urllib.request
import os

from werkzeug.utils import secure_filename
import re
import subprocess
from src.DB import Database
from src.vggnet import VGGNetFeat
from src.vggnet import VGGNet
import sys
from src.infer import infer

#Test
import numpy as np
import imageio
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG
use_gpu = torch.cuda.is_available()
#Test
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
        # configs for histogram
        VGG_model  = 'vgg19'  # model type
        pick_layer = 'avg'    # extract feature of this layer
        d_type     = 'd1'     # distance type
        depth      = 8
        #end config 
        query_idx = 0
        query = {'img': None, 'cls': None, 'hist': None}
        print(querypath)
        s1 =  querypath.split('/')
        d_cls = 'test'
        d_img = querypath
        print(d_img)
        print(d_cls)
        
        method = VGGNetFeat()
        query['img'] = d_img
        query['cls'] = d_cls
        #Count hist
        means = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
        vgg_model = VGGNet(requires_grad=False, model=VGG_model)
        vgg_model.eval()
        if use_gpu:
            vgg_model = vgg_model.cuda()
        img = imageio.imread(d_img, pilmode="RGB")
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= means[0]  # reduce B's mean
        img[1] -= means[1]  # reduce G's mean
        img[2] -= means[2]  # reduce R's mean
        img = np.expand_dims(img, axis=0)
        try:
          if use_gpu:
            inputs = torch.autograd.Variable(torch.from_numpy(img).cuda().float())
          else:
            inputs = torch.autograd.Variable(torch.from_numpy(img).float())
          d_hist = vgg_model(inputs)[pick_layer]
          d_hist = np.sum(d_hist.data.cpu().numpy(), axis=0)
          d_hist /= np.sum(d_hist)  # normalize
          query['hist']  = d_hist
        except:
          pass
       
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

@app.route('/css/<path:filename>')
def css(filename):
    return send_from_directory('css', filename)

@app.route('/js/<path:filename>')
def js(filename):
    return send_from_directory('js', filename)

@app.route('/img/<path:filename>')
def img(filename):
    return send_from_directory('templates/img', filename)

# @app.route('/fontawesome/css/<path:filename>')
# def fontawesome(filename):
#     return send_from_directory('fontawesome/css', filename)

if __name__ == "__main__":
    app.run()