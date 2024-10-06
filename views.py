from flask import render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from model import dehaze_image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Process the image
            result_image = dehaze_image(filepath)
            
            # Save the result
            result_filename = 'dehazed_' + filename
            result_filepath = os.path.join(UPLOAD_FOLDER, result_filename)
            result_image.save(result_filepath)
            
            return render_template('result.html', original=filename, result=result_filename)
    return render_template('index.html')