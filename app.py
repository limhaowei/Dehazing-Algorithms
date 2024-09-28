from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from model import load_model, dehaze_image
from utils import allowed_file, preprocess_image, postprocess_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super_secret_key'  # Required for flashing messages

model = load_model()

@app.route('/', methods=['GET', 'POST'])
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
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            image = Image.open(filepath)
            preprocessed_image = preprocess_image(image)
            dehazed_image = dehaze_image(model, preprocessed_image)
            result_image = postprocess_image(dehazed_image)
            
            # Save the result
            result_filename = 'dehazed_' + filename
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            result_image.save(result_filepath)
            
            return render_template('result.html', original=filename, result=result_filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    
    
    