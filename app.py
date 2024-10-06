from flask import Flask
from config import UPLOAD_FOLDER
from views import upload_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super_secret_key'  # Required for flashing messages

# Register routes
app.add_url_rule('/', 'upload_file', upload_file, methods=['GET', 'POST'])

if __name__ == '__main__':
    app.run(debug=True)