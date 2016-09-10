import os
import time
import skimage.io as skio
import numpy as np
import matplotlib.pyplot as plt
import classifier 

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug import secure_filename
from cStringIO import StringIO

# our model
model = classifier.get_model()
model.load('model.ckpt')

# Initialize the Flask application
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            filename = secure_filename(file.filename)
            file.save(filepath)

            img = skio.imread(filepath)
            img = classifier.normalize(img)
            img = classifier.describe(img)

            result = model.predict([img])

            return jsonify(result)

    return render_template('index.html')

if __name__ == '__main__':

    app.run(
        host="0.0.0.0",
        port=int("8080"),
        debug=True
    )

