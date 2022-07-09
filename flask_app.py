from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from preproccesing import preprocess_image, get_class_name, prepare_to_predict
import ast
import matplotlib
import tensorflow.keras
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def allowed_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1] in ALLOWED_EXTENSIONS


with open('model/plant_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('classes_dict') as class_file:
    class_dict = class_file.read()
    class_dict = ast.literal_eval(class_dict)

app = Flask(__name__)


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('index.html', prediction='No posted image')
    file = request.files['image']

    if file.filename == '':
        return render_template("index.html", prediction="You didn't selected the image")

    if file and allowed_files(file.filename):
        filename = secure_filename(file.filename)
        img = preprocess_image(file)
        img_predict = prepare_to_predict(image=img)
        prediction = model.predict(img_predict).argmax(1)
        prediction_class = get_class_name(predictions=prediction, class_dictionary=class_dict)
        return render_template('index.html', prediction='The predicted class for {0} is {1}'.format(filename,
                              prediction_class))
    else:
        return render_template('index.html', prediction='Invalid file extension. Support only for png, jpg, jpeg, webp')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)