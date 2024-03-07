import os
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(result):
    # Assuming result is a string containing the class name
    if result.lower() == "yes brain tumor":
        return "Brain Tumor detected"
    else:
        return "Brain Tumor detected"


def getResult(img_path, threshold=0.5):
    image = cv2.imread(img_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)

    if result.shape[1] == 1:
        # Binary classification case
        return "Yes Brain Tumor" if result[0][0] > threshold else "No Brain Tumor"
    else:
        # Multi-class classification case
        return get_className(result[0])


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Get the raw result from the model
        raw_result = getResult(file_path)
        # Process the result
        if "Brain Tumor detected" in raw_result:
            result = "Brain Tumor detected"
        else:
            result = "No Brain Tumor detected"

    return result



if __name__ == '__main__':
    app.run(debug=True)
