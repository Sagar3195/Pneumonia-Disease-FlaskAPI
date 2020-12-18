import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import *
from werkzeug.utils import secure_filename

##Define the flask app
app = Flask(__name__)

##Load trained model
model = load_model("pneumonia_model_vgg19.h5")

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224,224))

    #Preprocessing the image
    x = image.img_to_array(img)
    #scaling
    x = x/255.0
    x = np.expand_dims(x, axis = 0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis = 1)
    if preds == 0:
        preds = "Great!!!, You don't have Pneumonia Disease."
    else:
        preds = "Sorry, You have Pneumonia disease, kindly contact your doctor."

    return preds

@app.route('/', methods = ['GET'])
def index():
    #Main Page
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        ##Get the file from post request
        f = request.files['file']
        #save the file ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        #make predictions
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None

if __name__ == '__main__':
    app.run(port = 5001, debug= True)
