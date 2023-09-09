import os
import scipy
from flask import Flask, request, render_template
from PIL import Image
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import save_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


UPLOAD_FOLDER = os.getcwd() + '/static/uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
img_height = 180
img_width = 180

@app.route('/')
def home():
    return render_template('index.html')
# Route for handling file uploads and predictions
@app.route('/result', methods=['GET', 'POST'])
def predict():
    if 'file' not in request.files:
        return "No file part."

    file = request.files['file']
    if file.filename == '':
        return "No selected file."
        
    file.save(os.path.join(app.config['UPLOAD_FOLDER'],'temp.png'))
    try:
        model_mango = load_model('model.h5')

        print("Processed Incoming Image")

        image_gen_test = ImageDataGenerator(rescale = 1./255)
        test_data_gen = image_gen_test.flow_from_directory(batch_size=20,directory= 'static/',shuffle=True,target_size=(img_height, img_width))
        predictions = model_mango.predict(test_data_gen)
        for i in range(len(predictions)):
            if predictions[i][0] < 0:
                    return render_template('index.html',img_name='temp.png',status = 'Normal')
            else:
                    return render_template('index.html',img_name='temp.png', status = 'Knock Knees')
    except Exception as e:
        return "Error: {}".format(str(e))

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
