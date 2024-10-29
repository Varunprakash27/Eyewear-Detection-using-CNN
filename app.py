from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

app = Flask(__name__)
model = None
model_path = r'C:\Users\jmvar\Downloads\faces-spring-2020\Untitled1.ipynb'

def load_my_model():
    global model
    model = load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        img = image.load_img(f, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        result = model.predict(img_array)
        prediction = 'No Glasses' if result[0][0] == 1 else 'Glasses'
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    load_my_model()
    app.run(debug=True)
