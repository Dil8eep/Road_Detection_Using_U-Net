from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Create upload and result directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Define your loss and metric functions
def total_loss(y_true, y_pred): pass  # Replace with your actual function
def jacard_coef(y_true, y_pred): pass  # Replace with your actual function

# Load the model
model_path = "satellite_standard_unet_100epochs.hdf5"
custom_objects = {
    "dice_loss_plus_1focal_loss": total_loss,
    "jacard_coef": jacard_coef
}
model = load_model(model_path, custom_objects=custom_objects)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess image
        img = Image.open(file_path).convert("RGB")
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_input = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_input)
        predicted_mask = np.argmax(prediction, axis=3)[0]

        # Save the predicted mask
        plt.imsave(os.path.join(RESULT_FOLDER, 'mask.png'), predicted_mask)

        return render_template('result.html', 
                               input_image=file_path, 
                               output_image=os.path.join(RESULT_FOLDER, 'mask.png'))

if __name__ == '__main__':
    app.run(debug=True)
