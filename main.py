from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import joblib
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model

app = Flask(__name__)

# Load the trained model, scaler, and PCA object
model = joblib.load('wts/xgb_model_2.joblib')
scaler = joblib.load('wts/scaler.pkl')
pca = joblib.load('wts/optimal_pca.pkl')
class_names = ['Streets', 'Forest', 'Sea', 'Glacier', 'Building', 'Mountains']

def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return img

def extract_features(image):
    model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    img_array = np.expand_dims(image, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            try:
                image_bytes = file.read()
                img_array = np.frombuffer(image_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    raise ValueError("Failed to load the image.")
                
                preprocessed_img = preprocess_image(img)
                features = extract_features(preprocessed_img)
                features_scaled = scaler.transform([features])
                features_pca = pca.transform(features_scaled)
                
                prediction = model.predict(features_pca)
                predicted_class_index = prediction[0]
                predicted_class = class_names[predicted_class_index]
                
                return jsonify({'predicted_class': predicted_class})
            
            except Exception as e:
                return jsonify({'error': 'An error occurred while processing the image.'}), 500
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


# Development of a Flask application for image classification:

# Guide on setting up a Flask application for local image classification:
#    - Install the necessary dependencies using requirements.txt.
#    - Create a new Flask application and define the necessary routes and views.
#    - Load the trained model, scaler, and PCA object using Joblib.
#    - Define functions for preprocessing the uploaded image and extract features.
#    - Create an HTML template for the image upload form.
#    - Handle the image upload in the Flask route, preprocess the image, extract features, apply scaling and PCA, and make predictions using the loaded model.
#    - Return the predicted class as a JSON response.
#    - Run the Flask application locally and access it through a web browser.
